//===-- lib/Semantics/rewrite-directives.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "rewrite-directives.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include <list>

namespace Fortran::semantics {

using namespace parser::literals;

class DirectiveRewriteMutator {
public:
  explicit DirectiveRewriteMutator(SemanticsContext &context)
      : context_{context} {}

  // Default action for a parse tree node is to visit children.
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

protected:
  SemanticsContext &context_;
};

// Rewrite atomic constructs to add an explicit memory ordering to all that do
// not specify it, honoring in this way the `atomic_default_mem_order` clause of
// the REQUIRES directive.
class OmpRewriteMutator : public DirectiveRewriteMutator {
public:
  explicit OmpRewriteMutator(SemanticsContext &context)
      : DirectiveRewriteMutator(context) {}

  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

  bool Pre(parser::OpenMPAtomicConstruct &);
  bool Pre(parser::OpenMPRequiresConstruct &);
  bool Pre(parser::ExecutableConstruct &);

private:
  bool atomicDirectiveDefaultOrderFound_{false};
};

bool OmpRewriteMutator::Pre(parser::OpenMPAtomicConstruct &x) {
  // Find top-level parent of the operation.
  Symbol *topLevelParent{common::visit(
      [&](auto &atomic) {
        Symbol *symbol{nullptr};
        Scope *scope{
            &context_.FindScope(std::get<parser::Verbatim>(atomic.t).source)};
        do {
          if (Symbol * parent{scope->symbol()}) {
            symbol = parent;
          }
          scope = &scope->parent();
        } while (!scope->IsGlobal());

        assert(symbol &&
            "Atomic construct must be within a scope associated with a symbol");
        return symbol;
      },
      x.u)};

  // Get the `atomic_default_mem_order` clause from the top-level parent.
  std::optional<common::OmpAtomicDefaultMemOrderType> defaultMemOrder;
  common::visit(
      [&](auto &details) {
        if constexpr (std::is_convertible_v<decltype(&details),
                          WithOmpDeclarative *>) {
          if (details.has_ompAtomicDefaultMemOrder()) {
            defaultMemOrder = *details.ompAtomicDefaultMemOrder();
          }
        }
      },
      topLevelParent->details());

  if (!defaultMemOrder) {
    return false;
  }

  auto findMemOrderClause =
      [](const std::list<parser::OmpAtomicClause> &clauses) {
        return llvm::any_of(clauses, [](const auto &clause) {
          return std::get_if<parser::OmpMemoryOrderClause>(&clause.u);
        });
      };

  // Get the clause list to which the new memory order clause must be added,
  // only if there are no other memory order clauses present for this atomic
  // directive.
  std::list<parser::OmpAtomicClause> *clauseList = common::visit(
      common::visitors{[&](parser::OmpAtomic &atomicConstruct) {
                         // OmpAtomic only has a single list of clauses.
                         auto &clauses{std::get<parser::OmpAtomicClauseList>(
                             atomicConstruct.t)};
                         return !findMemOrderClause(clauses.v) ? &clauses.v
                                                               : nullptr;
                       },
          [&](auto &atomicConstruct) {
            // All other atomic constructs have two lists of clauses.
            auto &clausesLhs{std::get<0>(atomicConstruct.t)};
            auto &clausesRhs{std::get<2>(atomicConstruct.t)};
            return !findMemOrderClause(clausesLhs.v) &&
                    !findMemOrderClause(clausesRhs.v)
                ? &clausesRhs.v
                : nullptr;
          }},
      x.u);

  // Add a memory order clause to the atomic directive.
  if (clauseList) {
    atomicDirectiveDefaultOrderFound_ = true;
    switch (*defaultMemOrder) {
    case common::OmpAtomicDefaultMemOrderType::AcqRel:
      clauseList->emplace_back<parser::OmpMemoryOrderClause>(common::visit(
          common::visitors{[](parser::OmpAtomicRead &) -> parser::OmpClause {
                             return parser::OmpClause::Acquire{};
                           },
              [](parser::OmpAtomicCapture &) -> parser::OmpClause {
                return parser::OmpClause::AcqRel{};
              },
              [](auto &) -> parser::OmpClause {
                // parser::{OmpAtomic, OmpAtomicUpdate, OmpAtomicWrite}
                return parser::OmpClause::Release{};
              }},
          x.u));
      break;
    case common::OmpAtomicDefaultMemOrderType::Relaxed:
      clauseList->emplace_back<parser::OmpMemoryOrderClause>(
          parser::OmpClause{parser::OmpClause::Relaxed{}});
      break;
    case common::OmpAtomicDefaultMemOrderType::SeqCst:
      clauseList->emplace_back<parser::OmpMemoryOrderClause>(
          parser::OmpClause{parser::OmpClause::SeqCst{}});
      break;
    }
  }

  return false;
}

bool OmpRewriteMutator::Pre(parser::OpenMPRequiresConstruct &x) {
  for (parser::OmpClause &clause : std::get<parser::OmpClauseList>(x.t).v) {
    if (std::holds_alternative<parser::OmpClause::AtomicDefaultMemOrder>(
            clause.u) &&
        atomicDirectiveDefaultOrderFound_) {
      context_.Say(clause.source,
          "REQUIRES directive with '%s' clause found lexically after atomic "
          "operation without a memory order clause"_err_en_US,
          parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(
              llvm::omp::OMPC_atomic_default_mem_order)
                                         .str()));
    }
  }
  return false;
}

static void convertLoopControl(parser::LoopControl &loop) {
  // Extract relevant information from concurrent loop control.
  // TODO: Support LocalitySpec, IntegerTypeSpec, ScalarMaskExpr.
  auto &concurrent{std::get<parser::LoopControl::Concurrent>(loop.u)};
  auto &header{std::get<parser::ConcurrentHeader>(concurrent.t)};
  // TODO: Support multiple ConcurrentControl: create multiple loops + collapse.
  auto &control{
      std::get<std::list<parser::ConcurrentControl>>(header.t).front()};
  auto &[name, lower, upper, step] = control.t;

  // Create needed information for the new loop control.
  parser::ScalarName newName{std::move(name)};
  parser::Scalar newLower{
      common::Indirection{parser::Expr{std::move(lower.thing.thing.value())}}};
  parser::Scalar newUpper{
      common::Indirection{parser::Expr{std::move(upper.thing.thing.value())}}};
  std::optional<parser::Scalar<common::Indirection<parser::Expr>>> newStep;
  if (step) {
    newStep = parser::Scalar{common::Indirection{
        parser::Expr{std::move(step->thing.thing.value())}}};
  }

  // Replace loop control.
  loop.u = parser::LoopControl::Bounds{std::move(newName), std::move(newLower),
      std::move(newUpper), std::move(newStep)};
}

// TODO: Investigate crashes after sema. Looks like symbols need updating too.
bool OmpRewriteMutator::Pre(parser::ExecutableConstruct &x) {
  // TODO: Only mutate PFT if -fdo-concurrent-parallel is passed. Would have to
  // pass that information separately or add it to SemanticsContext.
  if (auto *doConstruct{
          std::get_if<common::Indirection<parser::DoConstruct>>(&x.u)}) {
    if (doConstruct->value().IsDoConcurrent()) {
      // Replace do-concurrent loop control.
      auto &doStmt{std::get<parser::Statement<parser::NonLabelDoStmt>>(
          doConstruct->value().t)};
      auto &loopControl{
          std::get<std::optional<parser::LoopControl>>(doStmt.statement.t)};
      convertLoopControl(*loopControl);

      // Construct OpenMP loop PFT node.
      // TODO: Select directive based on -fdo-concurrent-parallel instead.
      parser::OmpLoopDirective ompLoopDir{
          llvm::omp::Directive::OMPD_parallel_do};
      std::list<parser::OmpClause> clauses;
      parser::OmpClauseList ompClauses{std::move(clauses)};
      parser::OmpBeginLoopDirective beginLoopDir{
          std::move(ompLoopDir), std::move(ompClauses)};
      parser::OpenMPLoopConstruct loopConstruct{std::move(beginLoopDir)};
      std::get<1>(loopConstruct.t) = std::move(doConstruct->value());

      // Replace original loop with constructed OpenMP loop.
      x.u = common::Indirection{
          std::move(parser::OpenMPConstruct{std::move(loopConstruct)})};
    }
  }
  return true;
}

bool RewriteOmpParts(SemanticsContext &context, parser::Program &program) {
  if (!context.IsEnabled(common::LanguageFeature::OpenMP)) {
    return true;
  }
  OmpRewriteMutator ompMutator{context};
  parser::Walk(program, ompMutator);
  return !context.AnyFatalError();
}

} // namespace Fortran::semantics
