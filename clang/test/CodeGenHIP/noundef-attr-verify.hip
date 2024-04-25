// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -x hip -fcuda-is-device -emit-llvm %s \
// RUN:   -o - | FileCheck %s

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define WARP_SIZE 64

static constexpr int warpSize = __AMDGCN_WAVEFRONT_SIZE;

__device__ static inline unsigned int __lane_id() {
    return  __builtin_amdgcn_mbcnt_hi(
        -1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}

__device__
inline
int __shfl(int var, int src_lane, int width = warpSize) {
    int self = __lane_id();
    int index = src_lane + (self & ~(width-1));
    return __builtin_amdgcn_ds_bpermute(index<<2, var);
}

template <typename T>
static __device__
T __shfl_sync(unsigned mask, T val, int src_line, int width=WARP_SIZE)
{
   return __shfl(val, src_line, width);
}

// CHECK-LABEL: @_Z13shufflekernelv(
// CHECK: call noundef i32 @_ZL11__shfl_syncIiET_jS0_ii(i32 64, i32 %0, i32 0, i32 64)

__global__ void
shufflekernel()
{
    int res, t;
    res = __shfl_sync(WARP_SIZE, t, 0);
}
