#pragma once

#include <utils.hpp>

namespace cuda {

template <typename T>
DEVICE_INLINE
T __shfl_down(T var, unsigned int srcLane, int width=32) {
    return __shfl_down(var, srcLane, width);
}

template <>
DEVICE_INLINE
double __shfl_down<double>(double var, unsigned int srcLane, int width) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

template <>
DEVICE_INLINE
vec3<float> __shfl_down<vec3<float>>(vec3<float> var, unsigned int srcLane, int width) {
    var.x = __shfl_down(var.x, srcLane, width);
    var.y = __shfl_down(var.y, srcLane, width);
    var.z = __shfl_down(var.z, srcLane, width);
    return var;
}

template <typename T, typename OperationType>
DEVICE_INLINE
T warpReduce(T val, OperationType op, unsigned warpSize=32) {
    for (unsigned offset = warpSize/2; offset > 0; offset /= 2)
        val = op(val, __shfl_down(val, offset));
    return val;
}


template <typename T, typename OperationType, int BLOCK_DIM_X>
DEVICE_INLINE
T blockReduce(T value, OperationType op, unsigned warpSize = 32) {
    static __shared__ T
    warp_reductions[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    value = cuda::warpReduce(value, op);

    if (lane == 0) warp_reductions[wid] = value; // Write reduced value to shared memory
    __syncthreads();

    //read from shared memory only if that warp existed
    value = (threadIdx.x < blockDim.x / warpSize) ? warp_reductions[lane] : 0;

    if (wid==0) value = cuda::warpReduce(value, op); //Final reduce within first warp

    return value;
};

}