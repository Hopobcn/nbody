#pragma once

#include <functional>
#include <cuda_runtime.h>

#include <gpu/intrinsics.cuh>
#include <vector_types.hpp>


/**
 * calculates the min and max positions of the system & the root node
 */
template <typename T, unsigned blockDimx>
__global__
void calculateBoundingBox(typename vec4<T>::VecType* __restrict__ pos,
                          typename vec4<T>::VecType* __restrict__ max,
                          typename vec4<T>::VecType* __restrict__ min,
                          int num_bodies, int num_nodes) {
    typename vec3<T>::Type maximum, minimum;
    maximum = {-1000,-1000,-1000};
    minimum = { 1000, 1000, 1000};

    for (int index = blockIdx.x * blockDimx + threadIdx.x;
         index < num_bodies;
         index += blockDim.x * gridDim.x) {
        typename vec4<T>::Type position = pos[index];

        maximum = types::max(maximum, position);
        minimum = types::min(minimum, position);
    }

    auto calc_max = [&] (const vec3<T>& a, const vec3<T>& b) -> vec3<T> { return a.max(b); }; //max(a, b); };
    auto calc_min = [&] (const vec3<T>& a, const vec3<T>& b) -> vec3<T> { return a.min(b); };

    maximum = cuda::blockReduce<vec3<T>, decltype(calc_max), blockDimx>(maximum, calc_max);
    minimum = cuda::blockReduce<vec3<T>, decltype(calc_min), blockDimx>(minimum, calc_min);
}