#pragma once

#include <cmath>

#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <gpu/math.cuh>
#include <vector_types.hpp>
#include <gpu/bodySystemGPU.hpp>

__constant__ float softeningSquared;
__constant__ double softeningSquared_fp64;

cudaError_t setSofteningSquared(float softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared,
                              &softeningSq,
                              sizeof(float), 0,
                              cudaMemcpyHostToDevice);
}

cudaError_t setSofteningSquared(double softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared_fp64,
                              &softeningSq,
                              sizeof(double), 0,
                              cudaMemcpyHostToDevice);
}

template<class T>
struct SharedMemory
{
    DEVICE_INLINE operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    DEVICE_INLINE operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};




// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i+blockDim.x*threadIdx.y]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i,j) sharedPos[i+blockDim.x*j]

template <typename T>
DEVICE_INLINE T getSofteningSquared()
{
    return softeningSquared;
}
template <>
DEVICE_INLINE double getSofteningSquared<double>()
{
    return softeningSquared_fp64;
}

template <typename T>
DEVICE_INLINE typename vec3<T>::Type
bodyBodyInteraction(typename vec3<T>::Type ai,
        typename vec4<T>::Type bi,
        typename vec4<T>::Type bj)
{
    // r_ij  [3 FLOPS]
    typename vec3<T>::Type r = bj - bi;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    T distSqr = dot(r, r);
    distSqr  += getSofteningSquared<T>();

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist     = cuda::rsqrt(distSqr);
    T invDistCube = invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai += r * s;

    return ai;
}

template <typename T>
DEVICE_INLINE typename vec3<T>::Type
computeBodyAccel(typename vec4<T>::Type bodyPos,
        typename vec4<T>::VecType *positions,
        int numTiles)
{
    typename vec4<T>::VecType *sharedPos = SharedMemory<typename vec4<T>::VecType>();

    typename vec3<T>::Type acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++)
    {
        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

        __syncthreads();

        // This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128
        for (unsigned int counter = 0; counter < blockDim.x; counter++)
        {
            acc = bodyBodyInteraction<T>(acc, bodyPos, sharedPos[counter]);
        }

        __syncthreads();
    }

    return acc;
}

template<typename T>
__global__ void
integrateBodies(typename vec4<T>::VecType *__restrict__ newPos,
        typename vec4<T>::VecType *__restrict__ oldPos,
        typename vec4<T>::VecType *             vel,
        unsigned int deviceOffset, unsigned int deviceNumBodies,
        T deltaTime, T damping, int numTiles)
{
    for ( int index = blockIdx.x * blockDim.x + threadIdx.x;
          index < deviceNumBodies;
          index += blockDim.x * gridDim.x )
    {
        typename vec4<T>::Type position = oldPos[deviceOffset + index];

        typename vec3<T>::Type accel = computeBodyAccel<T>(position,
                oldPos,
                numTiles);

        // acceleration = force / mass;
        // new velocity = old velocity + acceleration * deltaTime
        // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
        // (because they cancel out).  Thus here force == acceleration
        typename vec4<T>::Type velocity = vel[deviceOffset + index];

        velocity += accel * deltaTime;
        velocity *= damping;

        // new position = old position + velocity * deltaTime
        position += velocity * deltaTime;

        // store new position and velocity
        newPos[deviceOffset + index] = static_cast<typename vec4<T>::VecType>(position);
        vel[deviceOffset + index]    = static_cast<typename vec4<T>::VecType>(velocity);
    }
}