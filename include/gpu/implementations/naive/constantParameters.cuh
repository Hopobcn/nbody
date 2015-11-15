#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <utils.hpp>

__constant__ float softeningSquared;
__constant__ double softeningSquared_fp64;

template <typename T>
struct constParameters {
    DEVICE_INLINE T getSofteningSquared() { return softeningSquared; };
};

template <>
struct constParameters<double> {
    DEVICE_INLINE double getSofteningSquared() { return softeningSquared_fp64; };
};

// Ideally this two functions would be member of class constParameters
// but I haven't been able to compile it without being free functions.
cudaError_t setSofteningSquared(float softeningSq) {
    return cudaMemcpyToSymbol(softeningSquared,
                              reinterpret_cast<const void*>(&softeningSq),
                              sizeof(float));
}

cudaError_t setSofteningSquared(double softeningSq) {
    return cudaMemcpyToSymbol(softeningSquared_fp64,
                              reinterpret_cast<const void*>(&softeningSq),
                              sizeof(double));
}