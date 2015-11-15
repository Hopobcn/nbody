#pragma once

#include <utils.hpp>

namespace cuda {

template<typename T>
DEVICE_INLINE T rsqrt(T x)
{
    return ::rsqrt(x);
}

template<>
DEVICE_INLINE float rsqrt<float>(float x)
{
    return ::rsqrtf(x);
}

template<>
DEVICE_INLINE double rsqrt<double>(double x)
{
    return ::rsqrt(x);
}

} //end namespace cuda