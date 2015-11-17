#pragma once

#include <utils.hpp>
#include <algorithm>

namespace cuda {

template<typename T>
HOST_DEVICE_INLINE T rsqrt(T x)
{
    return 1/std::sqrt(x);
}

template<>
DEVICE_INLINE float rsqrt<float>(float x)
{
#ifdef __CUDACC__
    return  rsqrtf(x);
#else
    return 1/std::sqrt(x);
#endif
}

template<>
DEVICE_INLINE double rsqrt<double>(double x)
{
#ifdef __CUDACC__
    return rsqrt(x);
#else
    return 1/std::sqrt(x);
#endif
}


template <typename T>
HOST_DEVICE_INLINE T max(const T& a, const T& b)
{
    return std::max(a, b);
}

template <>
HOST_DEVICE_INLINE float max<float>(const float& a, const float& b)
{
#ifdef __CUDACC__
    return fmaxf(a, b);
#else
    return std::max(a, b);
#endif
}
template <>
HOST_DEVICE_INLINE double max<double>(const double& a, const double& b)
{
#ifdef __CUDACC__
    return fmax(a, b);
#else
    return std::max(a, b);
#endif
}


template <typename T>
HOST_DEVICE_INLINE T min(const T& a, const T& b)
{
    return std::min(a, b);
}

template <>
HOST_DEVICE_INLINE float min<float>(const float& a, const float& b)
{
#ifdef __CUDACC__
    return fminf(a, b);
#else
    return std::min(a, b);
#endif
}
template <>
HOST_DEVICE_INLINE double min<double>(const double& a, const double& b)
{
#ifdef __CUDACC__
    return fmin(a, b);
#else
    return std::min(a, b);
#endif
}

} //end namespace cuda