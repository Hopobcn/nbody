//
// Created by pfarre on 19/09/15.
//

#pragma once

#include <vector_types.hpp>

template <typename T>
struct vec3;
template <typename T>
struct vec4;

template <typename T>
struct vector_type_traits {
    typedef T VecType;
};

template <>
struct vector_type_traits<vec3<float>> {
    typedef float3 VecType;
};

template <>
struct vector_type_traits<vec3<double>> {
    typedef double3 VecType;
};

template <>
struct vector_type_traits<vec4<float>> {
    typedef float4 VecType;
};

template <>
struct vector_type_traits<vec4<double>> {
    typedef double4 VecType;
};