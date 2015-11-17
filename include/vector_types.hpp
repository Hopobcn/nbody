//
// Created by pfarre on 19/09/15.
//

#pragma once

#include <utils.hpp>
#include <gpu/math.cuh>
#include <vector_type_traits.hpp>


template <typename T>
struct vec3 {
    using Type     = vec3<T>;
    using VecType  = typename vector_type_traits<Type>::VecType;
    using BaseType = T;

    T x;
    T y;
    T z;

    HOST_DEVICE_INLINE vec3()
    {}

    HOST_DEVICE_INLINE vec3(VecType v)
            : x{v.x}, y{v.y}, z{v.z}
    {}

    HOST_DEVICE_INLINE vec3(BaseType x, BaseType y, BaseType z)
            : x{x}, y{y}, z{z}
    {}

    HOST_DEVICE_INLINE Type& operator+=(const Type& lhs) {
        x += lhs.x;
        y += lhs.y;
        z += lhs.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator-=(const Type& lhs) {
        x -= lhs.x;
        y -= lhs.y;
        z -= lhs.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator*=(const Type& lhs) {
        x *= lhs.x;
        y *= lhs.y;
        z *= lhs.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator*=(BaseType scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
};

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator+(const vec3<T>& a, const vec3<T>& b) {
    vec3<T> r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
    return r;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator-(const vec3<T>& a, const vec3<T>& b) {
    vec3<T> r;
    r.x = a.x - b.x;
    r.y = a.y - b.y;
    r.z = a.z - b.z;
    return r;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator*(const vec3<T>& a, const vec3<T>& b) {
    vec3<T> r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator*(const vec3<T>& a, T scalar) {
    vec3<T> r;
    r.x = a.x * scalar;
    r.y = a.y * scalar;
    r.z = a.z * scalar;
    return r;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator*(T scalar, const vec3<T>& a) {
    vec3<T> r;
    r.x = a.x * scalar;
    r.y = a.y * scalar;
    r.z = a.z * scalar;
    return r;
}

template <typename T>
HOST_DEVICE_INLINE T dot(const vec3<T>& a, const vec3<T>& b) {
    vec3<T> r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r.x + r.y + r.z;
}

template <typename T>
HOST_DEVICE_INLINE T euclideanDist(const vec3<T>& a, const vec3<T>& b) {
    vec3<T> r = a - b;
    return sqrt(dot(r, r));
}

namespace types {

template <typename T>
DEVICE_INLINE vec3<T> max(const vec3<T>& a, const vec3<T>& b) {
    vec3<T> r;
    r.x = cuda::max(a.x, b.x);
    r.y = cuda::max(a.y, b.y);
    r.z = cuda::max(a.z, b.z);
    return r;
}

template <typename T>
DEVICE_INLINE vec3<T> min(const vec3<T>& a, const vec3<T>& b) {
    vec3<T> r;
    r.x = cuda::min(a.x, b.x);
    r.y = cuda::min(a.y, b.y);
    r.z = cuda::min(a.z, b.z);
    return r;
}

} // end namespace types


template <typename T>
struct vec4 {
    using Type     = vec4<T>;
    using Type3    = vec3<T>;
    using VecType  = typename vector_type_traits<Type>::VecType;
    using BaseType = T;

    T x; // x component
    T y; // y component
    T z; // z component
    T w; // mass

    HOST_DEVICE_INLINE vec4()
    {}

    HOST_DEVICE_INLINE vec4(VecType v)
            : x{v.x}, y{v.y}, z{v.z}, w{v.w}
    {}

    HOST_DEVICE_INLINE vec4(BaseType x, BaseType y, BaseType z, BaseType w)
            : x{x}, y{y}, z{z}, w{w}
    {}

    HOST_DEVICE_INLINE explicit operator VecType() const {
        return {x, y, z, w};
    }

    HOST_DEVICE_INLINE Type& operator+=(const Type other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator+=(const Type3& lhs) {
        x += lhs.x;
        y += lhs.y;
        z += lhs.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator-=(const Type& lhs) {
        x -= lhs.x;
        y -= lhs.y;
        z -= lhs.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator*=(const Type& lhs) {
        x *= lhs.x;
        y *= lhs.y;
        z *= lhs.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator*=(BaseType scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

};

template <typename T>
HOST_DEVICE_INLINE vec4<T> operator*(const vec4<T>& a, T scalar) {
    vec4<T> r;
    r.x = a.x * scalar;
    r.y = a.y * scalar;
    r.z = a.z * scalar;
    r.w = a.w * scalar;
    return r;
}

template <typename T>
HOST_DEVICE_INLINE vec4<T> operator*(T scalar, const vec4<T>& a) {
    vec4<T> r;
    r.x = a.x * scalar;
    r.y = a.y * scalar;
    r.z = a.z * scalar;
    r.w = a.w * scalar;
    return r;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator+(const vec4<T>& a, const vec4<T>& b) {
    vec3<T> r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
    return r;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator-(const vec4<T>& a, const vec4<T>& b) {
    vec3<T> r;
    r.x = a.x - b.x;
    r.y = a.y - b.y;
    r.z = a.z - b.z;
    return r;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator*(const vec4<T>& a, const vec4<T>& b) {
    vec3<T> r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r;
}

namespace types {

template <typename T>
DEVICE_INLINE vec3<T> max(const vec3<T>& a, const vec4<T>& b) {
    vec3<T> r;
    r.x = cuda::max(a.x, b.x);
    r.y = cuda::max(a.y, b.y);
    r.z = cuda::max(a.z, b.z);
    return r;
}

template <typename T>
DEVICE_INLINE vec3<T> min(const vec3<T>& a, const vec4<T>& b) {
    vec3<T> r;
    r.x = cuda::min(a.x, b.x);
    r.y = cuda::min(a.y, b.y);
    r.z = cuda::min(a.z, b.z);
    return r;
}

} // end namespace types