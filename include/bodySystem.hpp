#pragma once

#include <algorithm>
#include <string>
#include <utils.hpp>

enum NBodyConfig
{
    NBODY_CONFIG_RANDOM,
    NBODY_CONFIG_SHELL,
    NBODY_CONFIG_EXPAND,
    NBODY_NUM_CONFIGS
};

enum BodyArray
{
    BODYSYSTEM_POSITION,
    BODYSYSTEM_VELOCITY
};

template <typename T>
struct vec3 {
    typedef T BaseType;
    typedef vec3<T> Type;

    T x;
    T y;
    T z;

    HOST_DEVICE_INLINE Type& operator+=(const Type other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator-=(const Type other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator*=(const Type other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator*=(BaseType scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    HOST_DEVICE_INLINE BaseType dot() const {
        return x*x + y*y + z*z;
    }
};

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator+(vec3<T> rhs, vec3<T> lhs) {
    return rhs += lhs;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator-(vec3<T> rhs, vec3<T> lhs) {
    return rhs -= lhs;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator*(vec3<T> rhs, vec3<T> lhs) {
    return rhs *= lhs;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator*(vec3<T> rhs, T scalar) {
    return rhs *= scalar;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator*(T scalar, vec3<T> lhs) {
    return lhs *= scalar;
}

#if 0
template <>
struct vec3<float> {
    typedef float3 Type;
};

template <>
struct vec3<double> {
    typedef double3 Type;
};
#endif

template <typename T>
struct vec4 {
    typedef T BaseType;
    typedef vec4<T> Type;

    T x; // x component
    T y; // y component
    T z; // z component
    T w; // mass

    HOST_DEVICE_INLINE Type& operator+=(const Type other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator+=(const vec3<T> other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator-=(const Type other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator*=(const Type other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
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
HOST_DEVICE_INLINE vec4<T> operator*(vec4<T> rhs, T scalar) {
    return rhs *= scalar;
}

template <typename T>
HOST_DEVICE_INLINE vec4<T> operator*(T scalar, vec4<T> lhs) {
    return lhs *= scalar;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator+(vec4<T> rhs, vec4<T> lhs) {
    vec3<T> ret;
    ret.x = rhs.x + lhs.x;
    ret.y = rhs.y + lhs.y;
    ret.z = rhs.z + lhs.z;
    return ret;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator-(vec4<T> rhs, vec4<T> lhs) {
    vec3<T> ret;
    ret.x = rhs.x - lhs.x;
    ret.y = rhs.y - lhs.y;
    ret.z = rhs.z - lhs.z;
    return ret;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator*(vec4<T> rhs, vec4<T> lhs) {
    vec3<T> ret;
    ret.x = rhs.x * lhs.x;
    ret.y = rhs.y * lhs.y;
    ret.z = rhs.z * lhs.z;
    return ret;
}

template <typename T>
class BodySystem
{
public:
    BodySystem(unsigned numBodies) {}
    virtual ~BodySystem() {}

    virtual void loadFile(const std::string& filename) = 0;

    virtual void update(T deltaTime) = 0;

    virtual void setSoftening(T softening) = 0;
    virtual void setDamping(T damping) = 0;

    virtual T *getArray(BodyArray array) = 0;
    virtual void   setArray(BodyArray array, const T *data) = 0;

    virtual unsigned getCurrentReadBuffer() const = 0;

    virtual unsigned getNumBodies() const = 0;

    virtual void   synchronizeThreads() const {};

protected: // methods
    BodySystem() {} // default constructor

    virtual void _initialize(unsigned numBodies) = 0;
    virtual void _finalize() = 0;
};

inline float3
scalevec(float3 &vector, float scalar)
{
    float3 rt = vector;
    rt.x *= scalar;
    rt.y *= scalar;
    rt.z *= scalar;
    return rt;
}

inline float
normalize(float3 &vector)
{
    float dist = sqrtf(vector.x*vector.x + vector.y*vector.y + vector.z*vector.z);

    if (dist > 1e-6)
    {
        vector.x /= dist;
        vector.y /= dist;
        vector.z /= dist;
    }

    return dist;
}

inline float
dot(float3 v0, float3 v1)
{
    return v0.x*v1.x+v0.y*v1.y+v0.z*v1.z;
}

inline float3
cross(float3 v0, float3 v1)
{
    float3 rt;
    rt.x = v0.y*v1.z-v0.z*v1.y;
    rt.y = v0.z*v1.x-v0.x*v1.z;
    rt.z = v0.x*v1.y-v0.y*v1.x;
    return rt;
}


// utility function
template <typename T>
void randomizeBodies(NBodyConfig config, T *pos, T *vel, float *color, float clusterScale,
                     float velocityScale, int numBodies, bool vec4vel)
{
    switch (config)
    {
        default:
        case NBODY_CONFIG_RANDOM:
        {
            float scale = clusterScale * std::max<float>(1.0f, numBodies / (1024.0f));
            float vscale = velocityScale * scale;

            int p = 0, v = 0;
            int i = 0;

            while (i < numBodies)
            {
                float3 point;
                //const int scale = 16;
                point.x = rand() / (float) RAND_MAX * 2 - 1;
                point.y = rand() / (float) RAND_MAX * 2 - 1;
                point.z = rand() / (float) RAND_MAX * 2 - 1;
                float lenSqr = dot(point, point);

                if (lenSqr > 1)
                    continue;

                float3 velocity;
                velocity.x = rand() / (float) RAND_MAX * 2 - 1;
                velocity.y = rand() / (float) RAND_MAX * 2 - 1;
                velocity.z = rand() / (float) RAND_MAX * 2 - 1;
                lenSqr = dot(velocity, velocity);

                if (lenSqr > 1)
                    continue;

                pos[p++] = point.x * scale; // pos.x
                pos[p++] = point.y * scale; // pos.y
                pos[p++] = point.z * scale; // pos.z
                pos[p++] = 1.0f; // mass

                vel[v++] = velocity.x * vscale; // pos.x
                vel[v++] = velocity.y * vscale; // pos.x
                vel[v++] = velocity.z * vscale; // pos.x

                if (vec4vel) vel[v++] = 1.0f; // inverse mass

                i++;
            }
        }
            break;

        case NBODY_CONFIG_SHELL:
        {
            float scale = clusterScale;
            float vscale = scale * velocityScale;
            float inner = 2.5f * scale;
            float outer = 4.0f * scale;

            int p = 0, v=0;
            int i = 0;

            while (i < numBodies)//for(int i=0; i < numBodies; i++)
            {
                float x, y, z;
                x = rand() / (float) RAND_MAX * 2 - 1;
                y = rand() / (float) RAND_MAX * 2 - 1;
                z = rand() / (float) RAND_MAX * 2 - 1;

                float3 point = {x, y, z};
                float len = normalize(point);

                if (len > 1)
                    continue;

                pos[p++] =  point.x * (inner + (outer - inner) * rand() / (float) RAND_MAX);
                pos[p++] =  point.y * (inner + (outer - inner) * rand() / (float) RAND_MAX);
                pos[p++] =  point.z * (inner + (outer - inner) * rand() / (float) RAND_MAX);
                pos[p++] = 1.0f;

                x = 0.0f; // * (rand() / (float) RAND_MAX * 2 - 1);
                y = 0.0f; // * (rand() / (float) RAND_MAX * 2 - 1);
                z = 1.0f; // * (rand() / (float) RAND_MAX * 2 - 1);
                float3 axis = {x, y, z};
                normalize(axis);

                if (1 - dot(point, axis) < 1e-6)
                {
                    axis.x = point.y;
                    axis.y = point.x;
                    normalize(axis);
                }

                //if (point.y < 0) axis = scalevec(axis, -1);
                float3 vv = {(float)pos[4*i], (float)pos[4*i+1], (float)pos[4*i+2]};
                vv = cross(vv, axis);
                vel[v++] = vv.x * vscale;
                vel[v++] = vv.y * vscale;
                vel[v++] = vv.z * vscale;

                if (vec4vel) vel[v++] = 1.0f;

                i++;
            }
        }
            break;

        case NBODY_CONFIG_EXPAND:
        {
            float scale = clusterScale * numBodies / (1024.f);

            if (scale < 1.0f)
                scale = clusterScale;

            float vscale = scale * velocityScale;

            int p = 0, v = 0;

            for (int i=0; i < numBodies;)
            {
                float3 point;

                point.x = rand() / (float) RAND_MAX * 2 - 1;
                point.y = rand() / (float) RAND_MAX * 2 - 1;
                point.z = rand() / (float) RAND_MAX * 2 - 1;

                float lenSqr = dot(point, point);

                if (lenSqr > 1)
                    continue;

                pos[p++] = point.x * scale; // pos.x
                pos[p++] = point.y * scale; // pos.y
                pos[p++] = point.z * scale; // pos.z
                pos[p++] = 1.0f; // mass
                vel[v++] = point.x * vscale; // pos.x
                vel[v++] = point.y * vscale; // pos.x
                vel[v++] = point.z * vscale; // pos.x

                if (vec4vel) vel[v++] = 1.0f; // inverse mass

                i++;
            }
        }
            break;
    }

    if (color)
    {
        int v = 0;

        for (int i=0; i < numBodies; i++)
        {
            //const int scale = 16;
            color[v++] = rand() / (float) RAND_MAX;
            color[v++] = rand() / (float) RAND_MAX;
            color[v++] = rand() / (float) RAND_MAX;
            color[v++] = 1.0f;
        }
    }

}
