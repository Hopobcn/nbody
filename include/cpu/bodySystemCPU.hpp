#pragma once

#include <bodySystem.hpp>

// CPU Body System
template <typename T>
class BodySystemCPU : public BodySystem<T>
{
public:
    BodySystemCPU(unsigned numBodies);
    virtual ~BodySystemCPU();

    virtual void loadFile(const std::string &filename);

    virtual void update(T deltaTime);

    virtual void setSoftening(T softening)
    {
        m_softeningSquared = softening * softening;
    }
    virtual void setDamping(T damping)
    {
        m_damping = damping;
    }

    virtual T *getArray(BodyArray array);
    virtual void   setArray(BodyArray array, const T *data);

    virtual unsigned int getCurrentReadBuffer() const
    {
        return 0;
    }

    virtual unsigned int getNumBodies() const
    {
        return m_numBodies;
    }

protected: // methods
    BodySystemCPU() {} // default constructor

    virtual void _initialize(unsigned numBodies);
    virtual void _finalize();

    void _computeNBodyGravitation();
    void _integrateNBodySystem(T deltaTime);

protected: // data
    unsigned    m_numBodies;
    bool        m_bInitialized;

    T *m_pos;
    T *m_vel;
    T *m_force;

    T m_softeningSquared;
    T m_damping;
};

#include "bodySystemCPU-impl.hpp"
