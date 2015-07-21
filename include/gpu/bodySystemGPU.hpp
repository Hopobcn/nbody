#pragma once


#include <bodySystem.hpp>

template <typename T>
struct DeviceData
{
    T *dPos[2]; // mapped host pointers
    T *dVel;
    cudaEvent_t  event;
    unsigned int offset;
    unsigned int numBodies;
};

// CUDA BodySystem: runs on the GPU
template <typename T>
class BodySystemGPU : public BodySystem<T>
{
public:
    BodySystemGPU(unsigned int numBodies,
                  unsigned int numDevices,
                  unsigned int blockSize,
                  bool usePBO,
                  bool useSysMem = false);
    virtual ~BodySystemGPU();

    virtual void loadFile(const std::string &filename);

    virtual void update(T deltaTime);

    virtual void setSoftening(T softening);
    virtual void setDamping(T damping);

    virtual T *getArray(BodyArray array);
    virtual void   setArray(BodyArray array, const T *data);

    virtual unsigned int getCurrentReadBuffer() const
    {
        return m_pbo[m_currentRead];
    }

    virtual unsigned int getNumBodies() const
    {
        return m_numBodies;
    }

protected: // methods
    BodySystemGPU() {}

    virtual void _initialize(unsigned numBodies);
    virtual void _finalize();

protected: // data
    unsigned int m_numBodies;
    unsigned int m_numDevices;
    bool m_bInitialized;

    // Host data
    T *m_hPos[2];
    T *m_hVel;

    DeviceData<T> *m_deviceData;

    bool m_bUsePBO;
    bool m_bUseSysMem;
    unsigned int m_SMVersion;

    T m_damping;

    unsigned int m_pbo[2];
    cudaGraphicsResource *m_pGRes[2];
    unsigned int m_currentRead;
    unsigned int m_currentWrite;

    unsigned int m_blockSize;
};

#include "bodySystemGPU-impl.hpp"