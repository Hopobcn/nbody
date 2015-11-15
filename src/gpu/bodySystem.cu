
#include <gpu/implementations/naive/bodySystemGPU-naive.cuh>

template <typename T>
void integrateNbodySystem(DeviceData<T> *deviceData,
                          cudaGraphicsResource **pgres,
                          unsigned int currentRead,
                          float deltaTime,
                          float damping,
                          unsigned int numBodies,
                          unsigned int numDevices,
                          int blockSize,
                          bool bUsePBO)
{
    if (bUsePBO)
    {
        cudaGraphicsResourceSetMapFlags(pgres[currentRead],   cudaGraphicsMapFlagsReadOnly);
        cudaGraphicsResourceSetMapFlags(pgres[1-currentRead], cudaGraphicsMapFlagsWriteDiscard);
        cudaGraphicsMapResources(2, pgres, 0);
        size_t bytes;
        cudaGraphicsResourceGetMappedPointer((void **)&(deviceData[0].dPos[currentRead]),   &bytes, pgres[currentRead]);
        cudaGraphicsResourceGetMappedPointer((void **)&(deviceData[0].dPos[1-currentRead]), &bytes, pgres[1-currentRead]);
    }

    for (unsigned int dev = 0; dev != numDevices; dev++)
    {
        if (numDevices > 1)
        {
            cudaSetDevice(dev);
        }

        int numBlocks = (deviceData[dev].numBodies + blockSize-1) / blockSize;
        int numTiles = (numBodies + blockSize - 1) / blockSize;
        int sharedMemSize = blockSize * 4 * sizeof(T); // 4 floats for pos

        integrateBodies<T><<< numBlocks, blockSize, sharedMemSize >>>
                        ((typename vec4<T>::VecType *)deviceData[dev].dPos[1-currentRead],
                         (typename vec4<T>::VecType *)deviceData[dev].dPos[currentRead],
                         (typename vec4<T>::VecType *)deviceData[dev].dVel,
                                                   deviceData[dev].offset,
                                                   deviceData[dev].numBodies,
                                                   deltaTime, damping, numTiles);

        if (numDevices > 1)
        {
            cudaEventRecord(deviceData[dev].event);
            // MJH: Hack on older driver versions to force kernel launches to flush!
            cudaStreamQuery(0);
        }

        // check if kernel invocation generated an error
        //getLastCudaError("Kernel execution failed");
    }

    if (numDevices > 1)
    {
        for (unsigned int dev = 0; dev < numDevices; dev++)
        {
            cudaEventSynchronize(deviceData[dev].event);
        }
    }

    if (bUsePBO)
    {
        cudaGraphicsUnmapResources(2, pgres, 0);
    }
}


// Explicit specializations needed to generate code
template void integrateNbodySystem<float>(DeviceData<float> *deviceData,
                                          cudaGraphicsResource **pgres,
                                          unsigned int currentRead,
                                          float deltaTime,
                                          float damping,
                                          unsigned int numBodies,
                                          unsigned int numDevices,
                                          int blockSize,
                                          bool bUsePBO);

template void integrateNbodySystem<double>(DeviceData<double> *deviceData,
                                           cudaGraphicsResource **pgres,
                                           unsigned int currentRead,
                                           float deltaTime,
                                           float damping,
                                           unsigned int numBodies,
                                           unsigned int numDevices,
                                           int blockSize,
                                           bool bUsePBO);
