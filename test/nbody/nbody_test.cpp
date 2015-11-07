#include <gtest/gtest.h>
#include <test/util/FloatingPoint.hpp>

#include <cpu/bodySystemCPU.hpp>
#include <gpu/bodySystemGPU.hpp>


TEST(testing, this_test_fails) {
    ASSERT_EQ(1, 1);
}

TEST(testing, this_test_pass) {
    ASSERT_EQ(1, 1);
}

TEST(BodySystemTest, compareCpuGpuAbsoluteError) {
    const unsigned numBodies = 1024;

    BodySystemCPU<float> nbodyCpu(numBodies);
    BodySystemGPU<float> nbodyGpu(numBodies, 1, 256, false);

    nbodyCpu.setSoftening(1.0);
    nbodyCpu.setDamping(1.0);

    nbodyGpu.setSoftening(1.0);
    nbodyGpu.setDamping(1.0);

    float* PosCpu = new float[numBodies*4];
    float* PosGpu = new float[numBodies*4];
    float* VelCpu = new float[numBodies*4];
    float* VelGpu = new float[numBodies*4];

    nbodyCpu.setArray(BODYSYSTEM_POSITION, PosCpu);
    nbodyCpu.setArray(BODYSYSTEM_VELOCITY, VelCpu);
    nbodyGpu.setArray(BODYSYSTEM_POSITION, PosGpu);
    nbodyGpu.setArray(BODYSYSTEM_VELOCITY, VelGpu);

    nbodyCpu.update(0.001f);
    nbodyGpu.update(0.001f);

    float* hPos = nbodyCpu.getArray(BODYSYSTEM_POSITION);
    float* dPos = nbodyGpu.getArray(BODYSYSTEM_POSITION);

    float tolerance = 0.0005f;

    for (unsigned i = 0; i < numBodies; i++)
    {
        if (std::fabs(hPos[i] - dPos[i]) > tolerance)
        {
            EXPECT_TRUE(false);
            printf("Error: (host)%f != (device)%f\n", hPos[i], dPos[i]);
        }
    }
}

TEST(BodySystemTest, compareCpuGpuEpsilonRelativeError) {
    const unsigned numBodies = 1024;

    BodySystemCPU<float> nbodyCpu(numBodies);
    BodySystemGPU<float> nbodyGpu(numBodies, 1, 256, false);

    nbodyCpu.setSoftening(1.0);
    nbodyCpu.setDamping(1.0);

    nbodyGpu.setSoftening(1.0);
    nbodyGpu.setDamping(1.0);

    float* PosCpu = new float[numBodies*4];
    float* PosGpu = new float[numBodies*4];
    float* VelCpu = new float[numBodies*4];
    float* VelGpu = new float[numBodies*4];

    nbodyCpu.setArray(BODYSYSTEM_POSITION, PosCpu);
    nbodyCpu.setArray(BODYSYSTEM_VELOCITY, VelCpu);
    nbodyGpu.setArray(BODYSYSTEM_POSITION, PosGpu);
    nbodyGpu.setArray(BODYSYSTEM_VELOCITY, VelGpu);

    nbodyCpu.update(0.001f);
    nbodyGpu.update(0.001f);

    float* hPos = nbodyCpu.getArray(BODYSYSTEM_POSITION);
    float* dPos = nbodyGpu.getArray(BODYSYSTEM_POSITION);

    float tolerance = 0.0005f;

    for (unsigned i = 0; i < numBodies; i++)
    {
        if (std::fabs((hPos[i] - dPos[i])/hPos[i]) > tolerance)
        {
            EXPECT_TRUE(false);
            printf("Error: (host)%f != (device)%f\n", hPos[i], dPos[i]);
        }
    }
}

TEST(BodySystemTest, compareCpuGpuULP) {
    const unsigned numBodies = 1024;

    BodySystemCPU<float> nbodyCpu(numBodies);
    BodySystemGPU<float> nbodyGpu(numBodies, 1, 256, false);

    nbodyCpu.setSoftening(1.0);
    nbodyCpu.setDamping(1.0);

    nbodyGpu.setSoftening(1.0);
    nbodyGpu.setDamping(1.0);

    float* PosCpu = new float[numBodies*4];
    float* PosGpu = new float[numBodies*4];
    float* VelCpu = new float[numBodies*4];
    float* VelGpu = new float[numBodies*4];

    nbodyCpu.setArray(BODYSYSTEM_POSITION, PosCpu);
    nbodyCpu.setArray(BODYSYSTEM_VELOCITY, VelCpu);
    nbodyGpu.setArray(BODYSYSTEM_POSITION, PosGpu);
    nbodyGpu.setArray(BODYSYSTEM_VELOCITY, VelGpu);

    nbodyCpu.update(0.001f);
    nbodyGpu.update(0.001f);

    float* hPos = nbodyCpu.getArray(BODYSYSTEM_POSITION);
    float* dPos = nbodyGpu.getArray(BODYSYSTEM_POSITION);

    for (unsigned i = 0; i < numBodies; i++)
    {
        test::util::FloatingPoint<float> expected(hPos[i]);
        test::util::FloatingPoint<float> calculated(dPos[i]);

        bool is_equal_ULP = calculated.AlmostEquals(expected);

        EXPECT_TRUE(is_equal_ULP)
            << "Expected: " << expected   << " " << hPos[i] << std::endl
            << "Actual:   " << calculated << " " << dPos[i] << std::endl;
    }
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}