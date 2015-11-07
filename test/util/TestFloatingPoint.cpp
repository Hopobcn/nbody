#include <test/util/FloatingPoint.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <cmath>

namespace test { namespace util {

// Helper Traits class to convert from:
//  float  -> unsigned int             32 bit FP -> 32 bit UINT
//  double -> long long unsigned int   64 bit FP -> 64 bit UINT
template <typename T>
struct float_to_unsigned_traits {
    using type = T;
};

template <>
struct float_to_unsigned_traits<float> {
    using type = unsigned int;
};

template <>
struct float_to_unsigned_traits<double> {
    using type = long long unsigned int;
};

template <typename T>
class FloatingPointTest
        : public ::testing::Test
{
public:
    virtual void SetUp() {
        if (sizeof(T) == 4)
            epsilon = 1*10^(-5);
        else
            epsilon = 1*10^(-12);
    }
    virtual void TearDown() {}

    union float_2_bits {
        using float_type    = T;
        using unsigned_type = typename float_to_unsigned_traits<T>::type;

        unsigned_type as_unsigned;
        T             as_float;
    };

    T fp_one_ULP_above(T original) {
        float_2_bits bits;
        bits.as_float = original;
        bits.as_unsigned += 1;
        return bits.as_float;
    }

    T epsilon;
};

using FloatingPointTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(FloatingPointTest, FloatingPointTypes);

TYPED_TEST(FloatingPointTest, AbsoluteErrorComparison) {
    // Arbitrary epsilon
    TypeParam epsilon = FloatingPointTest<TypeParam>::epsilon;

    TypeParam expectedResult = 10000;
    //TypeParam result        = +10000.000977;   // The closest floatng point to 10,000 without being 10,000
    TypeParam result         = FloatingPointTest<TypeParam>::fp_one_ULP_above(expectedResult);
    TypeParam absolute_error = std::fabs(result - expectedResult);
    // diff is equal to 0.000977, which is 97.7 times larger than our epsilon!!
    bool is_equal_absolute_error = std::fabs(absolute_error) < epsilon;

    EXPECT_FALSE(is_equal_absolute_error);

    FloatingPoint<TypeParam> fpExpectedResult(expectedResult);
    FloatingPoint<TypeParam> fpResult(result);

    // since result is 1 ULP above expectedResult
    bool is_equal_0ULP = fpExpectedResult.AlmostEquals(fpResult, 0); // this should give false
    bool is_equal_1ULP = fpExpectedResult.AlmostEquals(fpResult, 1); // this should give true

    EXPECT_FALSE(is_equal_0ULP)
    << "Expected: " << fpExpectedResult << " " << expectedResult << std::endl
    << "Actual:   " << fpResult         << " " << result << std::endl;

    EXPECT_TRUE(is_equal_1ULP)
    << "Expected: " << fpExpectedResult << " " << expectedResult << std::endl
    << "Actual:   " << fpResult         << " " << result << std::endl;
}

TYPED_TEST(FloatingPointTest, RelativeErrorComparison) {
    // Arbitrary epsilon
    TypeParam epsilon = 0.00001; // 99.999% accuracy

    TypeParam expectedResult = +0.0;   // Positive 0
    TypeParam result         = -0.0;   // Negative 0
    TypeParam relative_error = fabs((result - expectedResult)/expectedResult);
    // +0 and -0 are 0 ULP form each other but a simble relative error comparison FAILS!
    bool is_equal_relative_error = std::fabs(relative_error) < epsilon;

    FloatingPoint<TypeParam> fpExpectedResult(expectedResult);
    FloatingPoint<TypeParam> fpResult(result);

    EXPECT_FALSE(is_equal_relative_error)
    << "Expected: " << fpExpectedResult << " " << expectedResult << std::endl
    << "Actual:   " << fpResult         << " " << result << std::endl;

    // since +0 and -0 are 0 ULP from each other:
    bool is_equal_0ULP = fpExpectedResult.AlmostEquals(fpResult, 0); // this should give true

    EXPECT_TRUE(is_equal_0ULP)
    << "Expected: " << fpExpectedResult << " " << expectedResult << std::endl
    << "Actual:   " << fpResult         << " " << result << std::endl;
}

TYPED_TEST(FloatingPointTest, UnitsLastPlaceErrorComparison) {
    //
    //   Representation
    //   Float value    Hexadecimal     Decimal
    //   +1.99999976    0x3FFFFFFE     1073741822
    //   +1.99999988    0x3FFFFFFF     1073741823
    //   +2.00000000    0x40000000     1073741824
    //   +2.00000024    0x40000001     1073741825
    //   +2.00000048    0x40000002     1073741826

    TypeParam expectedResult   = +2.00000000;
    TypeParam result_1ULP_away = FloatingPointTest<TypeParam>::fp_one_ULP_above(expectedResult);
    TypeParam result_2ULP_away = FloatingPointTest<TypeParam>::fp_one_ULP_above(result_1ULP_away);
    TypeParam result_3ULP_away = FloatingPointTest<TypeParam>::fp_one_ULP_above(result_2ULP_away);
    TypeParam result_4ULP_away = FloatingPointTest<TypeParam>::fp_one_ULP_above(result_3ULP_away);

    FloatingPoint<TypeParam> fpExpectedResult(expectedResult);
    FloatingPoint<TypeParam> fpResult1ULP(result_1ULP_away);
    FloatingPoint<TypeParam> fpResult2ULP(result_2ULP_away);
    FloatingPoint<TypeParam> fpResult3ULP(result_3ULP_away);
    FloatingPoint<TypeParam> fpResult4ULP(result_4ULP_away);

    #define PRINT_VALUES                                                                   \
            << "Expected:  " << fpExpectedResult << " " << expectedResult   << std::endl   \
            << "1ULP away: " << fpResult1ULP     << " " << result_1ULP_away << std::endl   \
            << "2ULP away: " << fpResult2ULP     << " " << result_2ULP_away << std::endl   \
            << "3ULP away: " << fpResult3ULP     << " " << result_3ULP_away << std::endl   \
            << "4ULP away: " << fpResult4ULP     << " " << result_4ULP_away << std::endl   \
            << "Actual:    " << fpExpectedResult << " " << expectedResult   << std::endl   \


    EXPECT_TRUE(  fpExpectedResult.AlmostEquals(fpExpectedResult, 0) )  PRINT_VALUES;
    EXPECT_FALSE( fpExpectedResult.AlmostEquals(fpResult1ULP, 0) )      PRINT_VALUES;
    EXPECT_FALSE( fpExpectedResult.AlmostEquals(fpResult2ULP, 0) )      PRINT_VALUES;
    EXPECT_FALSE( fpExpectedResult.AlmostEquals(fpResult3ULP, 0) )      PRINT_VALUES;
    EXPECT_FALSE( fpExpectedResult.AlmostEquals(fpResult4ULP, 0) )      PRINT_VALUES;

    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpExpectedResult, 1) )   PRINT_VALUES;
    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpResult1ULP, 1) )       PRINT_VALUES;
    EXPECT_FALSE( fpExpectedResult.AlmostEquals(fpResult2ULP, 1) )      PRINT_VALUES;
    EXPECT_FALSE( fpExpectedResult.AlmostEquals(fpResult3ULP, 1) )      PRINT_VALUES;
    EXPECT_FALSE( fpExpectedResult.AlmostEquals(fpResult4ULP, 1) )      PRINT_VALUES;

    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpExpectedResult, 2) )   PRINT_VALUES;
    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpResult1ULP, 2) )       PRINT_VALUES;
    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpResult2ULP, 2) )       PRINT_VALUES;
    EXPECT_FALSE( fpExpectedResult.AlmostEquals(fpResult3ULP, 2) )      PRINT_VALUES;
    EXPECT_FALSE( fpExpectedResult.AlmostEquals(fpResult4ULP, 2) )      PRINT_VALUES;

    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpExpectedResult, 3) )   PRINT_VALUES;
    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpResult1ULP, 3) )       PRINT_VALUES;
    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpResult2ULP, 3) )       PRINT_VALUES;
    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpResult3ULP, 3) )       PRINT_VALUES;
    EXPECT_FALSE( fpExpectedResult.AlmostEquals(fpResult4ULP, 3) )      PRINT_VALUES;

    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpExpectedResult, 4) )   PRINT_VALUES;
    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpResult1ULP, 4) )       PRINT_VALUES;
    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpResult2ULP, 4) )       PRINT_VALUES;
    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpResult3ULP, 4) )       PRINT_VALUES;
    EXPECT_TRUE( fpExpectedResult.AlmostEquals(fpResult4ULP, 4) )       PRINT_VALUES;
}

TYPED_TEST(FloatingPointTest, is_nan) {
    TypeParam nan = std::numeric_limits<TypeParam>::quiet_NaN();

    FloatingPoint<TypeParam> fp(nan);

    EXPECT_TRUE(fp.is_nan());
}

} // end namespace util 
} // end namespace test