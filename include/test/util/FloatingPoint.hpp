#pragma once

#include <iostream>
#include <cstdlib>
#include <ostream>
#include <bitset>
#include <limits>

namespace test { namespace util {

template <size_t size>
class TypeWithSize {
public:
    // This prevents the user from using TypeWithSize<N> with incorrect
    // values of N.
    using UInt = void;
};

// The specialization for size 2.
template <>
class TypeWithSize<2> {
public:
    // unsigned int has size 4 in gcc.
    using Int  = short;
    using UInt = unsigned short;
};

// The specialization for size 4.
template <>
class TypeWithSize<4> {
public:
    // unsigned int has size 4 in gcc.
    using Int  = int;
    using UInt = unsigned int;
};

// The specialization for size 8.
template <>
class TypeWithSize<8> {
public:
    using Int  = long long;
    using UInt = unsigned long long;
};

// Imlementation similar to:
// https://code.google.com/p/googletest/source/browse/trunk/include/gtest/internal/gtest-internal.h
// with the additoin that the kMaxULP is a paramter of AlmostEquals
// giving more flexibility to the user
template <typename RawType>
class FloatingPoint {
public:
    // Defines the unsigned integer type that has the same size as
    // the floating point number.
    using Bits = typename TypeWithSize<sizeof(RawType)>::UInt;

    // Constants

    // # of bits in a number
    static const size_t kBitCount = 8*sizeof(RawType);

    // # of fraction bits in a number
    static const size_t kFractionBitCount = std::numeric_limits<RawType>::digits - 1;

    // # of exponent bits in a number
    static const size_t kExponentBitCount = kBitCount - 1 - kFractionBitCount;

    // The mask for the sign bit
    static const Bits kSignBitMask = static_cast<Bits>(1) << (kBitCount - 1);

    // The mask for the fraction bits
    static const Bits kFractionBitMask = ~static_cast<Bits>(0) >> (kExponentBitCount + 1);

    // The mask for the exponent bits
    static const Bits kExponentBitMask = ~(kSignBitMask | kFractionBitMask);

    // Construct a FloatingPoint from a raw floating-point number
    // If x is a NAN bits are not preserved (but it's still a NaN)
    explicit FloatingPoint(const RawType& x) { u.value = x; };

    // Static Methods

    // Reinterprets a bit pattern as a floating-point number
    static RawType ReinterpretBits(const Bits bits) {
        FloatingPoint fp(0);
        fp.u.bits = bits;
        return fp.u.value;
    }

    // Returns the floating-point number that represent positive infinity
    static RawType Infinity() {
        return ReinterpretBits(kExponentBitMask);
    }

    // Returns the maximum representable finite floating-point number
    static RawType Max() {
        return std::numeric_limits<RawType>::max();
    }

    // Non-static methods

    // Returns the bits that represents this number
    const Bits& bits() const { return u.bits; }

    // Returns the exponent bits of this number.
    Bits exponent_bits() const { return kExponentBitMask & u.bits; }

    // Returns the fraction bits of this number.
    Bits fraction_bits() const { return kFractionBitMask & u.bits; }

    // Returns the sign bit of this number.
    Bits sign_bit() const { return kSignBitMask & u.bits; }

    // Returns true iff this is NAN (not a number).
    bool is_nan() const {
        // It's a NAN if the exponent bits are all ones and the fraction
        // bits are not entirely zeros.
        return (exponent_bits() == kExponentBitMask) and (fraction_bits() != 0);
    }

    // Returns true iff this number is at most kMaxUlps ULP's away from
    // rhs.  In particular, this function:
    //
    //   - returns false if either number is (or both are) NAN.
    //   - treats really large numbers as almost equal to infinity.
    //   - thinks +0.0 and -0.0 are 0 DLP's apart.
    bool AlmostEquals(const FloatingPoint& rhs, const size_t kMaxUlps = 4) const {
        // The IEEE standard says that any comparison operation involving
        // a NAN must return false.
        if (is_nan() || rhs.is_nan()) return false;

        return DistanceBetweenSignAndMagnitudeNumbers(u.bits, rhs.u.bits) <= kMaxUlps;
    }

    // Two numbers are equals if they are 0 ULP form each other
    bool Equals(const FloatingPoint& rhs) const {
        return AlmostEquals(rhs, 0);
    }

    Bits DistanceInULP(const FloatingPoint& rhs) const {
        if (is_nan() || rhs.is_nan()) return std::numeric_limits<Bits>::max();

        return DistanceBetweenSignAndMagnitudeNumbers(u.bits, rhs.u.bits);
    }

    std::ostream& print(std::ostream& os) const {
        std::bitset<kExponentBitCount> exp(exponent_bits());
        std::bitset<kFractionBitCount> man(fraction_bits());
        std::bitset<1>                 sign(sign_bit());
        return os << "sign: " << sign << " mantissa: " << man << " exp: " << exp;
    }

private:

    // The data type used to store the actual floating-point number.
    union FloatingPointUnion {
        RawType value;  // The raw floating-point number.
        Bits bits;      // The bits that represent the number.
    };

    // Converts an integer from the sign-and-magnitude representation to
    // the biased representation.  
    static Bits SignAndMagnitudeToBiased(const Bits& sam) {
        if (kSignBitMask & sam) {
            // sam represents a negative number.
            return ~sam + 1;
        } else {
            // sam represents a positive number.
            return kSignBitMask | sam;
        }
    }

    // Given two numbers in the sign-and-magnitude representation,
    // returns the distance between them as an unsigned number.
    static Bits DistanceBetweenSignAndMagnitudeNumbers(const Bits& sam1, const Bits& sam2) {
        const Bits biased1 = SignAndMagnitudeToBiased(sam1);
        const Bits biased2 = SignAndMagnitudeToBiased(sam2);
        return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
    }

    FloatingPointUnion u;
};

// Typedefs the instances of the FloatingPoint template class that we
// care to use.
using Single = FloatingPoint<float>;
using Double = FloatingPoint<double>;

template <typename T>
bool operator==(const FloatingPoint<T>& lhs, const FloatingPoint<T>& rhs) {
    return lhs.AlmostEquals(rhs,0);
}

template <typename T>
bool operator!=(const FloatingPoint<T>& lhs, const FloatingPoint<T>& rhs) {
    return not (lhs == rhs);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const FloatingPoint<T>& rhs) {
    return os << rhs.print(os);
}

} // end namespace util
} // end namespace test