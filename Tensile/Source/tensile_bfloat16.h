#include <cmath>
#include <cinttypes>
#include <iostream>

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

#define BFLOAT16_Q_NAN_VALUE 0xFFC1

typedef struct tensile_bfloat16
{
    tensile_bfloat16(): data(BFLOAT16_ZERO_VALUE) {}
    // zero extend lower 16 bits of bfloat16 to convert to IEEE float
    static float bfloat16_to_float(const tensile_bfloat16 v)
    {
        union
        {
            float fp32 = 0;
            uint16_t q[2];
        };
    
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        q[0] = v.data;
    #else
        q[1] = v.data;
    #endif
        return fp32;
    }
    
    // truncate lower 16 bits of IEEE float to convert to bfloat16
    static tensile_bfloat16 float_to_bfloat16(const float v)
    {
        tensile_bfloat16 bf16;
        if (std::isnan(v))
        {
            bf16.data = BFLOAT16_Q_NAN_VALUE;
            return bf16;
        }
        union {
            float fp32;
            uint16_t p[2];
        };
        fp32 = v;
    
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        bf16.data = p[0];
    #else
        bf16.data = p[1];
    #endif
        return bf16;
    }

    explicit tensile_bfloat16(const float v)
    {
        data = float_to_bfloat16(v).data;
    }

    explicit tensile_bfloat16(const double v) { data = float_to_bfloat16(static_cast<float>(v)).data; }
    explicit tensile_bfloat16(const int v) { data = float_to_bfloat16(static_cast<float>(v)).data; }
    explicit tensile_bfloat16(const uint32_t v) { data = float_to_bfloat16(static_cast<float>(v)).data; }

    explicit operator float() const
    {
        return bfloat16_to_float(*this);
    }

    explicit operator double() const
    {
        return static_cast<double>(float(*this));
    }

    explicit operator int() const
    {
        return static_cast<int>(float(*this));
    }

    explicit operator uint32_t() const
    {
        return static_cast<uint32_t>(float(*this));
    }

    uint16_t data;

    private:

        static const int16_t BFLOAT16_ZERO_VALUE = 0x00; 

} tensile_bfloat16;

inline std::ostream& operator<<(std::ostream& os, const tensile_bfloat16& bf16) { os << static_cast<float>(bf16); return os; }

inline tensile_bfloat16 operator+(tensile_bfloat16 a, tensile_bfloat16 b) { return static_cast<tensile_bfloat16>(static_cast<float>(a) + static_cast<float>(b)); }
inline tensile_bfloat16 operator+(int a, tensile_bfloat16 b) { return static_cast<tensile_bfloat16>(static_cast<float>(a) + static_cast<float>(b)); }
inline tensile_bfloat16 operator+(tensile_bfloat16 a, int b) { return static_cast<tensile_bfloat16>(static_cast<float>(a) + static_cast<float>(b)); }
inline tensile_bfloat16 operator-(tensile_bfloat16 a, tensile_bfloat16 b) { return static_cast<tensile_bfloat16>(static_cast<float>(a) - static_cast<float>(b)); }
inline tensile_bfloat16 operator*(tensile_bfloat16 a, tensile_bfloat16 b) { return static_cast<tensile_bfloat16>(static_cast<float>(a) * static_cast<float>(b)); }
inline tensile_bfloat16 operator/(tensile_bfloat16 a, tensile_bfloat16 b) { return static_cast<tensile_bfloat16>(static_cast<float>(a) / static_cast<float>(b)); }

inline bool operator<(tensile_bfloat16 a, tensile_bfloat16 b) { return static_cast<float>(a) < static_cast<float>(b); }
inline bool operator<=(tensile_bfloat16 a, tensile_bfloat16 b) { return static_cast<float>(a) <= static_cast<float>(b); }
inline bool operator==(tensile_bfloat16 a, tensile_bfloat16 b) { return static_cast<float>(a) == static_cast<float>(b); }
inline bool operator!=(tensile_bfloat16 a, tensile_bfloat16 b) { return static_cast<float>(a) != static_cast<float>(b); }
inline bool operator>(tensile_bfloat16 a, tensile_bfloat16 b) { return static_cast<float>(a) > static_cast<float>(b); }
inline bool operator>=(tensile_bfloat16 a, tensile_bfloat16 b) { return static_cast<float>(a) >= static_cast<float>(b); }

inline tensile_bfloat16& operator+=(tensile_bfloat16& a, tensile_bfloat16 b) { a = a + b; return a; }
inline tensile_bfloat16& operator-=(tensile_bfloat16& a, tensile_bfloat16 b) { a = a - b; return a; }
inline tensile_bfloat16& operator*=(tensile_bfloat16& a, tensile_bfloat16 b) { a = a * b; return a; }
inline tensile_bfloat16& operator/=(tensile_bfloat16& a, tensile_bfloat16 b) { a = a / b; return a; }

inline tensile_bfloat16 operator++(tensile_bfloat16& a) { a += tensile_bfloat16(1); return a; }
inline tensile_bfloat16 operator++(tensile_bfloat16& a, int) { tensile_bfloat16 original_value = a; ++a; return original_value;}

inline bool isinf(const tensile_bfloat16& a) { return std::isinf(static_cast<float>(a)); }
inline bool isnan(const tensile_bfloat16& a) { return std::isnan(static_cast<float>(a)); }
inline bool iszero(const tensile_bfloat16& a) { return (a.data & 0x7FFF) == 0; }

inline tensile_bfloat16 abs(const tensile_bfloat16& a) { return static_cast<tensile_bfloat16>(std::abs(static_cast<float>(a))); }
inline tensile_bfloat16 sin(const tensile_bfloat16& a) { return static_cast<tensile_bfloat16>(std::sin(static_cast<float>(a))); }
inline tensile_bfloat16 cos(const tensile_bfloat16& a) { return static_cast<tensile_bfloat16>(std::cos(static_cast<float>(a))); }
