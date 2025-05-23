#ifndef __CONV2D_FP16_FWD_HEADER__
#define __CONV2D_FP16_FWD_HEADER__

#define __in__
#define __out__
#define __in_out__

#define HOST_DEVICE __device__ __host__ __forceinline__

template <typename value_t>
HOST_DEVICE
value_t clz(value_t x) {
    for (int i = 31; i >= 0; --i) {
        if ((1 << i) & x)
            return value_t(31 - i);
    }
    return value_t(32);
}

template <typename value_t>
HOST_DEVICE
value_t find_log2(value_t x) {
    int a = int(31 - clz(x));
    a += (x & (x - 1)) != 0;  // Round up, add 1 if not a power of 2.
    return a;
}

struct FastDivmod {

	int divisor;
	unsigned int multiplier;
	unsigned int shift_right;

    /// Find quotient and remainder
    HOST_DEVICE
    void fast_divmod(int& quotient, int& remainder, int dividend) const {

        quotient = int((divisor != 1) ? int(((int64_t)dividend * multiplier) >> 32) >> shift_right : dividend);

        // The remainder.
        remainder = dividend - (quotient * divisor);
    }
    HOST_DEVICE
    FastDivmod() : divisor(0), multiplier(0), shift_right(0) { }

    HOST_DEVICE
    FastDivmod(int divisor) : divisor(divisor) {

        if (divisor != 1) {
            unsigned int p = 31 + find_log2(divisor);
            unsigned m = unsigned(((1ull << p) + unsigned(divisor) - 1) / unsigned(divisor));

            multiplier = m;
            shift_right = p - 32;
        }
        else { 
            multiplier = 0;
            shift_right = 0;
        }
    }

    /// Computes integer division and modulus using precomputed values.
    HOST_DEVICE
    void operator()(int& quotient, int& remainder, int dividend) const {
        fast_divmod(quotient, remainder, dividend);
    }
};

typedef struct
{
    _Float16*   in;                             //输入数据地址
    _Float16*   weight;                         //权值数据地址
    _Float16*   out;                            //输出数据地址
    unsigned int      n;                              //batch szie              default value 1
    unsigned int      c;                              //channel number          default value 32
    unsigned int      h;                              //数据高                  default value 32
    unsigned int      w;                              //数据宽                  default value 32
    unsigned int      k;                              //卷积核数量              default value 32
    unsigned int      r;                              //卷积核高                default value 1
    unsigned int      s;                              //卷积核宽                default value 1
    unsigned int      u;                              //卷积在高方向上的步长     default value 1
    unsigned int      v;                              //卷积在宽方向上的步长     default value 1
    unsigned int      p;                              //卷积在高方向上的补边     default value 0
    unsigned int      q;                              //卷积在宽方向上的补边     default value 0
    FastDivmod divmod_rs;
    FastDivmod divmod_s;
    FastDivmod divmod_pq;
    FastDivmod divmod_q;
    FastDivmod divmod_c;
}problem_t;

typedef struct
{
    unsigned int         blockx;                    //blockx  number
    unsigned int         blocky;                    //blocky  number
    unsigned int         blockz;                    //blockz  number
    unsigned int         threadx;                   //threadx number per block
    unsigned int         thready;                   //thready number per block
    unsigned int         threadz;                   //threadz number per block
    unsigned int         dynmicLdsSize;             //动态分配的lds大小，如果不使用动态分配的lds，则该值为0；
    void*       kernelPtr;                 //kernel ptr
}kernelInfo_t;


int getParamsize(__in__ problem_t* problem, __out__ int* paramSize);
int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param);


#endif