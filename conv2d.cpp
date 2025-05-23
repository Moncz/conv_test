#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "conv2d.h"

#define HMMA161616(RC0, RA0, RB0)                                                    \
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0" \
                 : "+v"(RC0)                                                                                \
                 : "v"(RA0), "v"(RB0))

#define HMMA16168(RD0, RA0, RB0, RC0)                                                    \
    asm volatile("v_mmac_16x16x16_f32 %0, %1, %2, %3" \
                 : "=v"(RD0)                                                                                \
                 : "v"(RA0), "v"(RB0), "v"(RC0))

#define DS_SYNC(num)                          \
    asm volatile("s_waitcnt lgkmcnt (%0) \n" \
                 :: "n"(num))

#define BUF_LDG128(dst, src, offset)                                   \
    asm volatile("buffer_load_dwordx4 %0, %1, %2, 0, offen, offset:0 \n" \
                 : "=v"(dst), "+v"(offset), "+s"(src)                    \
                 :)
   
#define BARRIER() \
    asm volatile("s_barrier \n" ::)

#define DS_SYNC(num) \
    asm volatile("s_waitcnt lgkmcnt (%0) \n" ::"n"(num))

#define CG_SYNC(num) \
    asm volatile("s_waitcnt vmcnt (%0) \n" ::"n"(num))        
      
using Float2 = __NATIVE_VECTOR__(2, float);
using Float4 = __NATIVE_VECTOR__(4, float);
typedef long BB  __attribute__((ext_vector_type(2)));

/*选手自定义的kernel入参结构体*/
typedef struct mykernelParamType
{
    _Float16*   pin;                            //输入数据地址
    _Float16*   pweight;                        //权值数据地址
    _Float16*   pout;                           //输出数据地址
    unsigned int      n;                              //batch szie            
    unsigned int      c;                              //channel number        
    unsigned int      h;                              //数据高                
    unsigned int      w;                              //数据宽                
    unsigned int      k;                              //卷积核数量            
    unsigned int      r;                              //卷积核高              
    unsigned int      s;                              //卷积核宽              
    unsigned int      u;                              //卷积在高方向上的步长  
    unsigned int      v;                              //卷积在宽方向上的步长  
    unsigned int      p;                              //卷积在高方向上的补边  
    unsigned int      q;                              //卷积在宽方向上的补边  
    unsigned int      Oh;                             //卷积在高方向上输出大小    
    unsigned int      Ow;                             //卷积在宽方向上输出大小
    FastDivmod      divmod_rs;                          //预留                          
    FastDivmod      divmod_s;                          //预留
    FastDivmod      divmod_pq;                          //预留
    FastDivmod      divmod_q;                          //预留
    FastDivmod      divmod_c;                          //预留
    unsigned int      splitKNum;                          //预留
    unsigned int      revs5;                          //预留
    unsigned int      revs6;                          //预留
    unsigned int      revs7;                          //预留
}mykernelParamType;                          
class Conv2dFpropFilterTileAccessIteratorOptimized
{
public:
    using Index = int32_t;
    using AccessType = Float4;

private:
    Index iteration_strided_;
    _Float16 const *pointer_;
    Index inc_next_k;
    Index inc_next_rs;
    Index inc_next_c;
    uint32_t predicates_;
    int filter_rs_;
    int filter_c_;
    int RS;
    int RSC;
    int iteration_;

public:
    HOST_DEVICE
    Conv2dFpropFilterTileAccessIteratorOptimized(
        mykernelParamType param,
        int offset_k,
        int offset_c,
        int iteration) : iteration_strided_(0),
                        predicates_(0),
                        filter_rs_(0),
                        filter_c_(0),
                        iteration_(iteration)
    {
        RS = param.r * param.s;
        RSC = param.r * param.s * param.c;
        inc_next_k = 8 * param.r * param.s * param.c;
        inc_next_rs = param.c;
        inc_next_c = (64 - Index(RS - 1) * param.c);

        filter_c_ = offset_c;

        // PRAGMA_UNROLL
        for (int s = 0; s < 4; ++s)
        {
            uint32_t pred = ((offset_k + s * 8 < param.k) ? 1u : 0);
            predicates_ |= (pred << s);
        }

        // PRAGMA_UNROLL
        pointer_ = param.pweight;
        pointer_ += offset_k * RSC + offset_c;
    }

    /// Adds a pointer offset in units of Element
    HOST_DEVICE
    void add_pointer_offset(Index pointer_offset)
    {
        pointer_ += pointer_offset;
    }

    HOST_DEVICE
    void advance()
    {
        // Index next = inc_next_rs;

        // // moves to the next tile
        // ++filter_rs_;
        // if (filter_rs_ == RS)
        // {

        //     filter_rs_ = 0;
        //     next = inc_next_c;
        //     filter_c_ += 64;
        // }

        // pointer_ += next;

        // moves to the next tile
        // filter_c_ += 64;
        // if (filter_c_ >= params_.c)
        // {
        //     filter_c_ -= params_.c;
        //     ++filter_rs_;
        // }

        pointer_ += 64;
    }

    /// Clears the predicates
    HOST_DEVICE
    void clear_mask(bool clear = true)
    {
        predicates_ = clear ? 0u : predicates_;
    }

    /// Returns true if the current coordinate is within the filter tensor W
    HOST_DEVICE
    bool valid()
    {
        return (predicates_ & (1u << iteration_strided_));
        // return true;
    }

    /// Returns a pointer to the vector starting at the current coordinate
    HOST_DEVICE
    AccessType const *get() const
    {
        return reinterpret_cast<AccessType const *>(pointer_);
    }

    /// Increments to the next memory access
    HOST_DEVICE
    Conv2dFpropFilterTileAccessIteratorOptimized &operator++()
    {

        ++iteration_strided_;
        if (iteration_strided_ < iteration_)
        {

            // Move to the next K coordinate within the tile
            pointer_ += inc_next_k;

            return *this;
        }
        iteration_strided_ = 0;
        pointer_ -= (iteration_ - 1) * inc_next_k;

        return *this;
    }
};
/*选手自己实现的kernel*/
extern "C" __global__ void myKernelConv2dGpu128x128x64(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;

    int warp_id = tx / 64;
    int lane_id = tx % 64;

    _Float16 * flt_ptr = param.pweight;
    _Float16 * act_ptr = param.pin;

    int gemm_k = param.c * param.r * param.s;
    int gemm_m = param.k;
    int gemm_n = param.n * param.Oh * param.Ow;

    // ldg reg
    _Float16 A_ldg_reg[4][8], B_ldg_reg[4][8];
    Float4 acc[4][4];
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            acc[i][j] = (Float4)(0.0f);
        }
    }
    int pos_flt[4], pos_act[4];
    int pos_h[4], pos_w[4];
    bool A_ldg_guard[4], B_ldg_guard[4];

    // smem
    __shared__ char smem[128 * 64 * 2 * sizeof(_Float16)];
    _Float16* smemA = reinterpret_cast<_Float16 *>(smem);
    _Float16* smemB = reinterpret_cast<_Float16 *>(smem + 128 * 64 * sizeof(_Float16));

    // lds reg
    Float2 A_lds_reg[4][2][2],  B_lds_reg[4][2][2];

    // ldg
    int first_tile_k = gemm_k - (((gemm_k + 63) / 64) - 1) * 64;

    int gemm_rowA = by * 128 + warp_id * 32 + (lane_id / 8);
    const int gemm_colA = (lane_id % 8) * 8;
    int c, rs, r, s;
    param.divmod_c(rs, c, gemm_colA);
    param.divmod_s(r, s, rs);
    int offset = (r * param.w + s) * param.c + c;
    for(int i = 0; i < 4; i++){
        int n, pq, p, q;
        param.divmod_pq(n, pq, gemm_rowA);
        param.divmod_q(p, q, pq);
        // output mapping input index
        pos_h[i] = p - param.p;
        pos_w[i] = q - param.q;
        // pos_act[i] = n * param.c * param.h * param.w + pos_h[i] * param.w + pos_w[i];
        pos_act[i] = (n * param.h * param.w + pos_h[i] * param.w + pos_w[i]) * param.c;
        A_ldg_guard[i] = n < param.n;
        // param.divmod_rs(c, rs, gemm_colA + j);
        // param.divmod_s(r, s, rs);

        int ih = pos_h[i] + r;
        int iw = pos_w[i] + s;
        bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w && gemm_colA < first_tile_k;
        *(Float4*)(&(A_ldg_reg[i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
        // for(int j = 0; j < 8; j++){
        //     param.divmod_rs(c, rs, gemm_colA + j);
        //     param.divmod_s(r, s, rs);
        //     int ih = pos_h[i] + r;
        //     int iw = pos_w[i] + s;
        //     int offset = (c * param.h + r) * param.w + s;
        //     bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w && gemm_colA + j < first_tile_k;
        //     A_ldg_reg[i][j] = guard ? act_ptr[pos_act[i] + offset] : (_Float16)0.0;
        // }
        gemm_rowA += 8;
    }

    int gemm_rowB = bx * 128 + warp_id * 32 + (lane_id / 8);
    const int gemm_colB = (lane_id % 8) * 8;
    for(int i = 0; i < 4; i++){
        pos_flt[i] = gemm_rowB * gemm_k + gemm_colB;
        B_ldg_guard[i] = gemm_rowB < gemm_m;

        bool guard = B_ldg_guard[i] && gemm_colB < first_tile_k;
        *(Float4*)(&(B_ldg_reg[i][0])) = guard ? *(Float4*)(&(flt_ptr[pos_flt[i]])) : (Float4)(0.0f);
        // for(int j = 0; j < 8; j++){
        //     bool guard = B_ldg_guard[i] && gemm_colB + j < first_tile_k;
        //     B_ldg_reg[i][j] = guard ? flt_ptr[pos_flt[i] + j] : (_Float16)0.0;
        // }
        gemm_rowB += 8;
    }

    // swizzle to avoid bank confilct
    int A_sts_addr = warp_id * 64 * 32 + (lane_id / 8) * 64 + ((lane_id % 8) ^ (lane_id / 8)) * 8;
    int B_sts_addr = warp_id * 64 * 32 + (lane_id / 8) * 64 + ((lane_id % 8) ^ (lane_id / 8)) * 8;
    // sts 0
    *(Float4*)(&(smemA[A_sts_addr          ])) = *(Float4*)(&(A_ldg_reg[0][0]));
    *(Float4*)(&(smemA[A_sts_addr + 8  * 64])) = *(Float4*)(&(A_ldg_reg[1][0]));
    *(Float4*)(&(smemA[A_sts_addr + 16 * 64])) = *(Float4*)(&(A_ldg_reg[2][0]));
    *(Float4*)(&(smemA[A_sts_addr + 24 * 64])) = *(Float4*)(&(A_ldg_reg[3][0]));

    *(Float4*)(&(smemB[B_sts_addr          ])) = *(Float4*)(&(B_ldg_reg[0][0]));
    *(Float4*)(&(smemB[B_sts_addr + 8  * 64])) = *(Float4*)(&(B_ldg_reg[1][0]));
    *(Float4*)(&(smemB[B_sts_addr + 16 * 64])) = *(Float4*)(&(B_ldg_reg[2][0]));
    *(Float4*)(&(smemB[B_sts_addr + 24 * 64])) = *(Float4*)(&(B_ldg_reg[3][0]));

    __syncthreads();

    // lds addr
    int lds_row = (lane_id % 16);
    int lds_col0 = ((lane_id / 16) % 8) ^ (lds_row % 8);
    int lds_col1 = (((lane_id / 16) + 4) % 8) ^ (lds_row % 8);
    int A_lds_addr0 = ((warp_id % 2) * 64 * 64) + lds_row * 64 + lds_col0 * 8;
    int B_lds_addr0 = ((warp_id / 2) * 64 * 64) + lds_row * 64 + lds_col0 * 8;
    int A_lds_addr1 = ((warp_id % 2) * 64 * 64) + lds_row * 64 + lds_col1 * 8;
    int B_lds_addr1 = ((warp_id / 2) * 64 * 64) + lds_row * 64 + lds_col1 * 8;

    // lds0
    *(Float4*)(&(A_lds_reg[0][0][0])) = *(Float4*)(&(smemA[A_lds_addr0]));
    *(Float4*)(&(B_lds_reg[0][0][0])) = *(Float4*)(&(smemB[B_lds_addr0]));
    *(Float4*)(&(A_lds_reg[1][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + 16 * 64]));
    *(Float4*)(&(B_lds_reg[1][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + 16 * 64]));
    *(Float4*)(&(A_lds_reg[2][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + 32 * 64]));
    *(Float4*)(&(B_lds_reg[2][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + 32 * 64]));
    *(Float4*)(&(A_lds_reg[3][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + 48 * 64]));
    *(Float4*)(&(B_lds_reg[3][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + 48 * 64]));

    for(int itile = first_tile_k; itile < gemm_k; itile+=64){
        // ldg 1
        param.divmod_c(rs, c, gemm_colA + itile);
        param.divmod_s(r, s, rs);
        int offset = (r * param.w + s) * param.c + c;
        for(int i = 0; i < 4; i++){
            int ih = pos_h[i] + r;
            int iw = pos_w[i] + s;
            bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w;
            *(Float4*)(&(A_ldg_reg[i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
        
            // for(int j = 0; j < 8; j++){
            //     param.divmod_rs(c, rs, gemm_colA + itile + j);
            //     param.divmod_s(r, s, rs);
            //     int offset = (c * param.h + r) * param.w + s;
            //     int ih = pos_h[i] + r;
            //     int iw = pos_w[i] + s;
            //     bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w && gemm_colA + j < first_tile_k;
            //     A_ldg_reg[i][j] = guard ? act_ptr[pos_act[i] + offset] : (_Float16)0.0;
            // }
        }
        for(int i = 0; i < 4; i++){
            *(Float4*)(&(B_ldg_reg[i][0])) = B_ldg_guard[i] ? *(Float4*)(&(flt_ptr[pos_flt[i] + itile])) : (Float4)(0.0f);
            // for(int j = 0; j < 8; j++){
            //     B_ldg_reg[i][j] = B_ldg_guard[i] ? flt_ptr[pos_flt[i] + itile + j] : (_Float16)0.0;
            // }
        }

        // lds 1
        *(Float4*)(&(A_lds_reg[0][1][0])) = *(Float4*)(&(smemA[A_lds_addr1]));
        *(Float4*)(&(B_lds_reg[0][1][0])) = *(Float4*)(&(smemB[B_lds_addr1]));
        *(Float4*)(&(A_lds_reg[1][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + 16 * 64]));
        *(Float4*)(&(B_lds_reg[1][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + 16 * 64]));
        *(Float4*)(&(A_lds_reg[2][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + 32 * 64]));
        *(Float4*)(&(B_lds_reg[2][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + 32 * 64]));
        *(Float4*)(&(A_lds_reg[3][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + 48 * 64]));
        *(Float4*)(&(B_lds_reg[3][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + 48 * 64]));

        // mma 0
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                for(int k = 0; k < 2; k++){
                    HMMA161616(acc[i][j], 
                    A_lds_reg[i][0][k],
                    B_lds_reg[j][0][k]);
                }
            }
        }

        __syncthreads();

        // sts 1
        *(Float4*)(&(smemA[A_sts_addr          ])) = *(Float4*)(&(A_ldg_reg[0][0]));
        *(Float4*)(&(smemA[A_sts_addr + 8  * 64])) = *(Float4*)(&(A_ldg_reg[1][0]));
        *(Float4*)(&(smemA[A_sts_addr + 16 * 64])) = *(Float4*)(&(A_ldg_reg[2][0]));
        *(Float4*)(&(smemA[A_sts_addr + 24 * 64])) = *(Float4*)(&(A_ldg_reg[3][0]));

        *(Float4*)(&(smemB[B_sts_addr          ])) = *(Float4*)(&(B_ldg_reg[0][0]));
        *(Float4*)(&(smemB[B_sts_addr + 8  * 64])) = *(Float4*)(&(B_ldg_reg[1][0]));
        *(Float4*)(&(smemB[B_sts_addr + 16 * 64])) = *(Float4*)(&(B_ldg_reg[2][0]));
        *(Float4*)(&(smemB[B_sts_addr + 24 * 64])) = *(Float4*)(&(B_ldg_reg[3][0]));

        __syncthreads();
        
        // lds 0
        *(Float4*)(&(A_lds_reg[0][0][0])) = *(Float4*)(&(smemA[A_lds_addr0]));
        *(Float4*)(&(B_lds_reg[0][0][0])) = *(Float4*)(&(smemB[B_lds_addr0]));
        *(Float4*)(&(A_lds_reg[1][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + 16 * 64]));
        *(Float4*)(&(B_lds_reg[1][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + 16 * 64]));
        *(Float4*)(&(A_lds_reg[2][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + 32 * 64]));
        *(Float4*)(&(B_lds_reg[2][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + 32 * 64]));
        *(Float4*)(&(A_lds_reg[3][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + 48 * 64]));
        *(Float4*)(&(B_lds_reg[3][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + 48 * 64]));

        // mma 1
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                for(int k = 0; k < 2; k++){
                    HMMA161616(acc[i][j], 
                    A_lds_reg[i][1][k], 
                    B_lds_reg[j][1][k]);
                }
            }
        }
    }

    // lds 1
    *(Float4*)(&(A_lds_reg[0][1][0])) = *(Float4*)(&(smemA[A_lds_addr1]));
    *(Float4*)(&(B_lds_reg[0][1][0])) = *(Float4*)(&(smemB[B_lds_addr1]));
    *(Float4*)(&(A_lds_reg[1][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + 16 * 64]));
    *(Float4*)(&(B_lds_reg[1][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + 16 * 64]));
    *(Float4*)(&(A_lds_reg[2][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + 32 * 64]));
    *(Float4*)(&(B_lds_reg[2][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + 32 * 64]));
    *(Float4*)(&(A_lds_reg[3][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + 48 * 64]));
    *(Float4*)(&(B_lds_reg[3][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + 48 * 64]));

    // mma 0
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            for(int k = 0; k < 2; k++){
                HMMA161616(acc[i][j], 
                A_lds_reg[i][0][k], 
                B_lds_reg[j][0][k]);
            }
        }
    }

    // mma 1
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            for(int k = 0; k < 2; k++){ 
                HMMA161616(acc[i][j], 
                A_lds_reg[i][1][k], 
                B_lds_reg[j][1][k]);
            }
        }
    }

    _Float16* smemC = reinterpret_cast<_Float16 *>(smem);

    for(int i = 0; i < 4; i++){
        __syncthreads();
        for(int j = 0; j < 4; j++){
            int C_sts_addr = warp_id * 64 * 16 + (lane_id % 16) * 64 + j * 16 + (lane_id / 16);
            smemC[C_sts_addr     ] = (_Float16)acc[i][j].x;
            smemC[C_sts_addr +  4] = (_Float16)acc[i][j].y;
            smemC[C_sts_addr +  8] = (_Float16)acc[i][j].z;
            smemC[C_sts_addr + 12] = (_Float16)acc[i][j].w;
        }
        __syncthreads();
        for(int j = 0; j < 2; j++){
            int C_lds_addr = warp_id * 64 * 16 + j * 64 * 8 + (lane_id / 8) * 64 + (lane_id % 8) * 8;
            int k = bx * 128 + (warp_id / 2) * 64 + (lane_id % 8) * 8;
            int npq = by * 128 + (warp_id % 2) * 64 + i * 16 + j * 8 + (lane_id / 8);
            bool guard = npq < param.n * param.Oh * param.Ow && k < param.k;
            if(guard){
                int offset = npq * param.k + k;
                *(Float4*)(&(param.pout[offset])) = *(Float4*)(&(smemC[C_lds_addr]));
            }
        }
    }
}
template<int TILEM, int TILEN, int TILEK = 128>
__global__ void myKernelConv2dGputemplate_128(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;

    int warp_id = tx / 64;
    int lane_id = tx % 64;

    int warp_div2 = warp_id >> 1;
    int warp_mod2 = warp_id & 1;

    int lane_mod8 = lane_id & 7;
    int lane_div8 = lane_id >> 3;
    int lane_mod16 = lane_id & 15;
    int lane_div16 = lane_id >> 4;

    _Float16 * flt_ptr = param.pweight;
    _Float16 * act_ptr = param.pin;

    int gemm_k = param.c * param.r * param.s;
    int gemm_m = param.k;
    int gemm_n = param.n * param.Oh * param.Ow;

    // ldg reg
    _Float16 A_ldg_reg[2][TILEM/32][8], B_ldg_reg[2][TILEN/32][8];
    Float4 acc[TILEM/32][TILEN/32];
    for(int i = 0; i < TILEM/32; i++){
        for(int j = 0; j < TILEN/32; j++){
            acc[i][j] = (Float4)(0.0f);
        }
    }
    int pos_flt[TILEN/32], pos_act[TILEM/32];
    int pos_h[TILEM/32], pos_w[TILEM/32];
    bool A_ldg_guard[TILEM/32], B_ldg_guard[TILEN/32];

    // smem
    __shared__ char smem[(TILEM+TILEN) * 64 * sizeof(_Float16)];
    _Float16* smemA = reinterpret_cast<_Float16 *>(smem);
    _Float16* smemB = reinterpret_cast<_Float16 *>(smem + TILEM * TILEK * sizeof(_Float16));

    // lds reg
    Float2 A_lds_reg[TILEM/32][2][2],  B_lds_reg[TILEN/32][2][2];

    // ldg
    int first_tile_k = gemm_k - (((gemm_k + TILEK-1) / TILEK) - 1) * TILEK;

    int gemm_rowA = by * TILEM + warp_id * TILEM / 4 + lane_div8;
    const int gemm_colA = lane_mod8 * 8;
    int c, rs, r, s;
    param.divmod_c(rs, c, gemm_colA);
    param.divmod_s(r, s, rs);
    int offset = (r * param.w + s) * param.c + c;
    for(int i = 0; i < TILEM/32; i++){
        int n, pq, p, q;
        param.divmod_pq(n, pq, gemm_rowA);
        param.divmod_q(p, q, pq);
        // output mapping input index
        pos_h[i] = p - param.p;
        pos_w[i] = q - param.q;
        pos_act[i] = (n * param.h * param.w + pos_h[i] * param.w + pos_w[i]) * param.c;
        A_ldg_guard[i] = n < param.n;
        int ih = pos_h[i] + r;
        int iw = pos_w[i] + s;
        bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w && gemm_colA < first_tile_k;
        *(Float4*)(&(A_ldg_reg[0][i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
        gemm_rowA += 8;
    }

    int gemm_rowB = bx * TILEN + warp_id * TILEN / 4 + lane_div8;
    const int gemm_colB = lane_mod8 * 8;
    for(int i = 0; i < TILEN/32; i++){
        pos_flt[i] = gemm_rowB * gemm_k + gemm_colB;
        B_ldg_guard[i] = gemm_rowB < gemm_m;
        bool guard = B_ldg_guard[i] && gemm_colB < first_tile_k;
        *(Float4*)(&(B_ldg_reg[0][i][0])) = guard ? *(Float4*)(&(flt_ptr[pos_flt[i]])) : (Float4)(0.0f);
        gemm_rowB += 8;
    }

    // swizzle to avoid bank confilct
    int A_sts_addr = warp_id * 64 * TILEM / 4 + lane_div8 * 64 + (lane_mod8 ^ lane_div8) * 8;
    int B_sts_addr = warp_id * 64 * TILEN / 4 + lane_div8 * 64 + (lane_mod8 ^ lane_div8) * 8;
    // sts 0
    for(int i = 0; i < TILEM/32; i++){
        *(Float4*)(&(smemA[A_sts_addr + i * 8 * 64])) = *(Float4*)(&(A_ldg_reg[0][i][0]));
    }
    for(int i = 0; i < TILEN/32; i++){
        *(Float4*)(&(smemB[B_sts_addr + i * 8 * 64])) = *(Float4*)(&(B_ldg_reg[0][i][0]));
    }
    // lds addr
    int lds_row = lane_mod16;
    int lds_col0 = (lane_div16 % 8) ^ (lds_row % 8);
    int lds_col1 = ((lane_div16 + 4) % 8) ^ (lds_row % 8);
    //TILEM要>=32
    int A_lds_addr0 = (warp_mod2 * 64 * TILEM / 2) + lds_row * 64 + lds_col0 * 8;
    int A_lds_addr1 = (warp_mod2 * 64 * TILEM / 2) + lds_row * 64 + lds_col1 * 8;
    
    int B_lds_addr0 = (warp_div2 * 64 * TILEN / 2) + lds_row * 64 + lds_col0 * 8;
    int B_lds_addr1 = (warp_div2 * 64 * TILEN / 2) + lds_row * 64 + lds_col1 * 8;

    // ldg 1
    param.divmod_c(rs, c, gemm_colA + 64);
    param.divmod_s(r, s, rs);
    offset = (r * param.w + s) * param.c + c;
    for(int i = 0; i < TILEM/32; i++){
        int ih = pos_h[i] + r;
        int iw = pos_w[i] + s;
        bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w && (gemm_colA + 64) < first_tile_k;
        *(Float4*)(&(A_ldg_reg[1][i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
    }
    for(int i = 0; i < TILEN/32; i++){
        bool guard = B_ldg_guard[i] && gemm_colB + 64 < first_tile_k;
        *(Float4*)(&(B_ldg_reg[1][i][0])) = guard ? *(Float4*)(&(flt_ptr[pos_flt[i] + 64])) : (Float4)(0.0f);
    }

    __syncthreads();

    // lds0
    for(int i = 0; i < TILEM/32; i++){
        *(Float4*)(&(A_lds_reg[i][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + i * 16 * 64]));
    }
    for(int i = 0; i < TILEN/32; i++){
        *(Float4*)(&(B_lds_reg[i][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + i * 16 * 64]));
    }

    for(int itile = first_tile_k; itile < gemm_k; itile+=128){
	//！！重新计算偏移
    param.divmod_c(rs, c, gemm_colA + itile);
    param.divmod_s(r, s, rs);
    offset = (r * param.w + s) * param.c + c;
        // mma 0
        for(int i = 0; i < TILEM/32; i++){
            for(int j = 0; j < TILEN/32; j++){
                if(j == 0){
                    int ih = pos_h[i] + r;
                    int iw = pos_w[i] + s;
                    bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w;
                    asm(";-----");
                    *(Float4*)(&(A_ldg_reg[0][i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
                    *(Float4*)(&(A_lds_reg[i][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + i * 16 * 64]));
                    asm(";-----");
                }
                //TILEN要>=64
                if(j == TILEN/32/2){
                    asm(";-----");
                    *(Float4*)(&(B_ldg_reg[0][i][0])) = B_ldg_guard[i] ? *(Float4*)(&(flt_ptr[pos_flt[i] + itile])) : (Float4)(0.0f);
                    *(Float4*)(&(B_lds_reg[i][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + i * 16 * 64]));
                    asm(";-----");
                }
                for(int k = 0; k < 2; k++){
                    HMMA161616(acc[i][j], 
                    A_lds_reg[i][0][k], 
                    B_lds_reg[j][0][k]);
                }
            }
        }

        __syncthreads();

        // mma 1
        for(int i = 0; i < TILEM/32; i++){
            for(int j = 0; j < TILEN/32; j++){
                if(j == 0){
                    asm(";-----");
                    *(Float4*)(&(smemA[A_sts_addr + i * 8 * 64])) = *(Float4*)(&(A_ldg_reg[1][i][0]));
                    asm(";-----");
                }
                if(j == TILEN/32/2){
                    asm(";-----");
                    *(Float4*)(&(smemB[B_sts_addr + i * 8 * 64])) = *(Float4*)(&(B_ldg_reg[1][i][0]));
                    asm(";-----");
                }
                for(int k = 0; k < 2; k++){
                    HMMA161616(acc[i][j], 
                    A_lds_reg[i][1][k], 
                    B_lds_reg[j][1][k]);
                }
            }
        }

        __syncthreads();
        
        // lds0
    for(int i = 0; i < TILEM/32; i++){
        *(Float4*)(&(A_lds_reg[i][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + i * 16 * 64]));
    }
    for(int i = 0; i < TILEN/32; i++){
        *(Float4*)(&(B_lds_reg[i][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + i * 16 * 64]));
    }


        param.divmod_c(rs, c, gemm_colA + itile + 64);
        param.divmod_s(r, s, rs);
        offset = (r * param.w + s) * param.c + c;
        // mma 0
        for(int i = 0; i < TILEM/32; i++){
            for(int j = 0; j < TILEN/32; j++){
                if(j == 0){
                    int ih = pos_h[i] + r;
                    int iw = pos_w[i] + s;
                    bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w;
                    asm(";-----");
                    *(Float4*)(&(A_ldg_reg[1][i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
                    *(Float4*)(&(A_lds_reg[i][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + i * 16 * 64]));
                    asm(";-----");
                }
                if(j == TILEN/32/2){
                    asm(";-----");
                    *(Float4*)(&(B_ldg_reg[1][i][0])) = B_ldg_guard[i] ? *(Float4*)(&(flt_ptr[pos_flt[i] + itile + 64])) : (Float4)(0.0f);
                    *(Float4*)(&(B_lds_reg[i][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + i * 16 * 64]));
                    asm(";-----");
                }
                for(int k = 0; k < 2; k++){
                    HMMA161616(acc[i][j], 
                    A_lds_reg[i][0][k], 
                    B_lds_reg[j][0][k]);
                }
            }
        }

        __syncthreads();

        // mma 1
        for(int i = 0; i < TILEM/32; i++){
            for(int j = 0; j < TILEN/32; j++){
                if(j == 0){
                    asm(";-----");
                    *(Float4*)(&(smemA[A_sts_addr + i * 8 * 64])) = *(Float4*)(&(A_ldg_reg[0][i][0]));
                    asm(";-----");
                }
                if(j == TILEN/32/2){
                    asm(";-----");
                    *(Float4*)(&(smemB[B_sts_addr + i * 8 * 64])) = *(Float4*)(&(B_ldg_reg[0][i][0]));
                    asm(";-----");
                }
                for(int k = 0; k < 2; k++){
                    HMMA161616(acc[i][j], 
                    A_lds_reg[i][1][k], 
                    B_lds_reg[j][1][k]);
                }
            }
        }
        __syncthreads();

     // lds0
    for(int i = 0; i < TILEM/32; i++){
        *(Float4*)(&(A_lds_reg[i][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + i * 16 * 64]));
    }
    for(int i = 0; i < TILEN/32; i++){
        *(Float4*)(&(B_lds_reg[i][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + i * 16 * 64]));
    }

    }

    // lds 1
    for(int i = 0; i < TILEM/32; i++){
        *(Float4*)(&(A_lds_reg[i][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + i * 16 * 64]));
    }
    for(int i = 0; i < TILEN/32; i++){
        *(Float4*)(&(B_lds_reg[i][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + i * 16 * 64]));
    }

    // mma 0
    for(int i = 0; i < TILEM/32; i++){
        for(int j = 0; j < TILEN/32; j++){
            for(int k = 0; k < 2; k++){
                HMMA161616(acc[i][j], 
                A_lds_reg[i][0][k], 
                B_lds_reg[j][0][k]);
            }
        }
    }
    __syncthreads();

    // sts 1
    for(int i = 0; i < TILEM/32; i++){
        *(Float4*)(&(smemA[A_sts_addr + i * 8 * 64])) = *(Float4*)(&(A_ldg_reg[1][i][0]));
    }
    for(int i = 0; i < TILEN/32; i++){
        *(Float4*)(&(smemB[B_sts_addr + i * 8 * 64])) = *(Float4*)(&(B_ldg_reg[1][i][0]));
    }

    // mma 1
    for(int i = 0; i < TILEM/32; i++){
        for(int j = 0; j < TILEN/32; j++){
            for(int k = 0; k < 2; k++){
                HMMA161616(acc[i][j], 
                A_lds_reg[i][1][k], 
                B_lds_reg[j][1][k]);
            }
        }
    }
    __syncthreads();

    // lds 0
        for(int i = 0; i < TILEM/32; i++){
            *(Float4*)(&(A_lds_reg[i][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + i * 16 * 64]));
        }
        for(int i = 0; i < TILEN/32; i++){
            *(Float4*)(&(B_lds_reg[i][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + i * 16 * 64]));
        }

    // lds 1
        for(int i = 0; i < TILEM/32; i++){
            *(Float4*)(&(A_lds_reg[i][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + i * 16 * 64]));
        }
        for(int i = 0; i < TILEN/32; i++){
            *(Float4*)(&(B_lds_reg[i][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + i * 16 * 64]));
        }

    __syncthreads();
    // mma 0
    for(int i = 0; i < TILEM/32; i++){
        for(int j = 0; j < TILEN/32; j++){
            for(int k = 0; k < 2; k++){
                HMMA161616(acc[i][j], 
                A_lds_reg[i][0][k], 
                B_lds_reg[j][0][k]);
            }
        }
    }

    // mma 1
    for(int i = 0; i < TILEM/32; i++){
        for(int j = 0; j < TILEN/32; j++){
            for(int k = 0; k < 2; k++){
                HMMA161616(acc[i][j], 
                A_lds_reg[i][1][k], 
                B_lds_reg[j][1][k]);
            }
        }
    }
    _Float16* smemC = reinterpret_cast<_Float16 *>(smem);

    for(int i = 0; i < TILEM/32; i++){
        __syncthreads();
        for(int j = 0; j < TILEN/32; j++){
            int C_sts_addr = warp_id * TILEK * 16 + (lane_id % 16) * TILEK + j * 16 + (lane_id / 16);
            smemC[C_sts_addr     ] = (_Float16)acc[i][j].x;
            smemC[C_sts_addr +  4] = (_Float16)acc[i][j].y;
            smemC[C_sts_addr +  8] = (_Float16)acc[i][j].z;
            smemC[C_sts_addr + 12] = (_Float16)acc[i][j].w;
        }
        __syncthreads();
        
        if(TILEN<64){
            int C_lds_addr = warp_id * 64 * 16 + (lane_id/2 / (TILEN/2/8)) * 64 + (lane_id/2 % (TILEN/2/8)) * 8;
            int k = bx * TILEN + (warp_id / 2) * TILEN/2 + (lane_id/2 % (TILEN/2/8)) * 8;
            int npq = by * TILEM + (warp_id % 2) * TILEM/2 + i * 16 + (lane_id/2 / (TILEN/2/8));
            bool guard = npq < param.n * param.Oh * param.Ow && k < param.k;
            if(guard){
                int offset = npq * param.k + k;
                *(Float4*)(&(param.pout[offset])) = *(Float4*)(&(smemC[C_lds_addr]));
            }
        }else{
            for(int j = 0; j < 16/(64/(TILEN/2/8)); j++){
                int C_lds_addr = warp_id * 64 * 16 + j * 64 * (64/(TILEN/2/8)) + (lane_id / (TILEN/2/8)) * 64 + (lane_id % (TILEN/2/8)) * 8;
                int k = bx * TILEN + (warp_id / 2) * TILEN/2 + (lane_id % (TILEN/2/8)) * 8;
                int npq = by * TILEM + (warp_id % 2) * TILEM/2 + i * 16 + j * (64/(TILEN/2/8)) + (lane_id / (TILEN/2/8));
                bool guard = npq < param.n * param.Oh * param.Ow && k < param.k;
            if(guard){
                int offset = npq * param.k + k;
                *(Float4*)(&(param.pout[offset])) = *(Float4*)(&(smemC[C_lds_addr]));
            }
        }
        }
    }

    
}
template<int TILEM, int TILEN, int TILEK = 64>
__global__ void myKernelConv2dGputemplate_64(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;

    int warp_id = tx >> 6;
    int lane_id = tx & 63;

    int warp_div2 = warp_id >> 1;
    int warp_mod2 = warp_id & 1;

    int lane_mod8 = lane_id & 7;
    int lane_div8 = lane_id >> 3;
    int lane_mod16 = lane_id & 15;
    int lane_div16 = lane_id >> 4;

    _Float16 * flt_ptr = param.pweight;
    _Float16 * act_ptr = param.pin;

    int gemm_k = param.c * param.r * param.s;
    int gemm_m = param.k;
    int gemm_n = param.n * param.Oh * param.Ow;

    // ldg reg
    _Float16 A_ldg_reg[TILEM/32][8], B_ldg_reg[TILEN/32][8];
    Float4 acc[TILEM/32][TILEN/32];
    for(int i = 0; i < TILEM/32; i++){
        for(int j = 0; j < TILEN/32; j++){
            acc[i][j] = (Float4)(0.0f);
        }
    }
    int pos_flt[TILEN/32], pos_act[TILEM/32];
    int pos_h[TILEM/32], pos_w[TILEM/32];
    bool A_ldg_guard[TILEM/32], B_ldg_guard[TILEN/32];

    // smem
    __shared__ char smem[(TILEM + TILEN) * TILEK * sizeof(_Float16)];
    _Float16* smemA = reinterpret_cast<_Float16 *>(smem);
    _Float16* smemB = reinterpret_cast<_Float16 *>(smem + TILEM * TILEK * sizeof(_Float16));

    // lds reg
    Float2 A_lds_reg[TILEM/32][2][2],  B_lds_reg[TILEN/32][2][2];

    // ldg
    int first_tile_k = gemm_k - (((gemm_k + TILEK - 1) / TILEK) - 1) * TILEK;

    int gemm_rowA = by * TILEM + warp_id * TILEM / 4 + lane_div8;
    const int gemm_colA = lane_mod8 * 8;
    int c, rs, r, s;
    for(int i = 0; i < TILEM/32; i++){
        int n, pq, p, q;
        param.divmod_pq(n, pq, gemm_rowA);
        param.divmod_q(p, q, pq);
        // output mapping input index
        pos_h[i] = p - param.p;
        pos_w[i] = q - param.q;

        pos_act[i] = (n * param.h * param.w + pos_h[i] * param.w + pos_w[i]) * param.c;
        A_ldg_guard[i] = n < param.n;
        // param.divmod_rs(c, rs, gemm_colA + j);
        // param.divmod_s(r, s, rs);
        param.divmod_c(rs, c, gemm_colA);
        param.divmod_s(r, s, rs);
        int ih = pos_h[i] + r;
        int iw = pos_w[i] + s;
        int offset = (r * param.w + s) * param.c + c;
        bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w && gemm_colA < first_tile_k;
        *(Float4*)(&(A_ldg_reg[i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
        
        gemm_rowA += 8;
    }

    int gemm_rowB = bx * TILEN + warp_id * TILEN / 4 + lane_div8;
    const int gemm_colB = lane_mod8 * 8;
    for(int i = 0; i < TILEN/32; i++){
        pos_flt[i] = gemm_rowB * gemm_k + gemm_colB;
        B_ldg_guard[i] = gemm_rowB < gemm_m;

        bool guard = B_ldg_guard[i] && gemm_colB < first_tile_k;
        *(Float4*)(&(B_ldg_reg[i][0])) = guard ? *(Float4*)(&(flt_ptr[pos_flt[i]])) : (Float4)(0.0f);
        
        gemm_rowB += 8;
    }

    // swizzle to avoid bank confilct
    int A_sts_addr = warp_id * 64 * TILEM / 4 + lane_div8 * 64 + (lane_mod8 ^ lane_div8) * 8;
    int B_sts_addr = warp_id * 64 * TILEN / 4 + lane_div8 * 64 + (lane_mod8 ^ lane_div8) * 8;
    // sts 0
    for(int i = 0; i < TILEM/32; i++){
        *(Float4*)(&(smemA[A_sts_addr + i * 8 * 64])) = *(Float4*)(&(A_ldg_reg[i][0]));
    }
    for(int i = 0; i < TILEN/32; i++){
        *(Float4*)(&(smemB[B_sts_addr + i * 8 * 64])) = *(Float4*)(&(B_ldg_reg[i][0]));
    }

    __syncthreads();

    // lds addr
    int lds_row = lane_mod16;
    int lds_col0 = (lane_div16 % 8) ^ (lds_row % 8);
    int lds_col1 = ((lane_div16 + 4) % 8) ^ (lds_row % 8);
    int A_lds_addr0 = (warp_mod2 * 64 * TILEM / 2) + lds_row * 64 + lds_col0 * 8;
    int A_lds_addr1 = (warp_mod2 * 64 * TILEM / 2) + lds_row * 64 + lds_col1 * 8;
    
    int B_lds_addr0 = (warp_div2 * 64 * TILEN / 2) + lds_row * 64 + lds_col0 * 8;
    int B_lds_addr1 = (warp_div2 * 64 * TILEN / 2) + lds_row * 64 + lds_col1 * 8;

    // lds0
    for(int i = 0; i < TILEM/32; i++){
        *(Float4*)(&(A_lds_reg[i][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + i * 16 * 64]));
    }
    for(int i = 0; i < TILEN/32; i++){
        *(Float4*)(&(B_lds_reg[i][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + i * 16 * 64]));
    }

    for(int itile = first_tile_k; itile < gemm_k; itile+=TILEK){
        // ldg 1
        for(int i = 0; i < TILEM/32; i++){
            param.divmod_c(rs, c, gemm_colA + itile);
            param.divmod_s(r, s, rs);
            int ih = pos_h[i] + r;
            int iw = pos_w[i] + s;
            int offset = (r * param.w + s) * param.c + c;
            bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w;
            *(Float4*)(&(A_ldg_reg[i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
        }
        for(int i = 0; i < TILEN/32; i++){
            *(Float4*)(&(B_ldg_reg[i][0])) = B_ldg_guard[i] ? *(Float4*)(&(flt_ptr[pos_flt[i] + itile])) : (Float4)(0.0f);
        }

        // lds 1
        for(int i = 0; i < TILEM/32; i++){
            *(Float4*)(&(A_lds_reg[i][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + i * 16 * 64]));
        }
        for(int i = 0; i < TILEN/32; i++){
            *(Float4*)(&(B_lds_reg[i][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + i * 16 * 64]));
        }

        // mma 0
        for(int i = 0; i < TILEM/32; i++){
            for(int j = 0; j < TILEN/32; j++){
                for(int k = 0; k < 2; k++){
                    HMMA161616(acc[i][j], 
                    A_lds_reg[i][0][k], 
                    B_lds_reg[j][0][k]);
                }
            }
        }

        __syncthreads();

        // sts 1
        for(int i = 0; i < TILEM/32; i++){
            *(Float4*)(&(smemA[A_sts_addr + i * 8 * 64])) = *(Float4*)(&(A_ldg_reg[i][0]));
        }
        for(int i = 0; i < TILEN/32; i++){
            *(Float4*)(&(smemB[B_sts_addr + i * 8 * 64])) = *(Float4*)(&(B_ldg_reg[i][0]));
        }
        __syncthreads();
        
        // lds 0
        for(int i = 0; i < TILEM/32; i++){
            *(Float4*)(&(A_lds_reg[i][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + i * 16 * 64]));
        }
        for(int i = 0; i < TILEN/32; i++){
            *(Float4*)(&(B_lds_reg[i][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + i * 16 * 64]));
        }
        // mma 1
        for(int i = 0; i < TILEM/32; i++){
            for(int j = 0; j < TILEN/32; j++){
                for(int k = 0; k < 2; k++){
                    HMMA161616(acc[i][j], 
                    A_lds_reg[i][1][k], 
                    B_lds_reg[j][1][k]);
                }
            }
        }
    }

    // lds 1
    for(int i = 0; i < TILEM/32; i++){
        *(Float4*)(&(A_lds_reg[i][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + i * 16 * 64]));
    }
    for(int i = 0; i < TILEN/32; i++){
        *(Float4*)(&(B_lds_reg[i][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + i * 16 * 64]));
    }

    // mma 0
    for(int i = 0; i < TILEM/32; i++){
        for(int j = 0; j < TILEN/32; j++){
            for(int k = 0; k < 2; k++){
                HMMA161616(acc[i][j], 
                A_lds_reg[i][0][k], 
                B_lds_reg[j][0][k]);
            }
        }
    }

    // mma 1
    for(int i = 0; i < TILEM/32; i++){
        for(int j = 0; j < TILEN/32; j++){
            for(int k = 0; k < 2; k++){
                HMMA161616(acc[i][j], 
                A_lds_reg[i][1][k], 
                B_lds_reg[j][1][k]);
            }
        }
    }

    _Float16* smemC = reinterpret_cast<_Float16 *>(smem);
    // nchw
    for (int j = 0; j < TILEM/32; j++)
    {
        __syncthreads();
        for (int i = 0; i < TILEN/32; i++)
        {
            int C_sts_addr = warp_id * TILEM / 2 * 16 + (lane_id / 16) * TILEM / 2 + i * 16 + (lane_id % 16);
            smemC[C_sts_addr                 ] = (_Float16)acc[i][j].x;
            smemC[C_sts_addr +  4 * TILEM / 2] = (_Float16)acc[i][j].y;
            smemC[C_sts_addr +  8 * TILEM / 2] = (_Float16)acc[i][j].z;
            smemC[C_sts_addr + 12 * TILEM / 2] = (_Float16)acc[i][j].w;
        }
        __syncthreads();
        for (int i = 0; i < 2; i++)
        {
            int C_lds_addr = warp_id * TILEM / 2 * 16 + i * TILEM / 2 * 8 + (lane_id / 8) * 64 + (lane_id % 8) * 8;
            int k = bx * 128 + (warp_id / 2) * 64 + i * 8 + j * 16 + (lane_id / 8);
            int npq = by * 128 + (warp_id % 2) * 64 + (lane_id % 8) * 8;
            int pq, n;
            param.divmod_pq(n, pq, npq);
            bool guard = pq < param.Oh * param.Ow && k < param.k;
            if (guard)
            {
                int offset = n * param.k * param.Oh * param.Ow + k * param.Oh * param.Ow + pq;
                *(Float4 *)(&(param.pout[offset])) = *(Float4 *)(&(smemC[C_lds_addr]));
            }
        }
    }

}

template<int TILEM, int TILEN, int TILEK = 64>
__global__ void myKernelConv2dGputemplate_case1(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;

    int warp_id = tx >> 6;
    int lane_id = tx & 63;

    int warp_div2 = warp_id >> 1;
    int warp_mod2 = warp_id & 1;

    int lane_mod8 = lane_id & 7;
    int lane_div8 = lane_id >> 3;
    int lane_mod16 = lane_id & 15;
    int lane_div16 = lane_id >> 4;

    _Float16 * flt_ptr = param.pweight;
    _Float16 * act_ptr = param.pin;

    int gemm_k = param.c * param.r * param.s;
    int gemm_m = param.k;
    int gemm_n = param.n * param.Oh * param.Ow;

    // ldg reg
    _Float16 A_ldg_reg[TILEM/32][TILEK/8], B_ldg_reg[4][TILEK/8];
    Float4 acc[TILEM/64][2];
    for(int i = 0; i < TILEM/64; i++){
        for(int j = 0; j < 2; j++){
            acc[i][j] = (Float4)(0.0f);
        }
    }
    int pos_flt[4], pos_act[TILEM/32];
    int pos_h[TILEM/32], pos_w[TILEM/32];
    bool A_ldg_guard[TILEM/32], B_ldg_guard[4];

    // smem
    __shared__ char smem[(TILEM + 32) * TILEK * sizeof(_Float16)];
    _Float16* smemA = reinterpret_cast<_Float16 *>(smem);
    _Float16* smemB = reinterpret_cast<_Float16 *>(smem + TILEM * TILEK * sizeof(_Float16));

    // lds reg
    // Float2 A_lds_reg[TILEM/32][2][2],  B_lds_reg[4][2][2];
    Float2 A_lds_reg[TILEM/64][2][2],  B_lds_reg[2][2][2];

    // ldg
    int first_tile_k = gemm_k - (((gemm_k + TILEK - 1) / TILEK) - 1) * TILEK;

    int gemm_rowA = by * TILEM + warp_id * TILEM / 4 + lane_div8;
    const int gemm_colA = lane_mod8 * TILEK / 8;
    int c, rs, r, s;
    for(int i = 0; i < TILEM/32; i++){
        int n, pq, p, q;
        param.divmod_pq(n, pq, gemm_rowA);
        param.divmod_q(p, q, pq);
        // output mapping input index
        pos_h[i] = p - param.p;
        pos_w[i] = q - param.q;

        pos_act[i] = (n * param.h * param.w + pos_h[i] * param.w + pos_w[i]) * param.c;
        A_ldg_guard[i] = n < param.n;
        // param.divmod_rs(c, rs, gemm_colA + j);
        // param.divmod_s(r, s, rs);
        param.divmod_c(rs, c, gemm_colA);
        param.divmod_s(r, s, rs);
        int ih = pos_h[i] + r;
        int iw = pos_w[i] + s;
        int offset = (r * param.w + s) * param.c + c;
        bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w && gemm_colA < first_tile_k;
        *(Float4*)(&(A_ldg_reg[i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
        
        gemm_rowA += 8;
    }

    // int gemm_rowB = bx * TILEN + warp_id * TILEN / 4 + lane_div8;
    int gemm_rowB = lane_div8;
    const int gemm_colB = lane_mod8 * TILEK / 8;
    for(int i = 0; i < 4; i++){
        pos_flt[i] = gemm_rowB * gemm_k + gemm_colB;
        B_ldg_guard[i] = gemm_rowB < gemm_m;

        bool guard = B_ldg_guard[i] && gemm_colB < first_tile_k;
        *(Float4*)(&(B_ldg_reg[i][0])) = guard ? *(Float4*)(&(flt_ptr[pos_flt[i]])) : (Float4)(0.0f);
        
        gemm_rowB += 8;
    }

    // swizzle to avoid bank confilct
    int A_sts_addr = warp_id * TILEK * TILEM / 4 + lane_div8 * TILEK + (lane_mod8 ^ lane_div8) * 8;
    // int B_sts_addr = warp_id * 64 * TILEN / 4 + lane_div8 * 64 + (lane_mod8 ^ lane_div8) * 8;
    int B_sts_addr = lane_div8 * 64 + (lane_mod8 ^ lane_div8) * 8;
    // sts 0
    for(int i = 0; i < TILEM/32; i++){
        *(Float4*)(&(smemA[A_sts_addr + i * 8 * TILEK])) = *(Float4*)(&(A_ldg_reg[i][0]));
    }
    for(int i = 0; i < 4; i++){
        *(Float4*)(&(smemB[B_sts_addr + i * 8 * TILEK])) = *(Float4*)(&(B_ldg_reg[i][0]));
    }

    __syncthreads();

    // lds addr
    int lds_row = lane_mod16;
    int lds_col0 = (lane_div16 % 8) ^ (lds_row % 8);
    int lds_col1 = ((lane_div16 + 4) % 8) ^ (lds_row % 8);
    // int A_lds_addr0 = (warp_mod2 * 64 * TILEM / 2) + lds_row * 64 + lds_col0 * 8;
    // int A_lds_addr1 = (warp_mod2 * 64 * TILEM / 2) + lds_row * 64 + lds_col1 * 8;
    int A_lds_addr0 = (warp_id * TILEK * TILEM / 4) + lds_row * TILEK + lds_col0 * 8;
    int A_lds_addr1 = (warp_id * TILEK * TILEM / 4) + lds_row * TILEK + lds_col1 * 8;
    
    // int B_lds_addr0 = (warp_div2 * 64 * TILEN / 2) + lds_row * 64 + lds_col0 * 8;
    // int B_lds_addr1 = (warp_div2 * 64 * TILEN / 2) + lds_row * 64 + lds_col1 * 8;
    int B_lds_addr0 = lds_row * 64 + lds_col0 * 8;
    int B_lds_addr1 = lds_row * 64 + lds_col1 * 8;

    // lds0
    for(int i = 0; i < TILEM/64; i++){
        *(Float4*)(&(A_lds_reg[i][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + i * 16 * 64]));
    }
    for(int i = 0; i < 2; i++){
        *(Float4*)(&(B_lds_reg[i][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + i * 16 * 64]));
    }

    for(int itile = first_tile_k; itile < gemm_k; itile+=TILEK){
        // ldg 1
        for(int i = 0; i < TILEM/32; i++){
            param.divmod_c(rs, c, gemm_colA + itile);
            param.divmod_s(r, s, rs);
            int ih = pos_h[i] + r;
            int iw = pos_w[i] + s;
            int offset = (r * param.w + s) * param.c + c;
            bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w;
            *(Float4*)(&(A_ldg_reg[i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
        }
        for(int i = 0; i < 4; i++){
            *(Float4*)(&(B_ldg_reg[i][0])) = B_ldg_guard[i] ? *(Float4*)(&(flt_ptr[pos_flt[i] + itile])) : (Float4)(0.0f);
        }

        // lds 1
        for(int i = 0; i < TILEM/64; i++){
            *(Float4*)(&(A_lds_reg[i][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + i * 16 * 64]));
        }
        for(int i = 0; i < 2; i++){
            *(Float4*)(&(B_lds_reg[i][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + i * 16 * 64]));
        }

        // mma 0
        for(int i = 0; i < TILEM/64; i++){
            for(int j = 0; j < 2; j++){
                for(int k = 0; k < 2; k++){
                    HMMA161616(acc[i][j], 
                    A_lds_reg[i][0][k], 
                    B_lds_reg[j][0][k]);
                }
            }
        }

        __syncthreads();

        // sts 1
        for(int i = 0; i < TILEM/32; i++){
            *(Float4*)(&(smemA[A_sts_addr + i * 8 * 64])) = *(Float4*)(&(A_ldg_reg[i][0]));
        }
        for(int i = 0; i < 4; i++){
            *(Float4*)(&(smemB[B_sts_addr + i * 8 * 64])) = *(Float4*)(&(B_ldg_reg[i][0]));
        }
        __syncthreads();
        
        // lds 0
        for(int i = 0; i < TILEM/64; i++){
            *(Float4*)(&(A_lds_reg[i][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + i * 16 * 64]));
        }
        for(int i = 0; i < 2; i++){
            *(Float4*)(&(B_lds_reg[i][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + i * 16 * 64]));
        }
        // mma 1
        for(int i = 0; i < TILEM/64; i++){
            for(int j = 0; j < 2; j++){
                for(int k = 0; k < 2; k++){
                    HMMA161616(acc[i][j], 
                    A_lds_reg[i][1][k], 
                    B_lds_reg[j][1][k]);
                }
            }
        }
    }
//
__syncthreads();
    // lds 1
    for(int i = 0; i < TILEM/64; i++){
        *(Float4*)(&(A_lds_reg[i][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + i * 16 * 64]));
    }
    for(int i = 0; i < 2; i++){
        *(Float4*)(&(B_lds_reg[i][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + i * 16 * 64]));
    }

    // mma 0
    for(int i = 0; i < TILEM/64; i++){
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 2; k++){
                HMMA161616(acc[i][j], 
                A_lds_reg[i][0][k], 
                B_lds_reg[j][0][k]);
            }
        }
    }

    // mma 1
    for(int i = 0; i < TILEM/64; i++){
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 2; k++){
                HMMA161616(acc[i][j], 
                A_lds_reg[i][1][k], 
                B_lds_reg[j][1][k]);
            }
        }
    }
//
__syncthreads();
    _Float16* smemC = reinterpret_cast<_Float16 *>(smem);

    for(int i = 0; i < TILEM/64; i++){
        __syncthreads();
        for(int j = 0; j < 2; j++){
            int C_sts_addr = warp_id * 32 * 16 + (lane_id % 16) * 32 + j * 16 + (lane_id / 16);
            smemC[C_sts_addr     ] = (_Float16)acc[i][j].x;
            smemC[C_sts_addr +  4] = (_Float16)acc[i][j].y;
            smemC[C_sts_addr +  8] = (_Float16)acc[i][j].z;
            smemC[C_sts_addr + 12] = (_Float16)acc[i][j].w;
        }
        __syncthreads();
        
        for(int j = 0; j < 2; j++){
            int C_lds_addr = warp_id * 32 * 16 + j * 32 * 8 + (lane_id / 8) * 32 + (lane_id % 8) * 8;
            // int k = bx * 128 + (warp_id / 2) * 64 + (lane_id % 8) * 8;
            int k = (lane_id % 8) * 8;
            // int npq = by * 128 + (warp_id % 2) * 64 + i * 16 + j * 8 + (lane_id / 8);
            int npq = by * TILEM + warp_id * TILEM / 4 + i * 16 + j * 8 + (lane_id / 8);
            bool guard = npq < param.n * param.Oh * param.Ow && k < param.k;
            if(guard){
                int offset = npq * param.k + k;
                for(int i =0;i<8;i++){
                    if(k+i<27)
                        param.pout[offset+i]=smemC[C_lds_addr+i];
                }
            }
        }
    }
}

template<int TILEM, int TILEN, int TILEK = 32>
__global__ void myKernelConv2dGputemplate_case1_1(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;

    int warp_id = tx >> 6;
    int lane_id = tx & 63;

    int warp_div2 = warp_id >> 1;
    int warp_mod2 = warp_id & 1;

    int lane_mod4 = lane_id & 3;
    int lane_div4 = lane_id >> 2;
    int lane_mod8 = lane_id & 7;
    int lane_div8 = lane_id >> 3;
    int lane_mod16 = lane_id & 15;
    int lane_div16 = lane_id >> 4;

    _Float16 * flt_ptr = param.pweight;
    _Float16 * act_ptr = param.pin;

    int gemm_k = param.c * param.r * param.s;
    int gemm_m = param.k;
    int gemm_n = param.n * param.Oh * param.Ow;

    // ldg reg
    _Float16 A_ldg_reg[TILEM/64][TILEK/4], B_ldg_reg[2][TILEK/4];

    Float4 acc[TILEM/64][2];
    for(int i = 0; i < TILEM/64; i++){
        for(int j = 0; j < 2; j++){
            acc[i][j] = (Float4)(0.0f);
        }
    }
    int pos_flt[2], pos_act[TILEM/64];
    int pos_h[TILEM/64], pos_w[TILEM/64];
    bool A_ldg_guard[TILEM/64], B_ldg_guard[2];

    // smem
    __shared__ char smem[(TILEM + 32) * TILEK * sizeof(_Float16)];
    _Float16* smemA = reinterpret_cast<_Float16 *>(smem);
    _Float16* smemB = reinterpret_cast<_Float16 *>(smem + TILEM * TILEK * sizeof(_Float16));

    // lds reg
    // Float2 A_lds_reg[TILEM/32][2][2],  B_lds_reg[4][2][2];
    Float2 A_lds_reg[TILEM/64][2][2],  B_lds_reg[2][2][2];

    // ldg
    int first_tile_k = gemm_k - (((gemm_k + TILEK - 1) / TILEK) - 1) * TILEK;

    int gemm_rowA = by * TILEM + warp_id * TILEM / 4 + lane_div4;
    const int gemm_colA = lane_mod4 * TILEK / 4;
    int c, rs, r, s;
    for(int i = 0; i < TILEM/64; i++){
        int n, pq, p, q;
        param.divmod_pq(n, pq, gemm_rowA);
        param.divmod_q(p, q, pq);
        // output mapping input index
        pos_h[i] = p - param.p;
        pos_w[i] = q - param.q;

        pos_act[i] = (n * param.h * param.w + pos_h[i] * param.w + pos_w[i]) * param.c;
        A_ldg_guard[i] = n < param.n;
        // param.divmod_rs(c, rs, gemm_colA + j);
        // param.divmod_s(r, s, rs);
        param.divmod_c(rs, c, gemm_colA);
        param.divmod_s(r, s, rs);
        int ih = pos_h[i] + r;
        int iw = pos_w[i] + s;
        int offset = (r * param.w + s) * param.c + c;
        bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w && gemm_colA < first_tile_k;
        *(Float4*)(&(A_ldg_reg[i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
        
        gemm_rowA += 16;
    }

    // int gemm_rowB = bx * TILEN + warp_id * TILEN / 4 + lane_div8;
    int gemm_rowB = lane_div4;
    const int gemm_colB = lane_mod4 * TILEK / 4;
    for(int i = 0; i < 2; i++){
        pos_flt[i] = gemm_rowB * gemm_k + gemm_colB;
        B_ldg_guard[i] = gemm_rowB < gemm_m;

        bool guard = B_ldg_guard[i] && gemm_colB < first_tile_k;
        *(Float4*)(&(B_ldg_reg[i][0])) = guard ? *(Float4*)(&(flt_ptr[pos_flt[i]])) : (Float4)(0.0f);
        
        gemm_rowB += 16;
    }
 
    // swizzle to avoid bank confilct
    int A_sts_addr = warp_id * TILEK * TILEM / 4 + lane_div4 * TILEK + lane_mod4 * 8;
    // int B_sts_addr = warp_id * 64 * TILEN / 4 + lane_div8 * 64 + (lane_mod8 ^ lane_div8) * 8;
    int B_sts_addr = lane_div8 * 64 + lane_mod8 * 8;
    // sts 0
    for(int i = 0; i < TILEM/64; i++){
        *(Float4*)(&(smemA[A_sts_addr + i * 16 * TILEK])) = *(Float4*)(&(A_ldg_reg[i][0]));
    }
    for(int i = 0; i < 2; i++){
        *(Float4*)(&(smemB[B_sts_addr + i * 16 * TILEK])) = *(Float4*)(&(B_ldg_reg[i][0]));
    }

    __syncthreads();

    // lds addr
    int lds_row = lane_mod16;
    int lds_col0 = (lane_div16 % 8);
    // int lds_col1 = ((lane_div16 + 4) % 8);
    // int A_lds_addr0 = (warp_mod2 * 64 * TILEM / 2) + lds_row * 64 + lds_col0 * 8;
    // int A_lds_addr1 = (warp_mod2 * 64 * TILEM / 2) + lds_row * 64 + lds_col1 * 8;
    int A_lds_addr0 = (warp_id * TILEK * TILEM / 4) + lds_row * TILEK + lds_col0 * 8;
    // int A_lds_addr1 = (warp_id * TILEK * TILEM / 4) + lds_row * TILEK + lds_col1 * 8;
    
    // int B_lds_addr0 = (warp_div2 * 64 * TILEN / 2) + lds_row * 64 + lds_col0 * 8;
    // int B_lds_addr1 = (warp_div2 * 64 * TILEN / 2) + lds_row * 64 + lds_col1 * 8;
    int B_lds_addr0 = lds_row * TILEK + lds_col0 * 8;
    // int B_lds_addr1 = lds_row * TILEK + lds_col1 * 8;

    // lds0
    for(int i = 0; i < TILEM/64; i++){
        *(Float4*)(&(A_lds_reg[i][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + i * 16 * 64]));
    }
    for(int i = 0; i < 2; i++){
        *(Float4*)(&(B_lds_reg[i][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + i * 16 * 64]));
    }

    for(int itile = first_tile_k; itile < gemm_k; itile+=TILEK){
        // ldg 1
        for(int i = 0; i < TILEM/64; i++){
            param.divmod_c(rs, c, gemm_colA + itile);
            param.divmod_s(r, s, rs);
            int ih = pos_h[i] + r;
            int iw = pos_w[i] + s;
            int offset = (r * param.w + s) * param.c + c;
            bool guard = A_ldg_guard[i] && ih >= 0 && ih < param.h && iw >= 0 && iw < param.w;
            *(Float4*)(&(A_ldg_reg[i][0])) = guard ? *(Float4*)(&(act_ptr[pos_act[i] + offset])) : (Float4)(0.0f);
        }
        for(int i = 0; i < 2; i++){
            *(Float4*)(&(B_ldg_reg[i][0])) = B_ldg_guard[i] ? *(Float4*)(&(flt_ptr[pos_flt[i] + itile])) : (Float4)(0.0f);
        }

        // lds 1
        // for(int i = 0; i < TILEM/64; i++){
        //     *(Float4*)(&(A_lds_reg[i][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + i * 16 * 64]));
        // }
        // for(int i = 0; i < 2; i++){
        //     *(Float4*)(&(B_lds_reg[i][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + i * 16 * 64]));
        // }

        // mma 0
        for(int i = 0; i < TILEM/64; i++){
            for(int j = 0; j < 2; j++){
                for(int k = 0; k < 2; k++){
                    HMMA161616(acc[i][j], 
                    A_lds_reg[i][0][k], 
                    B_lds_reg[j][0][k]);
                }
            }
        }

        __syncthreads();

        // sts 1
        for(int i = 0; i < TILEM/64; i++){
            *(Float4*)(&(smemA[A_sts_addr + i * 8 * 64])) = *(Float4*)(&(A_ldg_reg[i][0]));
        }
        for(int i = 0; i < 2; i++){
            *(Float4*)(&(smemB[B_sts_addr + i * 8 * 64])) = *(Float4*)(&(B_ldg_reg[i][0]));
        }
        __syncthreads();
        
        // lds 0
        for(int i = 0; i < TILEM/64; i++){
            *(Float4*)(&(A_lds_reg[i][0][0])) = *(Float4*)(&(smemA[A_lds_addr0 + i * 16 * 64]));
        }
        for(int i = 0; i < 2; i++){
            *(Float4*)(&(B_lds_reg[i][0][0])) = *(Float4*)(&(smemB[B_lds_addr0 + i * 16 * 64]));
        }
        // mma 1
        // for(int i = 0; i < TILEM/64; i++){
        //     for(int j = 0; j < 2; j++){
        //         for(int k = 0; k < 2; k++){
        //             HMMA161616(acc[i][j], 
        //             A_lds_reg[i][1][k], 
        //             B_lds_reg[j][1][k]);
        //         }
        //     }
        // }
    }
//
__syncthreads();
    // lds 1
    // for(int i = 0; i < TILEM/64; i++){
    //     *(Float4*)(&(A_lds_reg[i][1][0])) = *(Float4*)(&(smemA[A_lds_addr1 + i * 16 * 64]));
    // }
    // for(int i = 0; i < 2; i++){
    //     *(Float4*)(&(B_lds_reg[i][1][0])) = *(Float4*)(&(smemB[B_lds_addr1 + i * 16 * 64]));
    // }

    // mma 0
    for(int i = 0; i < TILEM/64; i++){
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 2; k++){
                HMMA161616(acc[i][j], 
                A_lds_reg[i][0][k], 
                B_lds_reg[j][0][k]);
            }
        }
    }

    // mma 1
    // for(int i = 0; i < TILEM/64; i++){
    //     for(int j = 0; j < 2; j++){
    //         for(int k = 0; k < 2; k++){
    //             HMMA161616(acc[i][j], 
    //             A_lds_reg[i][1][k], 
    //             B_lds_reg[j][1][k]);
    //         }
    //     }
    // }
//
__syncthreads();
    _Float16* smemC = reinterpret_cast<_Float16 *>(smem);

    for(int i = 0; i < TILEM/64; i++){
        __syncthreads();
        for(int j = 0; j < 2; j++){
            int C_sts_addr = warp_id * 32 * 16 + (lane_id % 16) * 32 + j * 16 + (lane_id / 16);
            smemC[C_sts_addr     ] = (_Float16)acc[i][j].x;
            smemC[C_sts_addr +  4] = (_Float16)acc[i][j].y;
            smemC[C_sts_addr +  8] = (_Float16)acc[i][j].z;
            smemC[C_sts_addr + 12] = (_Float16)acc[i][j].w;
        }
        __syncthreads();
        
        for(int j = 0; j < 2; j++){
            int C_lds_addr = warp_id * 32 * 16 + j * 32 * 8 + (lane_id / 8) * 32 + (lane_id % 8) * 8;
            // int k = bx * 128 + (warp_id / 2) * 64 + (lane_id % 8) * 8;
            int k = (lane_id % 8) * 8;
            // int npq = by * 128 + (warp_id % 2) * 64 + i * 16 + j * 8 + (lane_id / 8);
            int npq = by * TILEM + warp_id * TILEM / 4 + i * 16 + j * 8 + (lane_id / 8);
            bool guard = npq < param.n * param.Oh * param.Ow && k < param.k;
            if(guard){
                int offset = npq * param.k + k;
                for(int i =0;i<8;i++){
                    if(k+i<27)
                        param.pout[offset+i]=smemC[C_lds_addr+i];
                }
            }
        }
    }
}


template <int TILEM = 32, int TILEN = 32, int TILEK = 64, int STAGE = 1>
__global__ void myKernelConv2dGpu32x32x64_256(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1, 256)))
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;

    const int warp_id = tx >> 6;
    const int lane_id = tx & 63;

    int warp_div2 = warp_id >> 1;
    int warp_mod2 = warp_id & 1;

    int lane_mod8 = lane_id & 7;
    int lane_div8 = lane_id >> 3;
    int lane_mod16 = lane_id & 15;
    int lane_div16 = lane_id >> 4;

    _Float16 *flt_ptr = param.pweight;
    _Float16 *act_ptr = param.pin;
    BB globalReadA;
    globalReadA.x = (long)(param.pin);
    globalReadA.y = (((((long )0x20000)<<32) | 0x80000000));
    BB globalReadB;
    globalReadB.x = (long)(param.pweight);
    globalReadB.y = (((((long )0x20000)<<32) | 0x80000000));


    int gemm_k = param.c * param.r * param.s;
    int gemm_m = param.k;
    int gemm_n = param.n * param.Oh * param.Ow;

    // ldg reg
    Float4 A_ldg_reg[TILEM / 32][STAGE], B_ldg_reg[TILEN / 32][STAGE];
    Float4 acc[TILEM / 32][TILEN / 32];
    for (int i = 0; i < TILEM / 32; i++)
    {
        for (int j = 0; j < TILEN / 32; j++)
        {
            acc[i][j] = (Float4)(0.0f);
        }
    }
    int pos_flt[TILEN / 32], pos_act[TILEM / 32];
    int pos_h[TILEM / 32], pos_w[TILEM / 32];
    bool A_ldg_guard[TILEM / 32], B_ldg_guard[TILEN / 32];

    // smem
    __shared__ char smem[(TILEM + TILEN) * TILEK * 1 * sizeof(_Float16)];
    _Float16 *smemA = reinterpret_cast<_Float16 *>(smem);
    _Float16 *smemB = reinterpret_cast<_Float16 *>(smem + TILEM * TILEK * sizeof(_Float16));

    // lds reg
    Float2 A_lds_reg[TILEM / 32][2][2], B_lds_reg[TILEN / 32][2][2];

    int ldg_num = 0;
    int sts_num = 0;
    int lds_num = 0;
    // ldg
    int first_tile_k = gemm_k - (((gemm_k + TILEK - 1) / TILEK) - 1) * TILEK;

    int gemm_rowA = by * TILEM + warp_id * TILEM / 4 + lane_div8;
    int gemm_rowB = bx * TILEN + warp_id * TILEN / 4 + lane_div8;
    const int gemm_col = lane_mod8 * 8;
    int c, rs, r, s;
    for (int i = 0; i < TILEM / 32; i++)
    {
        int n, pq, p, q;
        param.divmod_pq(n, pq, gemm_rowA);
        param.divmod_q(p, q, pq);
        // output mapping input index
        pos_h[i] = p - param.p;
        pos_w[i] = q - param.q;
        pos_act[i] = (n * param.h * param.w + pos_h[i] * param.w + pos_w[i]) * param.c + gemm_col;
        gemm_rowA += 8;
    }

    for (int i = 0; i < TILEN / 32; i++)
    {
        pos_flt[i] = gemm_rowB * gemm_k + gemm_col;
        B_ldg_guard[i] = gemm_rowB < gemm_m;
        gemm_rowB += 8;
    }


    for (int i = 0; i < TILEM / 32; i++)
    {
        bool guard = pos_h[i] >= 0 && pos_h[i] < param.h && pos_w[i] >= 0 && pos_w[i] < param.w;
        int offset = guard ? pos_act[i] * 2 : -1;
        BUF_LDG128(*(Float4 *)(&(A_ldg_reg[i][0])), globalReadA, offset);

    }
    for (int i = 0; i < TILEN / 32; i++)
    {
        int offset = B_ldg_guard[i] ? pos_flt[i] * 2 : -1;
        BUF_LDG128(*(Float4 *)(&(B_ldg_reg[i][0])), globalReadB, offset);
    }

    CG_SYNC(0);
    // swizzle to avoid bank confilct
    int A_sts_addr = warp_id * 64 * TILEM / 4 + lane_div8 * 64 + (lane_mod8 ^ lane_div8) * 8;
    int B_sts_addr = warp_id * 64 * TILEN / 4 + lane_div8 * 64 + (lane_mod8 ^ lane_div8) * 8;

    for (int i = 0; i < TILEM / 32; i++)
    {
        *(Float4 *)(&(smemA[A_sts_addr + i * 8 * 64])) = *(Float4 *)(&(A_ldg_reg[i][0]));
    }
    for (int i = 0; i < TILEN / 32; i++)
    {
        *(Float4 *)(&(smemB[B_sts_addr + i * 8 * 64])) = *(Float4 *)(&(B_ldg_reg[i][0]));
    }
    
    __syncthreads();

    // lds addr
    int A_lds_addr[2], B_lds_addr[2];
    int lds_row = (lane_id % 16);
    int lds_col0 = ((lane_id / 16) % 8) ^ (lds_row % 8);
    int lds_col1 = (((lane_id / 16) + 4) % 8) ^ (lds_row % 8);
    A_lds_addr[0] = ((warp_id % 2) * 64 * TILEM / 2) + lds_row * 64 + lds_col0 * 8;
    B_lds_addr[0] = ((warp_id / 2) * 64 * TILEN / 2) + lds_row * 64 + lds_col0 * 8;
    A_lds_addr[1] = ((warp_id % 2) * 64 * TILEM / 2) + lds_row * 64 + lds_col1 * 8;
    B_lds_addr[1] = ((warp_id / 2) * 64 * TILEN / 2) + lds_row * 64 + lds_col1 * 8;
    // lds0
    for (int i = 0; i < TILEM / 32; i++)
    {
        *(Float4 *)(&(A_lds_reg[i][0][0])) = *(Float4 *)(&(smemA[A_lds_addr[0] + i * 16 * 64]));
    }
    for (int i = 0; i < TILEN / 32; i++)
    {
        *(Float4 *)(&(B_lds_reg[i][0][0])) = *(Float4 *)(&(smemB[B_lds_addr[0] + i * 16 * 64]));
    }

    for (int rtile = 0; rtile < 3; rtile++)
    {
        for (int stile = 0; stile < 3; stile++)
        {
            for (int ctile = 0; ctile < param.c; ctile += 64)
            {
                if (ctile == 0 && rtile == 0 && stile == 0)
                    continue;
                // ldg 1
                int act_offset = (rtile * param.w + stile) * param.c + ctile;
                int flt_offset = (rtile * 3 + stile) * param.c + ctile;
                for (int i = 0; i < TILEM / 32; i++)
                {
                    int ih = pos_h[i] + rtile;
                    int iw = pos_w[i] + stile;
                    bool guard = ih >= 0 && ih < param.h && iw >= 0 && iw < param.w;
                    int offset = guard ? (pos_act[i] + act_offset) * 2 : -1;
                    BUF_LDG128(*(Float4 *)(&(A_ldg_reg[i][0])), globalReadA, offset);
                }
                for (int i = 0; i < TILEN / 32; i++)
                {
                    int offset = B_ldg_guard[i] ? (pos_flt[i] + flt_offset) * 2 : -1;
                    BUF_LDG128(*(Float4 *)(&(B_ldg_reg[i][0])), globalReadB, offset);
                }
#pragma unroll
                for (int subk = 0; subk < 2; ++subk)
                {
                    if (subk == 1)
                    {
                        CG_SYNC(0);
                        __syncthreads();
                        // sts 1
                        for (int i = 0; i < TILEM / 32; i++)
                        {
                            *(Float4 *)(&(smemA[A_sts_addr + i * 8 * 64])) = *(Float4 *)(&(A_ldg_reg[i][0]));
                        }
                        for (int i = 0; i < TILEN / 32; i++)
                        {
                            *(Float4 *)(&(smemB[B_sts_addr + i * 8 * 64])) = *(Float4 *)(&(B_ldg_reg[i][0]));
                        }
                        __syncthreads();

                    }
                    // lds 1
                    for (int i = 0; i < TILEM / 32; i++)
                    {
                        *(Float4 *)(&(A_lds_reg[i][(subk + 1) % 2][0])) = *(Float4 *)(&(smemA[lds_num * TILEM * TILEK + A_lds_addr[(subk + 1) % 2] + i * 16 * 64]));
                    }
                    for (int i = 0; i < TILEN / 32; i++)
                    {
                        *(Float4 *)(&(B_lds_reg[i][(subk + 1) % 2][0])) = *(Float4 *)(&(smemB[lds_num * TILEN * TILEK + B_lds_addr[(subk + 1) % 2] + i * 16 * 64]));
                    }

                    // mma 0
                    for (int i = 0; i < TILEM / 32; i++)
                    {
                        for (int j = 0; j < TILEN / 32; j++)
                        {
                            for (int k = 0; k < 2; k++)
                            {
                                HMMA161616(acc[i][j],
                                           A_lds_reg[i][subk][k],
                                           B_lds_reg[j][subk][k]);
                            }
                        }
                    }
                }
            }
        }
    }
    // lds 1
    for (int i = 0; i < TILEM / 32; i++)
    {
        *(Float4 *)(&(A_lds_reg[i][1][0])) = *(Float4 *)(&(smemA[A_lds_addr[1] + i * 16 * 64]));
    }
    for (int i = 0; i < TILEN / 32; i++)
    {
        *(Float4 *)(&(B_lds_reg[i][1][0])) = *(Float4 *)(&(smemB[B_lds_addr[1] + i * 16 * 64]));
    }

    // mma 0
    for (int i = 0; i < TILEM / 32; i++)
    {
        for (int j = 0; j < TILEN / 32; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                HMMA161616(acc[i][j],
                           A_lds_reg[i][0][k],
                           B_lds_reg[j][0][k]);
            }
        }
    }

    // mma 1
    for (int i = 0; i < TILEM / 32; i++)
    {
        for (int j = 0; j < TILEN / 32; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                HMMA161616(acc[i][j],
                           A_lds_reg[i][1][k],
                           B_lds_reg[j][1][k]);
            }
        }
    }

    for (int i = 0; i < TILEM / 32; i++)
    {
        for (int j = 0; j < TILEN / 32; j++)
        {
            int npq, k, n, pq;
            // param.divmod_pq(n, pq, by * 128 + warp_mod2 * 64 + i * 16 + lane_mod16);
            npq = by * TILEM + warp_mod2 * TILEM / 2 + i * 16 + lane_mod16;
            k = bx * TILEN + warp_div2 * TILEN / 2 + j * 16 + lane_div16;
            param.divmod_pq(n, pq, npq);
            // k = by * 128 + warp_mod2 * 64 + i * 16 + lane_mod16;
            bool guard = pq < param.Oh * param.Ow;
            if (guard)
            {
                int offset = n * param.k * param.Oh * param.Ow + k * param.Oh * param.Ow + pq;
                // int offset = npq * param.k + k;
                if (k < param.k)
                    param.pout[offset] = (_Float16)acc[i][j].x;
                if (k + 4 < param.k)
                    param.pout[offset + 4 * param.Oh * param.Ow] = (_Float16)acc[i][j].y;
                if (k + 8 < param.k)
                    param.pout[offset + 8 * param.Oh * param.Ow] = (_Float16)acc[i][j].z;
                if (k + 12 < param.k)
                    param.pout[offset + 12 * param.Oh * param.Ow] = (_Float16)acc[i][j].w;
            }
        }
    }
}

__global__ void myKernelConv2dGpu128x128x64_256_splitK(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1, 256)))
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    // int32_t mask;
    int pos_flt[4], pos_act[4];
    int pos_h[4], pos_w[4];
    bool A_ldg_guard[4], B_ldg_guard[4];

    int warp_id = tx / 64;
    int lane_id = tx % 64;

    _Float16 *flt_ptr = param.pweight;
    _Float16 *act_ptr = param.pin;
    BB globalReadA;
    globalReadA.x = (long)(param.pin);
    globalReadA.y = (((((long )0x20000)<<32) | 0x80000000));
    BB globalReadB;
    globalReadB.x = (long)(param.pweight);
    globalReadB.y = (((((long )0x20000)<<32) | 0x80000000));

    int gemm_k = param.c * param.r * param.s;
    int gemm_m = param.k;
    int gemm_n = param.n * param.Oh * param.Ow;

    // ldg reg
    Float4 A_ldg_reg[4], B_ldg_reg[4];
    Float4 acc[4][4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            acc[i][j] = (Float4)(0.0f);
        }
    }
    // smem
    __shared__ char smem[128 * 64 * 2 * sizeof(_Float16)];
    _Float16 *smemA = reinterpret_cast<_Float16 *>(smem);
    _Float16 *smemB = reinterpret_cast<_Float16 *>(smem + 128 * 64 * sizeof(_Float16));

    int A_sts_addr = warp_id * 64 * 32 + (lane_id / 8) * 64 + ((lane_id % 8) ^ (lane_id / 8)) * 8;
    int B_sts_addr = warp_id * 64 * 32 + (lane_id / 8) * 64 + ((lane_id % 8) ^ (lane_id / 8)) * 8;
    // lds reg
    Float2 A_lds_reg[4][2][2], B_lds_reg[4][2][2];

    // lds addr
    int A_lds_addr[2], B_lds_addr[2];
    int lds_row = (lane_id % 16);
    int lds_col0 = ((lane_id / 16) % 8) ^ (lds_row % 8);
    int lds_col1 = (((lane_id / 16) + 4) % 8) ^ (lds_row % 8);
    A_lds_addr[0] = ((warp_id % 2) * 64 * 64) + lds_row * 64 + lds_col0 * 8;
    B_lds_addr[0] = ((warp_id / 2) * 64 * 64) + lds_row * 64 + lds_col0 * 8;
    A_lds_addr[1] = ((warp_id % 2) * 64 * 64) + lds_row * 64 + lds_col1 * 8;
    B_lds_addr[1] = ((warp_id / 2) * 64 * 64) + lds_row * 64 + lds_col1 * 8;

    // ldg
    int gemm_rowA = by * 128 + warp_id * 32 + (lane_id / 8);
    int gemm_rowB = bx * 128 + warp_id * 32 + (lane_id / 8);
    const int gemm_col = (lane_id % 8) * 8;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int n, pq, p, q;
        param.divmod_pq(n, pq, gemm_rowA);
        param.divmod_q(p, q, pq);
        // output mapping input index
        pos_h[i] = p - param.p;
        pos_w[i] = q - param.q;
        pos_act[i] = (n * param.h * param.w + (pos_h[i] + bz) * param.w + pos_w[i]) * param.c + gemm_col;
        bool guard = (pos_h[i] + bz) >= 0 && (pos_h[i] + bz) < param.h && pos_w[i] >= 0 && pos_w[i] < param.w;
        int offset = guard ? pos_act[i] * 2 : -1;
        BUF_LDG128(*(Float4 *)(&(A_ldg_reg[i])), globalReadA, offset);
        gemm_rowA += 8;

        pos_flt[i] = gemm_rowB * gemm_k + gemm_col;
        B_ldg_guard[i] = gemm_rowB < gemm_m;
        offset = B_ldg_guard[i] ? pos_flt[i] * 2 : -1;
        BUF_LDG128(*(Float4 *)(&(B_ldg_reg[i])), globalReadB, offset);
        gemm_rowB += 8;
    }

    CG_SYNC(0);
// sts 0
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(Float4 *)(&(smemA[A_sts_addr + i * 8 * 64])) = A_ldg_reg[i];
        *(Float4 *)(&(smemB[B_sts_addr + i * 8 * 64])) = B_ldg_reg[i];
    }
    // read_flag ^= 1;
    __syncthreads();

// lds0
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(Float4 *)(&(A_lds_reg[i][0][0])) = *(Float4 *)(&(smemA[A_lds_addr[0] + i * 16 * 64]));
        *(Float4 *)(&(B_lds_reg[i][0][0])) = *(Float4 *)(&(smemB[B_lds_addr[0] + i * 16 * 64]));
    }

    for (int rtile = 0; rtile < 3 / param.splitKNum; rtile++)
    {
        for (int stile = 0; stile < 3; stile++)
        {
            for (int ctile = 0; ctile < param.c; ctile += 64)
            {
                if (ctile == 0 && stile == 0)
                    continue;
                // ldg 1
                int act_offset = (stile) * param.c + ctile;
                int flt_offset = ((bz + rtile) * 3 + stile) * param.c + ctile;
                for (int i = 0; i < 4; i++)
                {
                    int ih = pos_h[i] + bz;
                    int iw = pos_w[i] + stile;
                    bool guard = ih >= 0 && ih < param.h && iw >= 0 && iw < param.w;
                    int offset = guard ? (pos_act[i] + act_offset) * 2 : -1;
                    BUF_LDG128(*(Float4 *)(&(A_ldg_reg[i])), globalReadA, offset);
                    offset = B_ldg_guard[i] ? (pos_flt[i] + flt_offset) * 2 : -1;
                    BUF_LDG128(*(Float4 *)(&(B_ldg_reg[i])), globalReadB, offset);
                }
#pragma unroll
                for (int subk = 0; subk < 2; ++subk)
                {
                    if (subk == 1)
                    {
                        CG_SYNC(0);
                        __syncthreads();
#pragma unroll
                        for (int i = 0; i < 4; i++)
                        {
                            *(Float4 *)(&(smemA[A_sts_addr + i * 8 * 64])) = A_ldg_reg[i];
                            *(Float4 *)(&(smemB[B_sts_addr + i * 8 * 64])) = B_ldg_reg[i];
                        }
                        __syncthreads();
                    }
// lds
#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        *(Float4 *)(&(A_lds_reg[i][(subk + 1) % 2][0])) = *(Float4 *)(&(smemA[A_lds_addr[(subk + 1) % 2] + i * 16 * 64]));
                        *(Float4 *)(&(B_lds_reg[i][(subk + 1) % 2][0])) = *(Float4 *)(&(smemB[B_lds_addr[(subk + 1) % 2] + i * 16 * 64]));
                    }
                    // mma
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            for (int k = 0; k < 2; k++)
                            {
                                HMMA161616(acc[i][j],
                                           A_lds_reg[i][subk][k],
                                           B_lds_reg[j][subk][k]);
                            }
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(Float4 *)(&(A_lds_reg[i][1][0])) = *(Float4 *)(&(smemA[A_lds_addr[1] + i * 16 * 64]));
        *(Float4 *)(&(B_lds_reg[i][1][0])) = *(Float4 *)(&(smemB[B_lds_addr[1] + i * 16 * 64]));
    }

    // mma 0
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                HMMA161616(acc[i][j],
                           A_lds_reg[i][0][k],
                           B_lds_reg[j][0][k]);
            }
        }
    }

    // mma 1
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                HMMA161616(acc[i][j],
                           A_lds_reg[i][1][k],
                           B_lds_reg[j][1][k]);
            }
        }
    }
    _Float16 *smemC = reinterpret_cast<_Float16 *>(smem);
    // nchw
    for (int j = 0; j < 4; j++)
    {
        __syncthreads();
        for (int i = 0; i < 4; i++)
        {
            int C_sts_addr = warp_id * 64 * 16 + (lane_id / 16) * 64 + i * 16 + (lane_id % 16);
            smemC[C_sts_addr] = (_Float16)acc[i][j].x;
            smemC[C_sts_addr + 4 * 64] = (_Float16)acc[i][j].y;
            smemC[C_sts_addr + 8 * 64] = (_Float16)acc[i][j].z;
            smemC[C_sts_addr + 12 * 64] = (_Float16)acc[i][j].w;
        }
        __syncthreads();
        for (int i = 0; i < 2; i++)
        {
            int C_lds_addr = warp_id * 64 * 16 + i * 64 * 8 + (lane_id / 8) * 64 + (lane_id % 8) * 8;
            int k = bx * 128 + (warp_id / 2) * 64 + i * 8 + j * 16 + (lane_id / 8);
            int npq = by * 128 + (warp_id % 2) * 64 + (lane_id % 8) * 8;
            int pq, n;
            param.divmod_pq(n, pq, npq);
            bool guard = pq < param.Oh * param.Ow && k < param.k;
            if (guard)
            {
                int offset = bz * param.n * param.k * param.Oh * param.Ow + n * param.k * param.Oh * param.Ow + k * param.Oh * param.Ow + pq;
                *(Float4 *)(&(param.pout[offset])) = *(Float4 *)(&(smemC[C_lds_addr]));
            }
        }
    }
}

__global__ void myKernelConv2dGpu128x128x64_256_1(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1, 256)))
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    // int32_t mask;
    int pos_flt[4], pos_act[4];
    int pos_h[4], pos_w[4];
    bool A_ldg_guard[4], B_ldg_guard[4];

    int warp_id = tx / 64;
    int lane_id = tx % 64;

    _Float16 *flt_ptr = param.pweight;
    _Float16 *act_ptr = param.pin;
    BB globalReadA;
    globalReadA.x = (long)(param.pin);
    globalReadA.y = (((((long )0x20000)<<32) | 0x80000000));
    BB globalReadB;
    globalReadB.x = (long)(param.pweight);
    globalReadB.y = (((((long )0x20000)<<32) | 0x80000000));

    int gemm_k = param.c * param.r * param.s;
    int gemm_m = param.k;
    int gemm_n = param.n * param.Oh * param.Ow;

    // ldg reg
    Float4 A_ldg_reg[4], B_ldg_reg[4];
    Float4 acc[4][4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            acc[i][j] = (Float4)(0.0f);
        }
    }
    // smem
    __shared__ char smem[128 * 64 * 2 * sizeof(_Float16)];
    _Float16 *smemA = reinterpret_cast<_Float16 *>(smem);
    _Float16 *smemB = reinterpret_cast<_Float16 *>(smem + 128 * 64 * sizeof(_Float16));

    int A_sts_addr = warp_id * 64 * 32 + (lane_id / 8) * 64 + ((lane_id % 8) ^ (lane_id / 8)) * 8;
    int B_sts_addr = warp_id * 64 * 32 + (lane_id / 8) * 64 + ((lane_id % 8) ^ (lane_id / 8)) * 8;
    // lds reg
    Float2 A_lds_reg[4][2][2], B_lds_reg[4][2][2];

    // lds addr
    int A_lds_addr[2], B_lds_addr[2];
    int lds_row = (lane_id % 16);
    int lds_col0 = ((lane_id / 16) % 8) ^ (lds_row % 8);
    int lds_col1 = (((lane_id / 16) + 4) % 8) ^ (lds_row % 8);
    A_lds_addr[0] = ((warp_id % 2) * 64 * 64) + lds_row * 64 + lds_col0 * 8;
    B_lds_addr[0] = ((warp_id / 2) * 64 * 64) + lds_row * 64 + lds_col0 * 8;
    A_lds_addr[1] = ((warp_id % 2) * 64 * 64) + lds_row * 64 + lds_col1 * 8;
    B_lds_addr[1] = ((warp_id / 2) * 64 * 64) + lds_row * 64 + lds_col1 * 8;

    // ldg
    int gemm_rowA = by * 128 + warp_id * 32 + (lane_id / 8);
    int gemm_rowB = bx * 128 + warp_id * 32 + (lane_id / 8);
    const int gemm_col = (lane_id % 8) * 8;
    Conv2dFpropFilterTileAccessIteratorOptimized iteratorB(param, gemm_rowB, gemm_col, 4);
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int n, pq, p, q;
        param.divmod_pq(n, pq, gemm_rowA);
        param.divmod_q(p, q, pq);
        // output mapping input index
        pos_h[i] = p - param.p;
        pos_w[i] = q - param.q;
        pos_act[i] = (n * param.h * param.w + pos_h[i] * param.w + pos_w[i]) * param.c + gemm_col;
        bool guard = pos_h[i] >= 0 && pos_h[i] < param.h && pos_w[i] >= 0 && pos_w[i] < param.w;
        int offset = guard ? pos_act[i] * 2 : -1;
        BUF_LDG128(*(Float4 *)(&(A_ldg_reg[i])), globalReadA, offset);
            
        pos_flt[i] = gemm_rowB * gemm_k + gemm_col;
        B_ldg_guard[i] = gemm_rowB < gemm_m;
        offset = B_ldg_guard[i] ? pos_flt[i] * 2 : -1;
        BUF_LDG128(*(Float4 *)(&(B_ldg_reg[i])), globalReadB, offset);
        
        gemm_rowA += 8;
        gemm_rowB += 8;
    }
    CG_SYNC(0);
// sts 0
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(Float4 *)(&(smemA[A_sts_addr + i * 8 * 64])) = A_ldg_reg[i];
        *(Float4 *)(&(smemB[B_sts_addr + i * 8 * 64])) = B_ldg_reg[i];
    }
    __syncthreads();

// lds0
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(Float4 *)(&(A_lds_reg[i][0][0])) = *(Float4 *)(&(smemA[A_lds_addr[0] + i * 16 * 64]));
        *(Float4 *)(&(B_lds_reg[i][0][0])) = *(Float4 *)(&(smemB[B_lds_addr[0] + i * 16 * 64]));
    }

    for (int rtile = 0; rtile < 3; rtile++)
    {
        for (int stile = 0; stile < 3; stile++)
        {
            for (int ctile = 0; ctile < param.c; ctile += 64)
            {
                if (ctile == 0 && rtile == 0 && stile == 0)
                    continue;
                // ldg 1
                int act_offset = (rtile * param.w + stile) * param.c + ctile;
                int flt_offset = (rtile * 3 + stile) * param.c + ctile;
                for (int i = 0; i < 4; i++)
                {
                    int ih = pos_h[i] + rtile;
                    int iw = pos_w[i] + stile;
                    bool guard = ih >= 0 && ih < param.h && iw >= 0 && iw < param.w;
                    int offset = guard ? (pos_act[i] + act_offset) * 2 : -1;
                    BUF_LDG128(*(Float4 *)(&(A_ldg_reg[i])), globalReadA, offset);
                    offset = B_ldg_guard[i] ? (pos_flt[i] + flt_offset) * 2 : -1;
                    BUF_LDG128(*(Float4 *)(&(B_ldg_reg[i])), globalReadB, offset);
                }

#pragma unroll
                for (int subk = 0; subk < 2; ++subk)
                {
                    if (subk == 1)
                    {
                        CG_SYNC(0);
                        __syncthreads();
#pragma unroll
                        for (int i = 0; i < 4; i++)
                        {
                            *(Float4 *)(&(smemA[A_sts_addr + i * 8 * 64])) = A_ldg_reg[i];
                            *(Float4 *)(&(smemB[B_sts_addr + i * 8 * 64])) = B_ldg_reg[i];
                        }
                        __syncthreads();
                    }
// lds
#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        *(Float4 *)(&(A_lds_reg[i][(subk + 1) % 2][0])) = *(Float4 *)(&(smemA[A_lds_addr[(subk + 1) % 2] + i * 16 * 64]));
                        *(Float4 *)(&(B_lds_reg[i][(subk + 1) % 2][0])) = *(Float4 *)(&(smemB[B_lds_addr[(subk + 1) % 2] + i * 16 * 64]));
                    }
                    // mma
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            for (int k = 0; k < 2; k++)
                            {
                                HMMA161616(acc[i][j],
                                           A_lds_reg[i][subk][k],
                                           B_lds_reg[j][subk][k]);
                            }
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(Float4 *)(&(A_lds_reg[i][1][0])) = *(Float4 *)(&(smemA[A_lds_addr[1] + i * 16 * 64]));
        *(Float4 *)(&(B_lds_reg[i][1][0])) = *(Float4 *)(&(smemB[B_lds_addr[1] + i * 16 * 64]));
    }

    // mma 0
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                HMMA161616(acc[i][j],
                           A_lds_reg[i][0][k],
                           B_lds_reg[j][0][k]);
            }
        }
    }

    // mma 1
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                HMMA161616(acc[i][j],
                           A_lds_reg[i][1][k],
                           B_lds_reg[j][1][k]);
            }
        }
    }
    _Float16 *smemC = reinterpret_cast<_Float16 *>(smem);
    // nchw
    for (int j = 0; j < 4; j++)
    {
        __syncthreads();
        for (int i = 0; i < 4; i++)
        {
            int C_sts_addr = warp_id * 64 * 16 + (lane_id / 16) * 64 + i * 16 + (lane_id % 16);
            smemC[C_sts_addr] = (_Float16)acc[i][j].x;
            smemC[C_sts_addr + 4 * 64] = (_Float16)acc[i][j].y;
            smemC[C_sts_addr + 8 * 64] = (_Float16)acc[i][j].z;
            smemC[C_sts_addr + 12 * 64] = (_Float16)acc[i][j].w;
        }
        __syncthreads();
        for (int i = 0; i < 2; i++)
        {
            int C_lds_addr = warp_id * 64 * 16 + i * 64 * 8 + (lane_id / 8) * 64 + (lane_id % 8) * 8;
            int k = bx * 128 + (warp_id / 2) * 64 + i * 8 + j * 16 + (lane_id / 8);
            int npq = by * 128 + (warp_id % 2) * 64 + (lane_id % 8) * 8;
            int pq, n;
            param.divmod_pq(n, pq, npq);
            bool guard = pq < param.Oh * param.Ow && k < param.k;
            if (guard)
            {
                int offset = bz * param.n * param.k * param.Oh * param.Ow + n * param.k * param.Oh * param.Ow + k * param.Oh * param.Ow + pq;
                *(Float4 *)(&(param.pout[offset])) = *(Float4 *)(&(smemC[C_lds_addr]));
            }
        }
    }
}

__global__ void myKernelConv2dGpu128x128x64_256_2(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1, 256)))
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    // int32_t mask;
    int pos_flt[4], pos_act[4];
    int pos_h[4], pos_w[4];
    bool A_ldg_guard[4], B_ldg_guard[4];

    int warp_id = tx / 64;
    int lane_id = tx % 64;

    _Float16 *flt_ptr = param.pweight;
    _Float16 *act_ptr = param.pin;
    BB globalReadA;
    globalReadA.x = (long)(param.pin);
    globalReadA.y = (((((long )0x20000)<<32) | 0x80000000));
    BB globalReadB;
    globalReadB.x = (long)(param.pweight);
    globalReadB.y = (((((long )0x20000)<<32) | 0x80000000));

    int gemm_k = param.c * param.r * param.s;
    int gemm_m = param.k;
    int gemm_n = param.n * param.Oh * param.Ow;

    // ldg reg
    Float4 A_ldg_reg[4], B_ldg_reg[4];
    Float4 acc[4][4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            acc[i][j] = (Float4)(0.0f);
        }
    }
    // smem
    __shared__ char smem[128 * 64 * 2 * sizeof(_Float16)];
    _Float16 *smemA = reinterpret_cast<_Float16 *>(smem);
    _Float16 *smemB = reinterpret_cast<_Float16 *>(smem + 128 * 64 * sizeof(_Float16));

    int A_sts_addr = warp_id * 64 * 32 + (lane_id / 8) * 64 + ((lane_id % 8) ^ (lane_id / 8)) * 8;
    int B_sts_addr = warp_id * 64 * 32 + (lane_id / 8) * 64 + ((lane_id % 8) ^ (lane_id / 8)) * 8;
    // lds reg
    Float2 A_lds_reg[4][2][2], B_lds_reg[4][2][2];

    // lds addr
    int A_lds_addr[2], B_lds_addr[2];
    int lds_row = (lane_id % 16);
    int lds_col0 = ((lane_id / 16) % 8) ^ (lds_row % 8);
    int lds_col1 = (((lane_id / 16) + 4) % 8) ^ (lds_row % 8);
    A_lds_addr[0] = ((warp_id % 2) * 64 * 64) + lds_row * 64 + lds_col0 * 8;
    B_lds_addr[0] = ((warp_id / 2) * 64 * 64) + lds_row * 64 + lds_col0 * 8;
    A_lds_addr[1] = ((warp_id % 2) * 64 * 64) + lds_row * 64 + lds_col1 * 8;
    B_lds_addr[1] = ((warp_id / 2) * 64 * 64) + lds_row * 64 + lds_col1 * 8;

    // ldg
    int gemm_rowA = by * 128 + warp_id * 32 + (lane_id / 8);
    int gemm_rowB = bx * 128 + warp_id * 32 + (lane_id / 8);
    const int gemm_col = (lane_id % 8) * 8;
    Conv2dFpropFilterTileAccessIteratorOptimized iteratorB(param, gemm_rowB, gemm_col, 4);
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int n, pq, p, q;
        param.divmod_pq(n, pq, gemm_rowA);
        param.divmod_q(p, q, pq);
        // output mapping input index
        pos_h[i] = p - param.p;
        pos_w[i] = q - param.q;
        pos_act[i] = (n * param.h * param.w + pos_h[i] * param.w + pos_w[i]) * param.c + gemm_col;
        bool guard = pos_h[i] >= 0 && pos_h[i] < param.h && pos_w[i] >= 0 && pos_w[i] < param.w;
        int offset = guard ? pos_act[i] * 2 : -1;
        BUF_LDG128(*(Float4 *)(&(A_ldg_reg[i])), globalReadA, offset);

        gemm_rowA += 8;
        B_ldg_reg[i] = iteratorB.valid() ? iteratorB.get()[0] : (Float4)(0.0f);
        ++iteratorB;
    }
    // iteratorA.advance();
    iteratorB.advance();
    CG_SYNC(0);
// sts 0
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(Float4 *)(&(smemA[A_sts_addr + i * 8 * 64])) = A_ldg_reg[i];
        *(Float4 *)(&(smemB[B_sts_addr + i * 8 * 64])) = B_ldg_reg[i];
    }
    // read_flag ^= 1;
    __syncthreads();

// lds0
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(Float4 *)(&(A_lds_reg[i][0][0])) = *(Float4 *)(&(smemA[A_lds_addr[0] + i * 16 * 64]));
        *(Float4 *)(&(B_lds_reg[i][0][0])) = *(Float4 *)(&(smemB[B_lds_addr[0] + i * 16 * 64]));
    }

    for (int rtile = 0; rtile < 3; rtile++)
    {
        for (int stile = 0; stile < 3; stile++)
        {
            for (int ctile = 0; ctile < param.c; ctile += 64)
            {
                if (ctile == 0 && rtile == 0 && stile == 0)
                    continue;
                // ldg 1
                int act_offset = (rtile * param.w + stile) * param.c + ctile;
                int flt_offset = (rtile * 3 + stile) * param.c + ctile;
                for (int i = 0; i < 4; i++)
                {
                    int ih = pos_h[i] + rtile;
                    int iw = pos_w[i] + stile;
                    bool guard = ih >= 0 && ih < param.h && iw >= 0 && iw < param.w;
                    int offset = guard ? (pos_act[i] + act_offset) * 2 : -1;
                    BUF_LDG128(*(Float4 *)(&(A_ldg_reg[i])), globalReadA, offset);
                    B_ldg_reg[i] = iteratorB.valid() ? iteratorB.get()[0] : (Float4)(0.0f);
                    ++iteratorB;
                }
                // iteratorA.advance();
                iteratorB.advance();
#pragma unroll
                for (int subk = 0; subk < 2; ++subk)
                {
                    if (subk == 1)
                    {
                        CG_SYNC(0);
                        __syncthreads();
#pragma unroll
                        for (int i = 0; i < 4; i++)
                        {
                            *(Float4 *)(&(smemA[A_sts_addr + i * 8 * 64])) = A_ldg_reg[i];
                            *(Float4 *)(&(smemB[B_sts_addr + i * 8 * 64])) = B_ldg_reg[i];
                        }
                        __syncthreads();
                    }
// lds
#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        *(Float4 *)(&(A_lds_reg[i][(subk + 1) % 2][0])) = *(Float4 *)(&(smemA[A_lds_addr[(subk + 1) % 2] + i * 16 * 64]));
                        *(Float4 *)(&(B_lds_reg[i][(subk + 1) % 2][0])) = *(Float4 *)(&(smemB[B_lds_addr[(subk + 1) % 2] + i * 16 * 64]));
                    }
                    // mma
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            for (int k = 0; k < 2; k++)
                            {
                                HMMA161616(acc[i][j],
                                           A_lds_reg[i][subk][k],
                                           B_lds_reg[j][subk][k]);
                            }
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(Float4 *)(&(A_lds_reg[i][1][0])) = *(Float4 *)(&(smemA[A_lds_addr[1] + i * 16 * 64]));
        *(Float4 *)(&(B_lds_reg[i][1][0])) = *(Float4 *)(&(smemB[B_lds_addr[1] + i * 16 * 64]));
    }

    // mma 0
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                HMMA161616(acc[i][j],
                           A_lds_reg[i][0][k],
                           B_lds_reg[j][0][k]);
            }
        }
    }

    // mma 1
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                HMMA161616(acc[i][j],
                           A_lds_reg[i][1][k],
                           B_lds_reg[j][1][k]);
            }
        }
    }
    _Float16 *smemC = reinterpret_cast<_Float16 *>(smem);
    // nchw
    for (int j = 0; j < 4; j++)
    {
        __syncthreads();
        for (int i = 0; i < 4; i++)
        {
            int C_sts_addr = warp_id * 64 * 16 + (lane_id / 16) * 64 + i * 16 + (lane_id % 16);
            smemC[C_sts_addr] = (_Float16)acc[i][j].x;
            smemC[C_sts_addr + 4 * 64] = (_Float16)acc[i][j].y;
            smemC[C_sts_addr + 8 * 64] = (_Float16)acc[i][j].z;
            smemC[C_sts_addr + 12 * 64] = (_Float16)acc[i][j].w;
        }
        __syncthreads();
        for (int i = 0; i < 2; i++)
        {
            int C_lds_addr = warp_id * 64 * 16 + i * 64 * 8 + (lane_id / 8) * 64 + (lane_id % 8) * 8;
            int k = bx * 128 + (warp_id / 2) * 64 + i * 8 + j * 16 + (lane_id / 8);
            int npq = by * 128 + (warp_id % 2) * 64 + (lane_id % 8) * 8;
            int pq, n;
            param.divmod_pq(n, pq, npq);
            bool guard = pq < param.Oh * param.Ow && k < param.k;
            if (guard)
            {
                int offset = bz * param.n * param.k * param.Oh * param.Ow + n * param.k * param.Oh * param.Ow + k * param.Oh * param.Ow + pq;
                *(Float4 *)(&(param.pout[offset])) = *(Float4 *)(&(smemC[C_lds_addr]));
            }
        }
    }
}

__global__ void myKernelConv2dGpu256x64x64_256(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1, 256)))
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    // int32_t mask;
    int pos_flt[2], pos_act[8];
    int pos_h[8], pos_w[8];
    bool A_ldg_guard[8], B_ldg_guard[2];

    int warp_id = tx / 64;
    int lane_id = tx % 64;

    _Float16 *flt_ptr = param.pweight;
    _Float16 *act_ptr = param.pin;
    BB globalReadA;
    globalReadA.x = (long)(param.pin);
    globalReadA.y = (((((long )0x20000)<<32) | 0x80000000));
    BB globalReadB;
    globalReadB.x = (long)(param.pweight);
    globalReadB.y = (((((long )0x20000)<<32) | 0x80000000));

    int gemm_k = param.c * param.r * param.s;
    int gemm_m = param.k;
    int gemm_n = param.n * param.Oh * param.Ow;

    // ldg reg
    Float4 A_ldg_reg[8], B_ldg_reg[2];
    Float4 acc[4][4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            acc[i][j] = (Float4)(0.0f);
        }
    }
    // smem
    __shared__ char smem[(256 * 64 + 64 * 64) * sizeof(_Float16)];
    _Float16 *smemA = reinterpret_cast<_Float16 *>(smem);
    _Float16 *smemB = reinterpret_cast<_Float16 *>(smem + 256 * 64 * sizeof(_Float16));

    int A_sts_addr = warp_id * 64 * 64 + (lane_id / 8) * 64 + ((lane_id % 8) ^ (lane_id / 8)) * 8;
    int B_sts_addr = warp_id * 64 * 16 + (lane_id / 8) * 64 + ((lane_id % 8) ^ (lane_id / 8)) * 8;
    // lds reg
    Float2 A_lds_reg[4][2][2], B_lds_reg[4][2][2];

    // lds addr
    int A_lds_addr[2], B_lds_addr[2];
    int lds_row = (lane_id % 16);
    int lds_col0 = ((lane_id / 16) % 8) ^ (lds_row % 8);
    int lds_col1 = (((lane_id / 16) + 4) % 8) ^ (lds_row % 8);
    A_lds_addr[0] = (warp_id * 64 * 64) + lds_row * 64 + lds_col0 * 8;
    B_lds_addr[0] = lds_row * 64 + lds_col0 * 8;
    A_lds_addr[1] = (warp_id * 64 * 64) + lds_row * 64 + lds_col1 * 8;
    B_lds_addr[1] = lds_row * 64 + lds_col1 * 8;

    // ldg
    int gemm_rowA = by * 256 + warp_id * 64 + (lane_id / 8);
    int gemm_rowB = bx * 64 + warp_id * 16 + (lane_id / 8);
    const int gemm_col = (lane_id % 8) * 8;
    Conv2dFpropFilterTileAccessIteratorOptimized iteratorB(param, gemm_rowB, gemm_col, 2);
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int n, pq, p, q;
        param.divmod_pq(n, pq, gemm_rowA);
        param.divmod_q(p, q, pq);
        // output mapping input index
        pos_h[i] = p - param.p;
        pos_w[i] = q - param.q;
        pos_act[i] = (n * param.h * param.w + pos_h[i] * param.w + pos_w[i]) * param.c + gemm_col;
        bool guard = pos_h[i] >= 0 && pos_h[i] < param.h && pos_w[i] >= 0 && pos_w[i] < param.w;
        int offset = guard ? pos_act[i] * 2 : -1;
        BUF_LDG128(*(Float4 *)(&(A_ldg_reg[i])), globalReadA, offset);
        gemm_rowA += 8;
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
    {
        B_ldg_reg[i] = iteratorB.valid() ? iteratorB.get()[0] : (Float4)(0.0f);
        ++iteratorB;
    }
    iteratorB.advance();
    CG_SYNC(0);
// sts 0
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        *(Float4 *)(&(smemA[A_sts_addr + i * 8 * 64])) = A_ldg_reg[i];
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
    {
        *(Float4 *)(&(smemB[B_sts_addr + i * 8 * 64])) = B_ldg_reg[i];
    }
    __syncthreads();

// lds0
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(Float4 *)(&(A_lds_reg[i][0][0])) = *(Float4 *)(&(smemA[A_lds_addr[0] + i * 16 * 64]));
        *(Float4 *)(&(B_lds_reg[i][0][0])) = *(Float4 *)(&(smemB[B_lds_addr[0] + i * 16 * 64]));
    }

    for (int rtile = 0; rtile < 3; rtile++)
    {
        for (int stile = 0; stile < 3; stile++)
        {
            for (int ctile = 0; ctile < param.c; ctile += 64)
            {
                if (ctile == 0 && rtile == 0 && stile == 0)
                    continue;
                // ldg 1
                int act_offset = (rtile * param.w + stile) * param.c + ctile;
                int flt_offset = (rtile * 3 + stile) * param.c + ctile;
                for (int i = 0; i < 8; i++)
                {
                    int ih = pos_h[i] + rtile;
                    int iw = pos_w[i] + stile;
                    bool guard = ih >= 0 && ih < param.h && iw >= 0 && iw < param.w;
                    int offset = guard ? (pos_act[i] + act_offset) * 2 : -1;
                    BUF_LDG128(*(Float4 *)(&(A_ldg_reg[i])), globalReadA, offset);
                }
                for (int i = 0; i < 2; i++)
                {
                    B_ldg_reg[i] = iteratorB.valid() ? iteratorB.get()[0] : (Float4)(0.0f);
                    ++iteratorB;
                }
                iteratorB.advance();
#pragma unroll
                for (int subk = 0; subk < 2; ++subk)
                {
                    if (subk == 1)
                    {
                        CG_SYNC(0);
                        __syncthreads();
#pragma unroll
                        for (int i = 0; i < 8; i++)
                        {
                            *(Float4 *)(&(smemA[A_sts_addr + i * 8 * 64])) = A_ldg_reg[i];
                        }
#pragma unroll
                        for (int i = 0; i < 2; i++)
                        {
                            *(Float4 *)(&(smemB[B_sts_addr + i * 8 * 64])) = B_ldg_reg[i];
                        }
                        __syncthreads();
                    }
// lds
#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        *(Float4 *)(&(A_lds_reg[i][(subk + 1) % 2][0])) = *(Float4 *)(&(smemA[A_lds_addr[(subk + 1) % 2] + i * 16 * 64]));
                        *(Float4 *)(&(B_lds_reg[i][(subk + 1) % 2][0])) = *(Float4 *)(&(smemB[B_lds_addr[(subk + 1) % 2] + i * 16 * 64]));
                    }
                    // mma
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            for (int k = 0; k < 2; k++)
                            {
                                HMMA161616(acc[i][j],
                                           A_lds_reg[i][subk][k],
                                           B_lds_reg[j][subk][k]);
                            }
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(Float4 *)(&(A_lds_reg[i][1][0])) = *(Float4 *)(&(smemA[A_lds_addr[1] + i * 16 * 64]));
        *(Float4 *)(&(B_lds_reg[i][1][0])) = *(Float4 *)(&(smemB[B_lds_addr[1] + i * 16 * 64]));
    }

    // mma 0
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                HMMA161616(acc[i][j],
                           A_lds_reg[i][0][k],
                           B_lds_reg[j][0][k]);
            }
        }
    }

    // mma 1
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                HMMA161616(acc[i][j],
                           A_lds_reg[i][1][k],
                           B_lds_reg[j][1][k]);
            }
        }
    }
    _Float16 *smemC = reinterpret_cast<_Float16 *>(smem);
    // nchw
    for (int j = 0; j < 4; j++)
    {
        __syncthreads();
        for (int i = 0; i < 4; i++)
        {
            int C_sts_addr = warp_id * 64 * 16 + (lane_id / 16) * 64 + i * 16 + (lane_id % 16);
            smemC[C_sts_addr] = (_Float16)acc[i][j].x;
            smemC[C_sts_addr + 4 * 64] = (_Float16)acc[i][j].y;
            smemC[C_sts_addr + 8 * 64] = (_Float16)acc[i][j].z;
            smemC[C_sts_addr + 12 * 64] = (_Float16)acc[i][j].w;
        }
        __syncthreads();
        for (int i = 0; i < 2; i++)
        {
            int C_lds_addr = warp_id * 64 * 16 + i * 64 * 8 + (lane_id / 8) * 64 + (lane_id % 8) * 8;
            int k = bx * 64 + i * 8 + j * 16 + (lane_id / 8);
            int npq = by * 256 + warp_id * 64 + (lane_id % 8) * 8;
            int pq, n;
            param.divmod_pq(n, pq, npq);
            bool guard = pq < param.Oh * param.Ow && k < param.k;
            if (guard)
            {
                int offset = n * param.k * param.Oh * param.Ow + k * param.Oh * param.Ow + pq;
                *(Float4 *)(&(param.pout[offset])) = *(Float4 *)(&(smemC[C_lds_addr]));
            }
        }
    }
}

/*选手需要返回自定义kernel入参结构体的size*/
int getParamsize(__in__ problem_t* problem, __out__ int* paramSize)
{
    *paramSize = sizeof(mykernelParamType);

    return 0;
}

/*选手需要返回自己优化的kernel的grid信息与kernel函数的指针*/
int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param)
{
    mykernelParamType* pArgs = (mykernelParamType*)param;

    unsigned int n = problem->n;
    unsigned int c = problem->c;
    unsigned int h = problem->h;
    unsigned int w = problem->w;

    unsigned int k = problem->k;
    unsigned int r = problem->r;
    unsigned int s = problem->s;
    unsigned int u = problem->u;
    unsigned int v = problem->v;
    unsigned int p = problem->p;
    unsigned int q = problem->q;
    FastDivmod divmod_rs = problem->divmod_rs;                       
    FastDivmod divmod_s = problem->divmod_s;                       
    FastDivmod divmod_pq = problem->divmod_pq;                        
    FastDivmod divmod_q = problem->divmod_q;
    FastDivmod divmod_c = problem->divmod_c;

    unsigned int outh = (h - r + 2*p)/u + 1;
    unsigned int outw = (w - s + 2*q)/v + 1;

    kernelInfo->blockz   = 1;                    //blockz  number
    kernelInfo->threadx  = 256;                   //threadx number per block
    kernelInfo->thready  = 1;                   //thready number per block
    kernelInfo->threadz  = 1;                   //threadz number per block
    kernelInfo->dynmicLdsSize = 0; 
    pArgs->splitKNum = 1;
    
    if (c == 128 || c == 320)
    {
        kernelInfo->blockx = (k + 32 - 1) / 32;                          // blockx  number
        kernelInfo->blocky = ((n * outh * outw + 32 - 1) / 32);          // blocky  number
        kernelInfo->kernelPtr = (void *)myKernelConv2dGpu32x32x64_256<>; // kernel ptr
        // kernelInfo->kernelPtr = (void *)myKernelConv2dGpu128x32x64_256<>; // kernel ptr
    }
    else if (c == 256)
    {
        kernelInfo->blockx = (k + 128 - 1) / 128;                        // blockx  number
        kernelInfo->blocky = ((n * outh * outw + 128 - 1) / 128);        // blocky  number
        kernelInfo->kernelPtr = (void *)myKernelConv2dGpu128x128x64_256_1; // kernel ptr
    }
    else if (c == 640)
    {
        kernelInfo->blockx = (k + 128 - 1) / 128;                        // blockx  number
        kernelInfo->blocky = ((n * outh * outw + 128 - 1) / 128);        // blocky  number
        kernelInfo->kernelPtr = (void *)myKernelConv2dGpu128x128x64_256_2; // kernel ptr
    }
    else if (c == 64)
    {
        kernelInfo->blockx = (k + 64 - 1) / 64;                         // blockx  number
        kernelInfo->blocky = ((n * outh * outw + 256 - 1) / 256);       // blocky  number
        kernelInfo->kernelPtr = (void *)myKernelConv2dGpu256x64x64_256; // kernel ptr
    }
    else if (c == 1920)
    {
        kernelInfo->blockx = (k + 128 - 1) / 128;                        // blockx  number
        kernelInfo->blocky = ((n * outh * outw + 128 - 1) / 128);        // blocky  number
        kernelInfo->kernelPtr = (void *)myKernelConv2dGpu128x128x64_256_splitK; // kernel ptr
        kernelInfo->blockz = 3;                                          // blockz  number
        pArgs->splitKNum = 3;
    }

    pArgs->pin = problem->in;
    pArgs->pweight = problem->weight;
    pArgs->pout = problem->out;
    pArgs->n = n;                              //batch szie              default value 1
    pArgs->c = c;                              //channel number          default value 32
    pArgs->h = h;                              //数据高                  default value 32
    pArgs->w = w;                              //数据宽                  default value 32
    pArgs->k = k;                              //卷积核数量              default value 32
    pArgs->r = r;                              //卷积核高                default value 1
    pArgs->s = s;                              //卷积核宽                default value 1
    pArgs->u = u;                              //卷积在高方向上的步长     default value 1
    pArgs->v = v;                              //卷积在宽方向上的步长     default value 1
    pArgs->p = p;                              //卷积在高方向上的补边     default value 0
    pArgs->q = q;                              //卷积在宽方向上的补边     default value 0
    pArgs->Oh = outh;
    pArgs->Ow = outw;     
    pArgs->divmod_rs = divmod_rs;
    pArgs->divmod_s = divmod_s;
    pArgs->divmod_pq = divmod_pq;
    pArgs->divmod_q = divmod_q;
    pArgs->divmod_c = divmod_c;

    return 0;
}