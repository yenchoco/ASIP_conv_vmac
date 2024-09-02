/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <string.h>
#include <limits.h>
// #include <arm_nnfunctions.h>
// #include <unity.h>
#include "arm_nn_types.h"
#include "arm_nn_compiler.h"
#include "basic/test_data.h"
// #include "../TestData/conv_2/test_data.h"
// #include "../TestData/conv_2x2_dilation/test_data.h"
// #include "../TestData/conv_2x2_dilation_5x5_input/test_data.h"
// #include "../TestData/conv_2x3_dilation/test_data.h"
// #include "../TestData/conv_3/test_data.h"
// #include "../TestData/conv_3x2_dilation/test_data.h"
// #include "../TestData/conv_3x3_dilation_5x5_input/test_data.h"
// #include "../TestData/conv_4/test_data.h"
// #include "../TestData/conv_5/test_data.h"
// #include "../TestData/conv_dilation_golden/test_data.h"
// #include "../TestData/conv_out_activation/test_data.h"
// #include "../TestData/stride2pad1/test_data.h"
#include "validate.h"

#define USE_FAST_DW_CONV_S16_FUNCTION(dw_conv_params, filter_dims, input_dims)                                         \
    (dw_conv_params->ch_mult == 1 && dw_conv_params->dilation.w == 1 && dw_conv_params->dilation.h == 1 &&             \
     filter_dims->w * filter_dims->h < 512)

#define LEFT_SHIFT(_shift) (_shift > 0 ? _shift : 0)
#define RIGHT_SHIFT(_shift) (_shift > 0 ? 0 : -_shift)
#define MASK_IF_ZERO(x) (x) == 0 ? ~0 : 0
#define MASK_IF_NON_ZERO(x) (x) != 0 ? ~0 : 0
#define SELECT_USING_MASK(mask, a, b) ((mask) & (a)) ^ (~(mask) & (b))

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define CLAMP(x, h, l) MAX(MIN((x), (h)), (l))
#define REDUCE_MULTIPLIER(_mult) ((_mult < 0x7FFF0000) ? ((_mult + (1 << 15)) >> 16) : 0x7FFF)

struct arm_nn_double
{
    uint32_t low;
    int32_t high;
};

union arm_nn_long_long
{
    int64_t long_long;
    struct arm_nn_double word;
};

int32_t arm_nn_doubling_high_mult_no_sat(const int32_t m1, const int32_t m2)
{
    int32_t result = 0;
    union arm_nn_long_long mult;

    // Rounding offset to add for a right shift of 31
    mult.word.low = 1 << 30;
    mult.word.high = 0;

    // Gets resolved as a SMLAL instruction
    mult.long_long = mult.long_long + (int64_t)m1 * m2;

    // Utilize all of the upper 32 bits. This is the doubling step
    // as well.
    result = (int32_t)(mult.long_long >> 31);

    return result;
}

int32_t arm_nn_divide_by_power_of_two(const int32_t dividend, const int32_t exponent)
{
    int32_t result = 0;
    const int32_t remainder_mask = (1 << exponent) - 1;
    int32_t remainder = remainder_mask & dividend;

    // Basic division
    result = dividend >> exponent;

    // Adjust 'result' for rounding (mid point away from zero)
    int32_t threshold = remainder_mask >> 1;
    if (result < 0)
    {
        threshold++;
    }
    if (remainder > threshold)
    {
        result++;
    }

    return result;
}

int32_t arm_nn_requantize(const int32_t val, const int32_t multiplier, const int32_t shift)
{
#ifdef CMSIS_NN_USE_SINGLE_ROUNDING
    const int64_t total_shift = 31 - shift;
    const int64_t new_val = val * (int64_t)multiplier;

    int32_t result = new_val >> (total_shift - 1);
    result = (result + 1) >> 1;

    return result;
#else
    return arm_nn_divide_by_power_of_two(arm_nn_doubling_high_mult_no_sat(val * (1 << LEFT_SHIFT(shift)), multiplier),
                                         RIGHT_SHIFT(shift));
#endif
}

void arm_memset_s8(int8_t *dst, const int8_t val, uint32_t block_size)
{
#if defined(ARM_MATH_MVEI)
    __asm volatile("   vdup.8                  q0, %[set_val]             \n"
                   "   wlstp.8                 lr, %[cnt], 1f             \n"
                   "2:                                                    \n"
                   "   vstrb.8                 q0, [%[in]], #16            \n"
                   "   letp                    lr, 2b                     \n"
                   "1:                                                    \n"
                   : [in] "+r"(dst)
                   : [cnt] "r"(block_size), [set_val] "r"(val)
                   : "q0", "memory", "r14");
#else
    memset(dst, val, block_size);
#endif
}

void arm_memcpy_s8(int8_t * dst, const int8_t * src, uint32_t block_size)
{
#if defined(ARM_MATH_MVEI)
    __asm volatile("   wlstp.8                 lr, %[cnt], 1f             \n"
                   "2:                                                    \n"
                   "   vldrb.8                 q0, [%[in]], #16            \n"
                   "   vstrb.8                 q0, [%[out]], #16           \n"
                   "   letp                    lr, 2b                     \n"
                   "1:                                                    \n"
                   : [in] "+r"(src), [out] "+r"(dst)
                   : [cnt] "r"(block_size)
                   : "q0", "memory", "r14");
#else
    memcpy(dst, src, block_size);
#endif
}

void arm_q7_to_q15_with_offset(const int8_t *src, int16_t *dst, int32_t block_size, int16_t offset)
{
    int32_t block_cnt;

#if defined(ARM_MATH_MVEI)

    int16x8_t source;
    const int16x8_t source_offset = vdupq_n_s16(offset);
    block_cnt = block_size / 8;

    while (block_cnt > 0)
    {
        source = vldrbq_s16(src);
        source = vaddq_s16(source, source_offset);
        vstrhq_s16(dst, source);
        dst += 8;
        src += 8;
        block_cnt--;
    }

    block_cnt = block_size & 0x7;

#elif defined(ARM_MATH_DSP)
    /* Run the below code for cores that support SIMD instructions  */
    int32_t in_q7x4;
    int32_t in_q15x2_1;
    int32_t in_q15x2_2;
    int32_t out_q15x2_1;
    int32_t out_q15x2_2;

    /*loop unrolling */
    block_cnt = block_size >> 2;

    /* First part of the processing with loop unrolling.  Compute 4 outputs at a time. */
    const int32_t offset_q15x2 = PKHBT(offset, offset, 16);
    while (block_cnt > 0)
    {
        /* convert from s8 to s16 and then store the results in the destination buffer */
        in_q7x4 = arm_nn_read_s8x4_ia(&src);

        /* Extract and sign extend each of the four s8 values to s16 */
        in_q15x2_1 = SXTAB16(offset_q15x2, ROR(in_q7x4, 8));
        in_q15x2_2 = SXTAB16(offset_q15x2, in_q7x4);

        out_q15x2_2 = PKHTB(in_q15x2_1, in_q15x2_2, 16);
        out_q15x2_1 = PKHBT(in_q15x2_2, in_q15x2_1, 16);

        arm_nn_write_q15x2_ia(&dst, out_q15x2_1);
        arm_nn_write_q15x2_ia(&dst, out_q15x2_2);

        block_cnt--;
    }
    /* Handle left over samples */
    block_cnt = block_size % 0x4;

#else
    /* Run the below code for Cortex-M0 */
    /* Loop over block_size number of values */
    block_cnt = block_size;
#endif

    while (block_cnt > 0)
    {
        *dst++ = (int16_t)*src++ + offset;

        /* Decrement the loop counter */
        block_cnt--;
    }
}

void arm_q7_to_q31_with_offset(const int8_t *src, int32_t *dst, int32_t block_size, int32_t offset)
{
    int32_t block_cnt;

#if defined(ARM_MATH_MVEI)

    int16x8_t source;
    const int16x8_t source_offset = vdupq_n_s16(offset);
    block_cnt = block_size / 8;

    while (block_cnt > 0)
    {
        source = vldrbq_s16(src);
        source = vaddq_s16(source, source_offset);
        vstrhq_s16(dst, source);
        dst += 8;
        src += 8;
        block_cnt--;
    }

    block_cnt = block_size & 0x7;

#elif defined(ARM_MATH_DSP)
    /* Run the below code for cores that support SIMD instructions  */
    int32_t in_q7x4;
    int32_t in_q15x2_1;
    int32_t in_q15x2_2;
    int32_t out_q15x2_1;
    int32_t out_q15x2_2;

    /*loop unrolling */
    block_cnt = block_size >> 2;

    /* First part of the processing with loop unrolling.  Compute 4 outputs at a time. */
    const int32_t offset_q15x2 = PKHBT(offset, offset, 16);
    while (block_cnt > 0)
    {
        /* convert from s8 to s16 and then store the results in the destination buffer */
        in_q7x4 = arm_nn_read_s8x4_ia(&src);

        /* Extract and sign extend each of the four s8 values to s16 */
        in_q15x2_1 = SXTAB16(offset_q15x2, ROR(in_q7x4, 8));
        in_q15x2_2 = SXTAB16(offset_q15x2, in_q7x4);

        out_q15x2_2 = PKHTB(in_q15x2_1, in_q15x2_2, 16);
        out_q15x2_1 = PKHBT(in_q15x2_2, in_q15x2_1, 16);

        arm_nn_write_q15x2_ia(&dst, out_q15x2_1);
        arm_nn_write_q15x2_ia(&dst, out_q15x2_2);

        block_cnt--;
    }
    /* Handle left over samples */
    block_cnt = block_size % 0x4;

#else
    /* Run the below code for Cortex-M0 */
    /* Loop over block_size number of values */
    block_cnt = block_size;
#endif

    while (block_cnt > 0)
    {
        *dst++ = (int32_t)*src++ + offset;

        /* Decrement the loop counter */
        block_cnt--;
    }
}

void matmul_leftover_fused_pipe(v4int_t *ip_a0,
                                    v4int_t *ip_b0,
                                    int32_t* ch_0_out_ptr,
                                    int32_t col_count,
                                    int32_t aligned_num_col_a
                                    ){
    // while (col_count)
        // {
        //     int8_t a0 = *ip_a0++;
        //     int16_t b0 = *ip_b0++;
        //     int16_t b1 = *ip_b1++;

        //     ch_0_out_0 += a0 * b0;
        //     ch_0_out_1 += a0 * b1;
        //     col_count--;
        // }
        printf("address a: %d\n",ip_a0);
        printf("address b: %d\n",ip_b0);
        alignas(32) v4int_t* ali_a = ip_a0;
        alignas(32) v4int_t* ali_b = ip_b0;
        printf("address align_a: %d\n",ali_a);
        printf("address align_b: %d\n",ali_b);
        
        v4int_t* vsum = (v4int_t*)ch_0_out_ptr;
        *vsum = vbc(0);
        for(int i=0; i < col_count/4;i++){   //i<2
            v4int_t aa0 = chess_protect(*ip_a0);
            for(int j=0;j<4;j++){
                printf("%d\n",i);
                *vsum = vmac(*vsum, vbc(vext(aa0,j)), *(ip_b0+j*aligned_num_col_a));
            }
        }
        // for (int ko = 0; ko < 8; ko++) {
        //     v4int_t aa0 = chess_protect(av[i][ko]);
        //     for (int ki=0; ki<4; ki++) {
        //     vsum = vmac(vsum, vbc(vext(aa0,ki)), bv[4*ko+ki][j]);
        //     }
        // }
        // c[i][j] = vsum;
  }

int8_t *arm_nn_mat_mult_kernel_s8_s16(const int32_t *input_a,    //filter array
                                      const int32_t *input_b,   //input array
                                      const uint16_t output_ch,
                                      const int32_t *out_shift,
                                      const int32_t *out_mult,
                                      const int32_t out_offset,
                                      const int16_t activation_min,
                                      const int16_t activation_max,
                                      const int32_t num_col_a,
                                      const int32_t aligned_num_col_a,
                                      const int32_t *const output_bias,
                                      int8_t *out_0)
{
#if !defined(ARM_MATH_MVEI)
    /* set up the second output pointers */
    int8_t *out_1 = out_0 + output_ch;  //&out_1 = &out_0 +1
    int8_t *out_2 = out_1 + output_ch;
    int8_t *out_3 = out_2 + output_ch;
    alignas(32) const int32_t *bias = output_bias;  //basic_biases[1] = {6388};

    uint16_t row_count = output_ch / 2; //0
    const int32_t *ip_a0 = input_a;
    /* this loop over rows in A */
    while (row_count)
    {
        /* setup pointers for B */
        const int32_t *ip_b0 = input_b;
        const int32_t *ip_b1 = ip_b0 + aligned_num_col_a;   //ip_b0 + rhs_col
        // const int16_t *ip_b2 = ip_b1 + aligned_num_col_a;   //ip_b0 + rhs_col
        // const int16_t *ip_b3 = ip_b2 + aligned_num_col_a;   //ip_b0 + rhs_col

        /* align the second pointer for A */
        const int32_t *ip_a1 = ip_a0 + num_col_a;    //ip_a0 + rhs_col

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;
        // int32_t ch_0_out_2 = 0;
        // int32_t ch_0_out_3 = 0;
        int32_t ch_1_out_0 = 0;
        int32_t ch_1_out_1 = 0;
        /* Init accumulator with bias for channel N and N + 1 */
        if (bias)
        {
            ch_0_out_0 = *bias;     //ch_0_out_0 = *bias
            ch_0_out_1 = *bias++;   //ch_0_out_1 = *bias, move to next bias
            ch_1_out_0 = *bias;
            ch_1_out_1 = *bias++;
        }

    #if defined(ARM_MATH_DSP)
        int32_t col_count = num_col_a / 4;
        /* accumulate over the vector */
        while (col_count)
        {
            int32_t a01, a02, a11, a12;
            int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad_reordered(ip_a0, &a01, &a02);
            ip_a1 = read_and_pad_reordered(ip_a1, &a11, &a12);

            ch_0_out_0 = SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a01, b1, ch_0_out_1);
            ch_1_out_0 = SMLAD(a11, b0, ch_1_out_0);
            ch_1_out_1 = SMLAD(a11, b1, ch_1_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = SMLAD(a12, b0, ch_1_out_0);
            ch_1_out_1 = SMLAD(a12, b1, ch_1_out_1);

            col_count--;
        } /* while over col_count */
        col_count = num_col_a & 0x3;
    #else
        int32_t col_count = num_col_a;
    #endif

        while (col_count)
        {
            int8_t a0 = *ip_a0++;
            int16_t b0 = *ip_b0++;
            int8_t a1 = *ip_a1++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            ch_1_out_0 += a1 * b0;
            ch_1_out_1 += a1 * b1;
            col_count--;
        } /* while over col_count */

        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (int8_t)ch_0_out_0;

        ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (int8_t)ch_0_out_1;
        out_mult++;
        out_shift++;

        ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
        ch_1_out_0 += out_offset;
        ch_1_out_0 = MAX(ch_1_out_0, activation_min);
        ch_1_out_0 = MIN(ch_1_out_0, activation_max);
        *out_0++ = (int8_t)ch_1_out_0;

        ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
        ch_1_out_1 += out_offset;
        ch_1_out_1 = MAX(ch_1_out_1, activation_min);
        ch_1_out_1 = MIN(ch_1_out_1, activation_max);
        *out_1++ = (int8_t)ch_1_out_1;
        out_mult++;
        out_shift++;

        /* skip row */
        ip_a0 += num_col_a;
        row_count--;
    }

    /* compute the last odd numbered row if any */
    if (output_ch & 0x1)
    {
        /* setup pointers for B */
        alignas(32) const int32_t *ip_b0 = input_b;
        alignas(32) const int32_t *ip_b1 = ip_b0 + aligned_num_col_a;
        // const int16_t *ip_b2 = ip_b1 + aligned_num_col_a;
        // const int16_t *ip_b3 = ip_b2 + aligned_num_col_a;

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;
        int32_t ch_0_out_2 = 0;
        int32_t ch_0_out_3 = 0;

        /* load the bias */
        if (bias)
        {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;
            ch_0_out_2 = *bias++;
            ch_0_out_3 = *bias++;
        }

    #if defined(ARM_MATH_DSP)
        int32_t col_count = num_col_a >> 2;
        while (col_count)
        {
            int32_t a01, a02;
            int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad_reordered(ip_a0, &a01, &a02);

            ch_0_out_0 = SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a01, b1, ch_0_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);
            ch_0_out_0 = SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a02, b1, ch_0_out_1);

            col_count--;
        }
        col_count = num_col_a & 0x3;
    #else
        int32_t col_count = num_col_a;  //rhs_col
    #endif
        int32_t* ch_0_out_ptr;
        matmul_leftover_fused_pipe((v4int_t*)ip_a0, (v4int_t*)ip_b0, ch_0_out_ptr, col_count, aligned_num_col_a);
        // while (col_count)
        // {
        //     int8_t a0 = *ip_a0++;
        //     int16_t b0 = *ip_b0++;
        //     int16_t b1 = *ip_b1++;

        //     ch_0_out_0 += a0 * b0;
        //     ch_0_out_1 += a0 * b1;
        //     col_count--;
        // }
        ch_0_out_0 = arm_nn_requantize(*ch_0_out_ptr++, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (int8_t)ch_0_out_0;

        ch_0_out_1 = arm_nn_requantize(*ch_0_out_ptr++, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (int8_t)ch_0_out_1;

        ch_0_out_2 = arm_nn_requantize(*ch_0_out_ptr++, *out_mult, *out_shift);
        ch_0_out_2 += out_offset;
        ch_0_out_2 = MAX(ch_0_out_2, activation_min);
        ch_0_out_2 = MIN(ch_0_out_2, activation_max);
        *out_2++ = (int8_t)ch_0_out_2;

        ch_0_out_0 = arm_nn_requantize(*ch_0_out_ptr, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_3++ = (int8_t)ch_0_out_0;
        // ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        // ch_0_out_0 += out_offset;
        // ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        // ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        // *out_0++ = (int8_t)ch_0_out_0;

        // ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        // ch_0_out_1 += out_offset;
        // ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        // ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        // *out_1++ = (int8_t)ch_0_out_1;

        out_mult++;
        out_shift++;
    }

    out_0 += output_ch;

    /* return the new output pointer with offset */
    return out_0;
#else
    (void)input_a;
    (void)input_b;
    (void)output_ch;
    (void)out_shift;
    (void)out_mult;
    (void)out_offset;
    (void)activation_min;
    (void)activation_max;
    (void)aligned_num_col_a, (void)num_col_a;
    (void)output_bias;
    (void)out_0;
    /* To be completed */
    return NULL;
#endif
}


int8_t *arm_nn_mat_mult_kernel_row_offset_s8_s16(const int8_t *input_a,
                                                 const int16_t *input_b,
                                                 const uint16_t output_ch,
                                                 const int32_t *out_shift,
                                                 const int32_t *out_mult,
                                                 const int32_t out_offset,
                                                 const int16_t activation_min,
                                                 const int16_t activation_max,
                                                 const int32_t num_col_a,
                                                 const int32_t aligned_num_col_a,
                                                 const int32_t *const output_bias,
                                                 const int32_t row_address_offset,
                                                 int8_t *out_0)
{

#if !defined(ARM_MATH_MVEI)
    /* set up the second output pointers */

    int8_t *out_1 = out_0 + row_address_offset;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch / 2;
    const int8_t *ip_a0 = input_a;
    /* this loop over rows in A */
    while (row_count)
    {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + aligned_num_col_a;

        /* align the second pointer for A */
        const int8_t *ip_a1 = ip_a0 + num_col_a;

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;
        int32_t ch_1_out_0 = 0;
        int32_t ch_1_out_1 = 0;
        /* Init accumulator with bias for channel N and N + 1 */
        if (bias)
        {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;
            ch_1_out_0 = *bias;
            ch_1_out_1 = *bias++;
        }

    #if defined(ARM_MATH_DSP)
        int32_t col_count = num_col_a / 4;
        /* accumulate over the vector */
        while (col_count)
        {
            int32_t a01, a02, a11, a12;
            int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad_reordered(ip_a0, &a01, &a02);
            ip_a1 = read_and_pad_reordered(ip_a1, &a11, &a12);

            ch_0_out_0 = SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a01, b1, ch_0_out_1);
            ch_1_out_0 = SMLAD(a11, b0, ch_1_out_0);
            ch_1_out_1 = SMLAD(a11, b1, ch_1_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = SMLAD(a12, b0, ch_1_out_0);
            ch_1_out_1 = SMLAD(a12, b1, ch_1_out_1);

            col_count--;
        } /* while over col_count */

        col_count = num_col_a & 0x3;

    #else
        int32_t col_count = num_col_a;
    #endif
        while (col_count)
        {
            int8_t a0 = *ip_a0++;
            int16_t b0 = *ip_b0++;
            int8_t a1 = *ip_a1++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            ch_1_out_0 += a1 * b0;
            ch_1_out_1 += a1 * b1;
            col_count--;
        } /* while over col_count */

        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (int8_t)ch_0_out_0;

        ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (int8_t)ch_0_out_1;
        out_mult++;
        out_shift++;

        ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
        ch_1_out_0 += out_offset;
        ch_1_out_0 = MAX(ch_1_out_0, activation_min);
        ch_1_out_0 = MIN(ch_1_out_0, activation_max);
        *out_0++ = (int8_t)ch_1_out_0;

        ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
        ch_1_out_1 += out_offset;
        ch_1_out_1 = MAX(ch_1_out_1, activation_min);
        ch_1_out_1 = MIN(ch_1_out_1, activation_max);
        *out_1++ = (int8_t)ch_1_out_1;
        out_mult++;
        out_shift++;

        /* skip row */
        ip_a0 += num_col_a;
        row_count--;
    }

    /* compute the last odd numbered row if any */
    if (output_ch & 0x1)
    {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + aligned_num_col_a;

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;

        /* load the bias */
        if (bias)
        {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;
        }

    #if defined(ARM_MATH_DSP)
        int32_t col_count = num_col_a >> 2;
        while (col_count)
        {
            int32_t a01, a02;
            int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad_reordered(ip_a0, &a01, &a02);

            ch_0_out_0 = SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a01, b1, ch_0_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);
            ch_0_out_0 = SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a02, b1, ch_0_out_1);

            col_count--;
        }
        col_count = num_col_a & 0x3;

    #else
        int32_t col_count = num_col_a;
    #endif
        while (col_count)
        {
            int8_t a0 = *ip_a0++;
            int16_t b0 = *ip_b0++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            col_count--;
        }

        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (int8_t)ch_0_out_0;

        ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (int8_t)ch_0_out_1;
        out_mult++;
        out_shift++;
    }

    out_0 += 2 * row_address_offset - output_ch;

    /* return the new output pointer with offset */
    return out_0;
#else
    (void)input_a;
    (void)input_b;
    (void)output_ch;
    (void)out_shift;
    (void)out_mult;
    (void)out_offset;
    (void)activation_min;
    (void)activation_max;
    (void)aligned_num_col_a, (void)num_col_a;
    (void)output_bias;
    (void)row_address_offset;
    (void)out_0;
    return NULL;
#endif
}
/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_s8.c
 * Description:  s8 version of convolution using symmetric quantization.
 *
 * $Date:        27 February 2024
 * $Revision:    V.3.7.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */
// #include "arm_nnsupportfunctions.h"
/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Basic s8 convolution function.
 *
 * Refer header file for details. Optimal use case for the DSP/MVE implementation is when input and output channels
 * are multiples of 4 or atleast greater than 4.
 *
 */
arm_cmsis_nn_status arm_convolve_s8(const cmsis_nn_context *ctx,
                                    const cmsis_nn_conv_params *conv_params,
                                    const cmsis_nn_per_channel_quant_params *quant_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int8_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const int8_t *filter_data,
                                    const cmsis_nn_dims *bias_dims,
                                    const int32_t *bias_data,
                                    const cmsis_nn_dims *output_dims,
                                    int8_t *output_data)
{
    (void)bias_dims;

    if (ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int32_t *buffer_a = (int32_t *)ctx->buf;    //ptr to buffer
    

    const int32_t input_batches = input_dims->n;    //1
    const uint16_t input_x = input_dims->w;
    const uint16_t input_y = input_dims->h;
    const uint16_t input_ch = input_dims->c;    //1
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;
    const uint16_t kernel_ch = filter_dims->c;  //1
    const uint16_t output_x = output_dims->w;   //4
    const uint16_t output_y = output_dims->h;   //5
    const uint16_t output_ch = output_dims->c;  //1

    const uint16_t pad_x = conv_params->padding.w;
    const uint16_t pad_y = conv_params->padding.h;
    const uint16_t stride_x = conv_params->stride.w;
    const uint16_t stride_y = conv_params->stride.h;
    const int32_t dilation_x = conv_params->dilation.w; //1
    const int32_t dilation_y = conv_params->dilation.h; //1
    const int32_t out_offset = conv_params->output_offset;
    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;
    const int32_t input_offset = conv_params->input_offset;

    const int32_t groups = input_ch / kernel_ch;    //1
    const int32_t rhs_cols = kernel_x * kernel_y * kernel_ch;   //unroll the filter =>8
    const int32_t output_ch_per_group = output_ch / groups; //1

    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;

    if (input_ch % groups != 0 || output_ch % groups != 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t remainder = rhs_cols % 4;
    const int32_t aligned_rhs_cols = remainder != 0 ? rhs_cols + 4 - remainder : rhs_cols;

    for (int i_batch = 0; i_batch < input_batches; i_batch++)
    {

#if defined(ARM_MATH_MVEI)
        const int32_t aligned_rhs_cols_offset = aligned_rhs_cols - rhs_cols;

        /* Generate up to four columns from the input tensor a GEMM computation */
        int8_t *im2col_buf = (int8_t *)buffer_a;
#else
        /* Use as a ping-pong buffer for unordered elements */
        int8_t *im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 4;
        // int16_t *im2col_buf_start_s16 = buffer_a;
        int32_t *im2col_buf_start_s32 = buffer_a;
        int32_t *buffer_filt = (int32_t*)malloc(rhs_cols*sizeof(int32_t));
        int16_t *temp_change = (int16_t*)malloc((2 * aligned_rhs_cols) * (int32_t)sizeof(int16_t));
        
#endif
        int32_t lhs_rows = 0;

        const int8_t *filter_data_ptr = &filter_data[0];
        const int32_t *bias_data_ptr = &bias_data[0];
        const int32_t *output_mult_ptr = &output_mult[0];
        const int32_t *output_shift_ptr = &output_shift[0];

        /* This part implements the im2col function */
        for (int32_t i_group = 0; i_group < groups; i_group++)
        {
            int8_t *out = output_data + i_group * output_ch_per_group;  //output_data
            for (int i_out_y = 0; i_out_y < output_y; i_out_y++)    //5
            {
                for (int i_out_x = 0; i_out_x < output_x; i_out_x++)    //4
                {
                    const int32_t base_idx_x = stride_x * i_out_x - pad_x;  //i_out_x
                    const int32_t base_idx_y = stride_y * i_out_y - pad_y;  //i_out_y

                    for (int32_t i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)    //unroll input data
                    {
                        for (int32_t i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                        {
                            const int32_t k_y = base_idx_y + dilation_y * i_ker_y;  //i_out_y + i_ker_y
                            const int32_t k_x = base_idx_x + dilation_x * i_ker_x;

                            if (k_y < 0 || k_y >= input_y || k_x < 0 || k_x >= input_x)
                            {
                                arm_memset_s8(im2col_buf, (int8_t)-input_offset, sizeof(int8_t) * kernel_ch);
                            }
                            else
                            {
                                arm_memcpy_s8(im2col_buf,
                                              input_data + (k_y * input_x + k_x) * input_ch + i_group * kernel_ch,
                                              sizeof(int8_t) * kernel_ch);
                            }
                            im2col_buf += kernel_ch;    //ptr location += kernel_ch
                        }
                    }
                    lhs_rows++;

#if defined(ARM_MATH_MVEI)
                    im2col_buf += aligned_rhs_cols_offset;

                    /* Computation is filed for every 4 columns */
                    if (lhs_rows == 4)
                    {
                        arm_nn_mat_mult_nt_t_s8((int8_t *)buffer_a,
                                                filter_data_ptr,
                                                bias_data_ptr,
                                                out,
                                                output_mult_ptr,
                                                output_shift_ptr,
                                                lhs_rows,
                                                output_ch_per_group,
                                                rhs_cols,
                                                input_offset,
                                                out_offset,
                                                out_activation_min,
                                                out_activation_max,
                                                output_ch,
                                                aligned_rhs_cols);

                        out += lhs_rows * output_ch;

                        lhs_rows = 0;
                        im2col_buf = (int8_t *)buffer_a;
                    }
#else
    #if defined(ARM_MATH_DSP)
                    /* Copy one column with input offset and no ordering */
                    arm_s8_to_s16_unordered_with_offset(
                        im2col_buf - rhs_cols, im2col_buf_start_s16, rhs_cols, (int16_t)input_offset);
    #else
                    // int32_t intput_offset_32 = 
                    // arm_q7_to_q15_with_offset(
                    //     im2col_buf - rhs_cols, im2col_buf_start_s16, rhs_cols, (int16_t)input_offset);  //1st para=> start ptr of transformed im2col array
                        //put a filter size of the im2col input data to buffer_a
                    arm_q7_to_q31_with_offset(
                        im2col_buf - rhs_cols, im2col_buf_start_s32, rhs_cols, input_offset);
                    arm_q7_to_q31_with_offset(
                        filter_data_ptr, buffer_filt, rhs_cols, input_offset);

    #endif
                    im2col_buf_start_s32 += aligned_rhs_cols;   //move the ptr for the next iteration to put

                    if (lhs_rows == 4)
                    {
                        if (groups > 1)
                        {
                            /*out = arm_nn_mat_mult_kernel_row_offset_s8_s16(filter_data_ptr,
                                                                           buffer_a,
                                                                           output_ch_per_group,
                                                                           output_shift_ptr,
                                                                           output_mult_ptr,
                                                                           out_offset,
                                                                           out_activation_min,
                                                                           out_activation_max,
                                                                           rhs_cols,
                                                                           aligned_rhs_cols,
                                                                           bias_data_ptr,
                                                                           output_ch,
                                                                           out);*/
                              out = arm_nn_mat_mult_kernel_row_offset_s8_s16(filter_data_ptr,
                                                                           temp_change,
                                                                           output_ch_per_group,
                                                                           output_shift_ptr,
                                                                           output_mult_ptr,
                                                                           out_offset,
                                                                           out_activation_min,
                                                                           out_activation_max,
                                                                           rhs_cols,
                                                                           aligned_rhs_cols,
                                                                           bias_data_ptr,
                                                                           output_ch,
                                                                           out);
                        }
                        else    //this one
                        {
                            // out = arm_nn_mat_mult_kernel_s8_s16(filter_data_ptr,    //filter array 1st element
                            //                                     buffer_a,           //intput array
                            //                                     output_ch_per_group,
                            //                                     output_shift_ptr,
                            //                                     output_mult_ptr,
                            //                                     out_offset,
                            //                                     out_activation_min,
                            //                                     out_activation_max,
                            //                                     rhs_cols,
                            //                                     aligned_rhs_cols,
                            //                                     bias_data_ptr,      //basic_biases[1] = {6388};
                            //                                     out);
                            out = arm_nn_mat_mult_kernel_s8_s16(buffer_filt,    //filter array 1st element
                                                                buffer_a,           //intput array
                                                                output_ch_per_group,
                                                                output_shift_ptr,
                                                                output_mult_ptr,
                                                                out_offset,
                                                                out_activation_min,
                                                                out_activation_max,
                                                                rhs_cols,
                                                                aligned_rhs_cols,
                                                                bias_data_ptr,      //basic_biases[1] = {6388};
                                                                out);
                        }

                        /* counter reset */
                        im2col_buf_start_s32 = buffer_a;
                        im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 4;
                        lhs_rows = 0;
                    }
#endif
                }
            }

            if (out == NULL)
            {
                return ARM_CMSIS_NN_NO_IMPL_ERROR;
            }

            /* Handle left over columns */  //若總數為奇數會有剩餘的，處理剩餘的
            if (lhs_rows != 0)
            {
#if defined(ARM_MATH_MVEI)
                arm_nn_mat_mult_nt_t_s8((int8_t *)buffer_a,
                                        filter_data_ptr,
                                        bias_data_ptr,
                                        out,
                                        output_mult_ptr,
                                        output_shift_ptr,
                                        lhs_rows,
                                        output_ch_per_group,
                                        rhs_cols,
                                        input_offset,
                                        out_offset,
                                        out_activation_min,
                                        out_activation_max,
                                        output_ch,
                                        aligned_rhs_cols);

                out += lhs_rows * output_ch;
                lhs_rows = 0;
                im2col_buf = (int8_t *)buffer_a;
#else // #if defined(ARM_MATH_MVEI)

                const int8_t *ker_a = filter_data_ptr;
                int i;

                for (i = 0; i < output_ch_per_group; i++)
                {
                    /* Load the accumulator with bias first */
                    int32_t sum = 0;
                    if (bias_data_ptr)
                    {
                        sum = bias_data_ptr[i];
                    }

                    const int32_t *ip_as_col = buffer_a;

    #if defined(ARM_MATH_DSP)
                    /* 4 multiply and accumulates are done in one loop. */
                    uint16_t col_count = rhs_cols / 4;
                    while (col_count)
                    {
                        int32_t ker_a1, ker_a2;
                        int32_t ip_b1, ip_b2;

                        ker_a = read_and_pad_reordered(ker_a, &ker_a1, &ker_a2);

                        ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
                        sum = SMLAD(ker_a1, ip_b1, sum);
                        ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
                        sum = SMLAD(ker_a2, ip_b2, sum);

                        col_count--;
                    }
                    /* Handle left over mac */
                    col_count = rhs_cols & 0x3;
    #else
                    uint16_t col_count = rhs_cols;

    #endif
                    while (col_count)
                    {
                        int8_t ker_a1 = *ker_a++;   //filter array data =>a[i][k]
                        int16_t ip_b1 = *ip_as_col++;   //input array data=> b[k][j]

                        sum += ker_a1 * ip_b1;
                        col_count--;
                    }
                    

                    sum = arm_nn_requantize(sum, output_mult_ptr[i], output_shift_ptr[i]);
                    sum += out_offset;
                    sum = MAX(sum, out_activation_min);
                    sum = MIN(sum, out_activation_max);
                    *out++ = (int8_t)sum;   //c[i][j] = sum
                }

                im2col_buf_start_s32 = buffer_a;
                im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 2;
                lhs_rows = 0;
#endif // #if defined(ARM_MATH_MVEI)
            }
            filter_data_ptr += output_ch_per_group * rhs_cols;
            bias_data_ptr += output_ch_per_group;
            output_mult_ptr += output_ch_per_group;
            output_shift_ptr += output_ch_per_group;
        }
        /* Advance to the next batch */
        input_data += (input_x * input_y * input_ch);
        output_data += (output_x * output_y * output_ch);
    }

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */

int32_t arm_convolve_s8_get_buffer_size_32(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_convolve_s8_get_buffer_size_mve(input_dims, filter_dims);
#else
    const int32_t rhs_cols = filter_dims->w * filter_dims->h * input_dims->c;
    const int32_t remainder = rhs_cols % 4;
    const int32_t aligned_rhs_cols = remainder != 0 ? rhs_cols + 4 - remainder : rhs_cols;
    return (2 * aligned_rhs_cols) * (int32_t)sizeof(int32_t);
#endif
}

int32_t arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_convolve_s8_get_buffer_size_mve(input_dims, filter_dims);
#else
    const int32_t rhs_cols = filter_dims->w * filter_dims->h * input_dims->c;
    const int32_t remainder = rhs_cols % 4;
    const int32_t aligned_rhs_cols = remainder != 0 ? rhs_cols + 4 - remainder : rhs_cols;
    return (2 * aligned_rhs_cols) * (int32_t)sizeof(int16_t);
#endif
}


int main()
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[BASIC_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = basic_biases;
    const int8_t *kernel_data = basic_weights;
    const int8_t *input_data = basic_input;
    const int8_t *output_ref = basic_output_ref;
    const int32_t output_ref_size = BASIC_DST_SIZE;

    input_dims.n = BASIC_INPUT_BATCHES;
    input_dims.w = BASIC_INPUT_W;
    input_dims.h = BASIC_INPUT_H;
    input_dims.c = BASIC_IN_CH;
    filter_dims.w = BASIC_FILTER_X;
    filter_dims.h = BASIC_FILTER_Y;
    filter_dims.c = BASIC_IN_CH;
    output_dims.w = BASIC_OUTPUT_W;
    output_dims.h = BASIC_OUTPUT_H;
    output_dims.c = BASIC_OUT_CH;

    conv_params.padding.w = BASIC_PAD_X;
    conv_params.padding.h = BASIC_PAD_Y;
    conv_params.stride.w = BASIC_STRIDE_X;
    conv_params.stride.h = BASIC_STRIDE_Y;
    conv_params.dilation.w = BASIC_DILATION_X;
    conv_params.dilation.h = BASIC_DILATION_Y;

    conv_params.input_offset = BASIC_INPUT_OFFSET;
    conv_params.output_offset = BASIC_OUTPUT_OFFSET;
    conv_params.activation.min = BASIC_OUT_ACTIVATION_MIN;
    conv_params.activation.max = BASIC_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)basic_output_mult;
    quant_params.shift = (int32_t *)basic_output_shift;

    unsigned int cycle1, cycle2;
    cycle1 = chess_cycle_count();

    int32_t buf_size = arm_convolve_s8_get_buffer_size_32(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    cycle2 = chess_cycle_count();
    printf("  cycle count : %d\n",cycle2-cycle1);

    printf("output array : ");
    for(int i=0;i<BASIC_DST_SIZE;i++){
        printf("%d ", output[i]);
    }

    printf("output ref array : ");
    for(int i=0;i<BASIC_DST_SIZE;i++){
        printf("%d ", output_ref[i]);
    }
    // TEST_ASSERT_EQUAL(expected, result);
    // TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    // memset(output, 0, sizeof(output));

    // buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    // ctx.buf = malloc(buf_size);
    // ctx.size = 0;

    // result = arm_convolve_wrapper_s8(&ctx,
    //                                  &conv_params,
    //                                  &quant_params,
    //                                  &input_dims,
    //                                  input_data,
    //                                  &filter_dims,
    //                                  kernel_data,
    //                                  &bias_dims,
    //                                  bias_data,
    //                                  &output_dims,
    //                                  output);

    // if (ctx.buf)
    // {
    //     memset(ctx.buf, 0, buf_size);
    //     free(ctx.buf);
    // }
    // TEST_ASSERT_EQUAL(expected, result);
    // TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));

    return 0;
}

