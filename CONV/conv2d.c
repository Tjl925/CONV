#include <stddef.h>
#include <math.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

typedef float CONVFLOAT;
typedef int CONVINT;

/*
 * vectorize across output columns, never across one output's reduction.
 * Input, kernel and output are separate buffers in the official benchmark.
 * Keep jk/ik order and use fused multiply-add to match the ARM reference.
 */
void conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth,
            CONVFLOAT* output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;

#pragma omp parallel for
    for (CONVINT j = 0; j < outputHeight; ++j) {
        CONVFLOAT* out = output + (size_t)j * outputWidth;
        CONVINT i = 0;
#if defined(__aarch64__)
        for (; i <= outputWidth - 32; i += 32) {
            float32x4_t a0 = vdupq_n_f32(0.0f);
            float32x4_t a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f);
            float32x4_t a3 = vdupq_n_f32(0.0f);
            float32x4_t a4 = vdupq_n_f32(0.0f);
            float32x4_t a5 = vdupq_n_f32(0.0f);
            float32x4_t a6 = vdupq_n_f32(0.0f);
            float32x4_t a7 = vdupq_n_f32(0.0f);
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* p = input + (size_t)(j + jk) * inputWidth + i;
                const CONVFLOAT* k = kernel + (size_t)jk * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const CONVFLOAT w = k[ik];
                    const CONVFLOAT* x = p + ik;
                    a0 = vfmaq_n_f32(a0, vld1q_f32(x),      w);
                    a1 = vfmaq_n_f32(a1, vld1q_f32(x + 4),  w);
                    a2 = vfmaq_n_f32(a2, vld1q_f32(x + 8),  w);
                    a3 = vfmaq_n_f32(a3, vld1q_f32(x + 12), w);
                    a4 = vfmaq_n_f32(a4, vld1q_f32(x + 16), w);
                    a5 = vfmaq_n_f32(a5, vld1q_f32(x + 20), w);
                    a6 = vfmaq_n_f32(a6, vld1q_f32(x + 24), w);
                    a7 = vfmaq_n_f32(a7, vld1q_f32(x + 28), w);
                }
            }
            vst1q_f32(out + i,      a0);
            vst1q_f32(out + i + 4,  a1);
            vst1q_f32(out + i + 8,  a2);
            vst1q_f32(out + i + 12, a3);
            vst1q_f32(out + i + 16, a4);
            vst1q_f32(out + i + 20, a5);
            vst1q_f32(out + i + 24, a6);
            vst1q_f32(out + i + 28, a7);
        }
        /* Handle any remaining complete groups of four without over-reading. */
        for (; i <= outputWidth - 4; i += 4) {
            float32x4_t a = vdupq_n_f32(0.0f);
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* p = input + (size_t)(j + jk) * inputWidth + i;
                const CONVFLOAT* k = kernel + (size_t)jk * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                    a = vfmaq_n_f32(a, vld1q_f32(p + ik), k[ik]);
            }
            vst1q_f32(out + i, a);
        }
#endif
        for (; i < outputWidth; ++i) {
            CONVFLOAT sum = 0.0f;
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* p = input + (size_t)(j + jk) * inputWidth + i;
                const CONVFLOAT* k = kernel + (size_t)jk * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                    sum = fmaf(p[ik], k[ik], sum);
            }
            out[i] = sum;
        }
    }
}
