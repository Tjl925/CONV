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
static void conv_neon(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
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

#if defined(__aarch64__) && defined(__linux__)
#include <arm_sve.h>
#include <sys/auxv.h>

/* v03: vector length is queried per thread; each lane owns one output. */
__attribute__((target("arch=armv8.2-a+sve")))
static void conv_sve(const float* input, int H, int W,
                     const float* kernel, int KH, int KW, float* output)
{
    const int OH=H-KH+1, OW=W-KW+1;
#pragma omp parallel for schedule(static)
    for (int j=0; j<OH; ++j) {
        const int lanes=(int)svcntw();
        const svbool_t all=svptrue_b32();
        float* out=output+(size_t)j*OW;
        int i=0;
        for (; i<=OW-8*lanes; i+=8*lanes) {
            svfloat32_t a0=svdup_n_f32(0.0f);
            svfloat32_t a1=svdup_n_f32(0.0f);
            svfloat32_t a2=svdup_n_f32(0.0f);
            svfloat32_t a3=svdup_n_f32(0.0f);
            svfloat32_t a4=svdup_n_f32(0.0f);
            svfloat32_t a5=svdup_n_f32(0.0f);
            svfloat32_t a6=svdup_n_f32(0.0f);
            svfloat32_t a7=svdup_n_f32(0.0f);
            for (int jk=0; jk<KH; ++jk) {
                const float* p=input+(size_t)(j+jk)*W+i;
                const float* k=kernel+(size_t)jk*KW;
                for (int ik=0; ik<KW; ++ik) {
                    const svfloat32_t weight=svdup_n_f32(k[ik]);
                    a0=svmla_f32_m(all,a0,svld1_f32(all,p+ik+0*lanes),weight);
                    a1=svmla_f32_m(all,a1,svld1_f32(all,p+ik+1*lanes),weight);
                    a2=svmla_f32_m(all,a2,svld1_f32(all,p+ik+2*lanes),weight);
                    a3=svmla_f32_m(all,a3,svld1_f32(all,p+ik+3*lanes),weight);
                    a4=svmla_f32_m(all,a4,svld1_f32(all,p+ik+4*lanes),weight);
                    a5=svmla_f32_m(all,a5,svld1_f32(all,p+ik+5*lanes),weight);
                    a6=svmla_f32_m(all,a6,svld1_f32(all,p+ik+6*lanes),weight);
                    a7=svmla_f32_m(all,a7,svld1_f32(all,p+ik+7*lanes),weight);
                }
            }
            svst1_f32(all,out+i+0*lanes,a0);
            svst1_f32(all,out+i+1*lanes,a1);
            svst1_f32(all,out+i+2*lanes,a2);
            svst1_f32(all,out+i+3*lanes,a3);
            svst1_f32(all,out+i+4*lanes,a4);
            svst1_f32(all,out+i+5*lanes,a5);
            svst1_f32(all,out+i+6*lanes,a6);
            svst1_f32(all,out+i+7*lanes,a7);
        }
        for (; i<OW; i+=lanes) {
            const svbool_t pg=svwhilelt_b32(i,OW);
            svfloat32_t a=svdup_n_f32(0.0f);
            for (int jk=0; jk<KH; ++jk) {
                const float* p=input+(size_t)(j+jk)*W+i;
                const float* k=kernel+(size_t)jk*KW;
                for (int ik=0; ik<KW; ++ik)
                    a=svmla_f32_m(pg,a,svld1_f32(pg,p+ik),svdup_n_f32(k[ik]));
            }
            svst1_f32(pg,out+i,a);
        }
    }
}
#endif
void conv2d(const float* input, int H, int W, const float* kernel, int KH, int KW, float* output)
{
#if defined(__aarch64__) && defined(__linux__)
    if (getauxval(AT_HWCAP) & (1UL<<22)) {
        conv_sve(input,H,W,kernel,KH,KW,output);
        return;
    }
#endif
    conv_neon(input,H,W,kernel,KH,KW,output);
}
