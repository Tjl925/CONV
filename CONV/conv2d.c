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

#if defined(__aarch64__) && defined(__linux__)
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>

/*
 * v05-C: 16 output rows x 64 output columns, four ZA tiles.
 * Read contiguous input directly: no input transpose or input panel.
 * B is a vertically shifted kernel. For output row r, nonzero terms are
 * visited in t-r (kernel row), ik order, matching the reference recurrence.
 * Use ZA vertical slices on store. Save d8-d15 across streaming transitions.
 */
void conv_sme16x64(const float*, const float*, float*, size_t, size_t, int, int);
__asm__(
".pushsection .text\n"
".arch armv8.2-a+sve\n"
".arch_extension sme\n"
".align 4\n"
".global conv_sme16x64\n"
".hidden conv_sme16x64\n"
".type conv_sme16x64, %function\n"
"conv_sme16x64:\n"
"sub sp, sp, #64\n"
"stp d8, d9, [sp]\n"
"stp d10, d11, [sp, #16]\n"
"stp d12, d13, [sp, #32]\n"
"stp d14, d15, [sp, #48]\n"
"smstart\n"
"ptrue p0.s\n"
"zero {za}\n"
"1:\n"
"mov x9, x0\n"
"mov w7, w6\n"
"tbz w7, #0, 4f\n"
"ld1w {z0.s}, p0/z, [x9]\n"
"ld1w {z1.s}, p0/z, [x9, #1, mul vl]\n"
"ld1w {z2.s}, p0/z, [x9, #2, mul vl]\n"
"ld1w {z3.s}, p0/z, [x9, #3, mul vl]\n"
"ld1w {z4.s}, p0/z, [x1]\n"
"add x9, x9, #4\n"
"add x1, x1, #64\n"
"fmopa za0.s, p0/m, p0/m, z0.s, z4.s\n"
"fmopa za1.s, p0/m, p0/m, z1.s, z4.s\n"
"fmopa za2.s, p0/m, p0/m, z2.s, z4.s\n"
"fmopa za3.s, p0/m, p0/m, z3.s, z4.s\n"
"subs w7, w7, #1\n"
"cbz w7, 5f\n"
"4:\n"
"ld1w {z0.s}, p0/z, [x9]\n"
"ld1w {z1.s}, p0/z, [x9, #1, mul vl]\n"
"ld1w {z2.s}, p0/z, [x9, #2, mul vl]\n"
"ld1w {z3.s}, p0/z, [x9, #3, mul vl]\n"
"ld1w {z4.s}, p0/z, [x1]\n"
"add x9, x9, #4\n"
"add x1, x1, #64\n"
"ld1w {z16.s}, p0/z, [x9]\n"
"ld1w {z17.s}, p0/z, [x9, #1, mul vl]\n"
"ld1w {z18.s}, p0/z, [x9, #2, mul vl]\n"
"ld1w {z19.s}, p0/z, [x9, #3, mul vl]\n"
"ld1w {z20.s}, p0/z, [x1]\n"
"add x9, x9, #4\n"
"add x1, x1, #64\n"
"fmopa za0.s, p0/m, p0/m, z0.s, z4.s\n"
"fmopa za1.s, p0/m, p0/m, z1.s, z4.s\n"
"fmopa za2.s, p0/m, p0/m, z2.s, z4.s\n"
"fmopa za3.s, p0/m, p0/m, z3.s, z4.s\n"
"fmopa za0.s, p0/m, p0/m, z16.s, z20.s\n"
"fmopa za1.s, p0/m, p0/m, z17.s, z20.s\n"
"fmopa za2.s, p0/m, p0/m, z18.s, z20.s\n"
"fmopa za3.s, p0/m, p0/m, z19.s, z20.s\n"
"subs w7, w7, #2\n"
"b.ne 4b\n"
"5:\n"
"add x0, x0, x3\n"
"subs w5, w5, #1\n"
"b.ne 1b\n"
"add x10, x2, #64\n"
"add x11, x10, #64\n"
"add x13, x11, #64\n"
"mov w12, #0\n"
"3:\n"
"mova z0.s, p0/m, za0v.s[w12, 0]\n"
"mova z1.s, p0/m, za1v.s[w12, 0]\n"
"mova z2.s, p0/m, za2v.s[w12, 0]\n"
"mova z3.s, p0/m, za3v.s[w12, 0]\n"
"st1w {z0.s}, p0, [x2]\n"
"st1w {z1.s}, p0, [x10]\n"
"st1w {z2.s}, p0, [x11]\n"
"st1w {z3.s}, p0, [x13]\n"
"add x2, x2, x4\n"
"add x10, x10, x4\n"
"add x11, x11, x4\n"
"add x13, x13, x4\n"
"add w12, w12, #1\n"
"cmp w12, #16\n"
"b.ne 3b\n"
"smstop\n"
"ldp d8, d9, [sp]\n"
"ldp d10, d11, [sp, #16]\n"
"ldp d12, d13, [sp, #32]\n"
"ldp d14, d15, [sp, #48]\n"
"add sp, sp, #64\n"
"ret\n"
".size conv_sme16x64, .-conv_sme16x64\n"
".arch armv8-a\n"
".popsection\n");

/* Ordered SVE rectangle, also used for the narrow right edge. */
__attribute__((target("arch=armv8.2-a+sve")))
static void conv_rectangle(const float* input, int W, const float* kernel,
    int KH, int KW, float* output, int OW, int j, int rows, int i, int cols)
{
    const int lanes=(int)svcntw();
    for (int r=0;r<rows;++r)
        for (int c=0;c<cols;c+=lanes) {
            svbool_t pg=svwhilelt_b32(c,cols);
            svfloat32_t a=svdup_n_f32(0);
            for(int jk=0;jk<KH;++jk)
                for(int ik=0;ik<KW;++ik)
                    a=svmla_f32_m(pg,a,svld1_f32(pg,input+(size_t)(j+r+jk)*W+i+c+ik),
                                 svdup_n_f32(kernel[(size_t)jk*KW+ik]));
            svst1_f32(pg,output+(size_t)(j+r)*OW+i+c,a);
        }
}

static int conv_sme(const float* input, int H, int W, const float* kernel, int KH, int KW, float* output)
{
    const int OH=H-KH+1, OW=W-KW+1;
    if (OH<16 || OW<64 || KH<1 || KW<1 || KH>256 || KW>256 ||
        !(getauxval(AT_HWCAP2)&(1UL<<23)) || (prctl(64,0,0,0,0)&65535)!=64)
        return 0;
    const int T=KH+15;
    float* weights=(float*)aligned_alloc(64,(size_t)T*KW*16*sizeof(float));
    if (!weights) return 0;
    memset(weights,0,(size_t)T*KW*16*sizeof(float));
    /* ZA rows are output columns; ZA columns are output rows.
       For output row r, t-r visits kernel rows in increasing order. */
    for (int t=0;t<T;++t)
        for (int ik=0;ik<KW;++ik)
            for (int r=0;r<16;++r)
                if(t>=r && t-r<KH)
                    weights[((size_t)t*KW+ik)*16+r]=kernel[(size_t)(t-r)*KW+ik];
    const int fullRows=(OH/16)*16, fullCols=(OW/64)*64;
#pragma omp parallel
    {
        const int usable=(prctl(64,0,0,0,0)&65535)==64;
#pragma omp for collapse(2) schedule(static)
        for(int j=0;j<fullRows;j+=16)
            for(int i=0;i<fullCols;i+=64) {
                if(usable)
                    conv_sme16x64(input+(size_t)j*W+i,weights,
                        output+(size_t)j*OW+i,(size_t)W*4,(size_t)OW*4,T,KW);
                else
                    conv_rectangle(input,W,kernel,KH,KW,output,OW,j,16,i,64);
            }
#pragma omp for schedule(static)
        for(int j=0;j<OH;++j) {
            if(j>=fullRows)
                conv_rectangle(input,W,kernel,KH,KW,output,OW,j,1,0,OW);
            else if(fullCols<OW)
                conv_rectangle(input,W,kernel,KH,KW,output,OW,j,1,fullCols,OW-fullCols);
        }
    }
    free(weights);
    return 1;
}

#endif

void conv2d(const float* input, int H, int W, const float* kernel, int KH, int KW, float* output)
{
#if defined(__aarch64__) && defined(__linux__)
    if (conv_sme(input,H,W,kernel,KH,KW,output)) return;
    if (getauxval(AT_HWCAP) & (1UL<<22)) {
        conv_sve(input,H,W,kernel,KH,KW,output);
        return;
    }
#endif
    conv_neon(input,H,W,kernel,KH,KW,output);
}
