#include <iostream>
#include "softposit_cpp.h"
#include <cstring>
#include <iomanip>

#define alpha(i, j)    x[(i) * ld_x + j]
#define beta(i, j)     y[(i) * ld_y + j]
#define gamma(i, j)    z[(i) * ld_z + j]

#define min(x, y)    ((x) < (y) ? (x) : (y))

int MC = 480, KC = 328, NC = 4032;

/* Register micro-tiles */
int MR = 6;
int NR = 16;

typedef uint32_t real32;
typedef void (*gemm_pack_kernel)(int k, int xn, int stride, real32 *x,
    int ld_x, real32 *x_tilde, bool transpose);

/* ======== GEMM MICROKERNELS ======== */

template <typename T>
void gemm_ukernel_6x16(int k, T *x_tilde,
    T *y_tilde, T *z, int ld_z)
{
    for(int p = 0; p < k; p++) {
        for(int i = 0; i < MR; i++) {
            for(int j = 0; j < NR; j++) {
                gamma(i, j) += x_tilde[i] * y_tilde[j];
            }
        }

        x_tilde += MR;
        y_tilde += NR;
    }
}

/* ======== GEMM MICROKERNELS END ======== */

/* ======== GEMM PACKING KERNEL ======== */

void gemm_pack_kernel_float32(int k, int xn, int stride, real32 *xr,
    int ld_x, real32 *xr_tilde, bool transpose)
{
    float *x = (float *) xr;
    float *xt = (float *) xr_tilde;

    for(int ir = 0; ir < xn; ir += stride) {
        int irb = min(stride, xn - ir);
        for(int p = 0; p < k; p++) {
            int i = 0;
            for(; i < irb; i++)
                *xt++ = transpose ? alpha(ir + i, p) : alpha(p, ir + i);

            /* Fill remainders with 0. */
            for(; i < stride; i++)
                *xt++ = 0.0f;
        }
    }
}


void gemm_pack_kernel_posit32(int k, int xn, int stride, real32 *xr,
    int ld_x, real32 *xr_tilde, bool transpose)
{
    posit32 *x = (posit32 *) xr;
    posit32 *xt = (posit32 *) xr_tilde;

    for(int ir = 0; ir < xn; ir += stride) {
        int irb = min(stride, xn - ir);
        for(int p = 0; p < k; p++) {
            int i = 0;
            for(; i < irb; i++)
                *xt++ = transpose ? alpha(ir + i, p) : alpha(p, ir + i);

            /* Fill remainders with 0. */
            for(; i < stride; i++)
                *xt++ = 0.0f;
        }
    }
}


/* ======== GEMM PACKING KERNEL END ======== */

template <typename T>
inline void gemm_loop1(
    int m, int n, int k,
    T *x_tilde,
    T *y_tilde,
    T *z,
    int ld_z
) {
    /* 1st outer loop to slice the block MCxKC to MRxKC micro-panel. */
    for(int i = 0; i < m; i += MR) {
        int ib = min(MR, m - i);

        if(ib == MR && n == NR) {
            /* Dispatch to microkernel. */
            gemm_ukernel_6x16(k, &x_tilde[i * k], y_tilde, &gamma(i, 0), ld_z);
        } else {
            T dest_shell[MR][NR] = { 0 };

            gemm_ukernel_6x16(k, &x_tilde[i * k], y_tilde, (T *) dest_shell, NR);

            /* Copy the shell values to original `z`. */
            for(int ir = 0; ir < ib; ir++) {
                for(int jr = 0; jr < n; jr++) {
                    gamma(i + ir, jr) = dest_shell[ir][jr];
                }
            }
        }
    }
}

template <typename T>
__attribute__((flatten)) inline void gemm_loop2(
    int m, int n, int k,
    T *x_tilde,
    T *y_tilde,
    T *z,
    int ld_z
) {
    /* 2nd outer loop to slice the panel KCxNC to KCxNR micro-panel. */
    for(int j = 0; j < n; j += NR) {
        int jb = min(NR, n - j);
        /* Dispatch to final outer loop. */
        gemm_loop1(m, jb, k, x_tilde, &y_tilde[k * j], &gamma(0, j), ld_z);
    }
}


template <typename T>
__attribute__((flatten)) inline void gemm_loop3(
    bool transp_x,
    int m, int n, int k,
    T *x,
    int ld_x,
    T *y_tilde,
    T *z,
    int ld_z,
    gemm_pack_kernel pack_kernel,
    T *x_tilde
) {
    /* 3rd outer loop to block over the dimension `m`, affected on both `x` and `z`. */
    for(int i = 0; i < m; i += MC) {
        int ib = min(MC, m - i);
        pack_kernel(k, ib, MR, reinterpret_cast<real32*>(&alpha(i, 0)), ld_x,
            reinterpret_cast<real32*>(x_tilde), !transp_x);
        /* Dispatch to 2nd outer loop. */
        gemm_loop2(ib, n, k, x_tilde, y_tilde, &gamma(i, 0), ld_z);
    }
}


template <typename T>
__attribute__((flatten)) inline void gemm_loop4(
    bool transp_x, bool transp_y,
    int m, int n, int k,
    T *x, int ld_x,
    T *y, int ld_y,
    T *z, int ld_z,
    gemm_pack_kernel pack_kernel,
    T *x_tilde,
    T *y_tilde
) {
    /* 4th outer loop to slice over the dimension `k`. */
    for(int p = 0; p < k; p += KC) {
        int pb = min(KC, k - p);
        /* Pack the panel KCxNC. */
        pack_kernel(pb, n, NR, reinterpret_cast<real32*>(&beta(p, 0)), ld_y,
            reinterpret_cast<real32*>(y_tilde), transp_y);
        /* Dispatch to outer loop 3. */
        gemm_loop3(transp_x, m, n, pb, &alpha(0, p), ld_x, y_tilde, z,
            ld_z, pack_kernel, x_tilde);
    }
}


template <typename T>
__attribute__((flatten)) inline void gemm_loop5(
    bool transp_x, bool transp_y,
    int m, int n, int k,
    T *x, int ld_x,
    T *y, int ld_y,
    T *z, int ld_z,
    gemm_pack_kernel pack_kernel
) {
    /* Memory allocation for `MCxKC` block of `x` and `KCxNC` panel of `y`. */
    T *x_tilde = new T[MC * KC];
    T *y_tilde = new T[KC * NC];

    /* It is a shame that we cannot use SIMD here due to lack of SIMD support 
     * for posits. Otherwise this operation would have been handled in the
     * microkernel. */
    memset(z, 0, m * n * sizeof(T));

    /* 5th outer loop to slice `y` over NC. */
    for(int j = 0; j < n; j += NC) {
        int jb = min(NC, n - j);
        /* Dispatch to 4th outer loop. */
        gemm_loop4(transp_x, transp_y, m, jb, k, x, ld_x, &beta(0, j), ld_y,
            &gamma(0, j), ld_z, pack_kernel, x_tilde, y_tilde);
    }

    delete[] x_tilde;
    delete[] y_tilde;
}


void mygemm_float32(
    bool transp_x, bool transp_y,
    int m, int n, int k,
    float *x, int ld_x,
    float *y, int ld_y,
    float *z, int ld_z
) {
    gemm_loop5<float>(transp_x, transp_y, m, n, k, x, ld_x, y, ld_y, z, ld_z,
        gemm_pack_kernel_float32);
}


void mygemm_posit32(
    bool transp_x, bool transp_y,
    int m, int n, int k,
    posit32 *x, int ld_x,
    posit32 *y, int ld_y,
    posit32 *z, int ld_z
) {
    gemm_loop5<posit32>(transp_x, transp_y, m, n, k, x, ld_x, y, ld_y, z, ld_z,
        gemm_pack_kernel_posit32);
}


float randf(float low, float high) {
    return ((float) rand() / RAND_MAX) * (high - low) + low;
}


bool mat_equal(float *mat_a, posit32 *mat_b, size_t size) {
    for(int i = 0; i < size; i++) {
        if(mat_b[i].toDouble() - (double) mat_a[i] > 1e-5) {
            std::cerr << std::fixed << std::setprecision(6) << "Value match failed: [float32]: " <<
                mat_a[i] << " != [posit32]: " << mat_b[i].toDouble() << std::endl;
            std::cerr << "This might be due to 32-bit float numerical instability" << std::endl;
            return false;
        }
    }

    return true;
}


int main() {
    // dimensions to test our GEMM against.
    int ntest = 6;
    int test_shapes[ntest] = { 33, 12, 17, 64, 78, 512 };

    std::cout << "Testing GEMM, float32 against posit32:" << std::endl;
    bool transpose = 0;
    for(int i = 0; i < ntest; i++) {
        int N = test_shapes[i];

        float *zf = new float[N * N];
        float *xf = new float[N * N];
        float *yf = new float[N * N];

        posit32 *zp = new posit32[N * N];
        posit32 *xp = new posit32[N * N];
        posit32 *yp = new posit32[N * N];


        // Initialize
        for(int i = 0; i < N * N; i++) {
            xf[i] = randf(0.0, 1.0f);
            yf[i] = randf(0.0, 1.0f);

            xp[i] = static_cast<double>(xf[i]);
            yp[i] = static_cast<double>(yf[i]);
        }

        mygemm_float32(transpose, transpose, N, N, N, xf, N, yf, N, zf, N);
        mygemm_posit32(transpose, transpose, N, N, N, xp, N, yp, N, zp, N);
        transpose ^= 1; // Next time transpose

        std::cout << "dimension: " << N << "x" <<
            N << std::endl;
        std::cout << (mat_equal(zf, zp, N * N) ? "Success :D" : "Failure :(") << std::endl;

        delete[] zf;
        delete[] xf;
        delete[] yf;

        delete[] zp;
        delete[] xp;
        delete[] yp;
    }
}
