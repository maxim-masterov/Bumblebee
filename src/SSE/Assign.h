/*
 * Assign.h
 *
 *  Created on: Jul 19, 2016
 *      Author: maxim
 */

#ifndef ASSIGN_H_
#define ASSIGN_H_

/*!
 * \file Assign.h
 * \brief Contains SSE2 optimized scalar assignment and vector copy
 *
 * Contains SSE2 and OpenMP optimized methods for coefficient-wise vector
 * copy and assignment to scalar, such as
 *  - \f$ x = 0 \f$
 *  - \f$ x = \alpha \f$
 *  - \f$ x = y \f$
 */

#include "../Macros.h"
#ifdef BUMBLEBEE_USE_OPENMP
#include <omp.h>
#endif

namespace wrp {
namespace low {
/*!
 * \ingroup SSE2
 * \brief Vector assignment to zero (low-level) (64-bit doubles)
 *
 * Low level vector assignment to zero for SSE2 intrinsics.
 * Assign all elements of the given vector to 0.
 *
 * @param x Vector to be updated
 * @param N Size of the vectors
 */
inline void _zero128_d(double x[], const uint32_t N) {
    uint32_t i = 0;
    const __m128d c = _mm_setzero_pd();
    for(; i < ROUND_DOWN(N, 2); i += 2) {
        _mm_store_pd(x + i, c);
    }
    for(; i < N; i++) {
        x[i] = 0.0;
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector assignment to zero (low-level) (32-bit integers)
 *
 * Low level vector assignment to zero for SSE2 intrinsics.
 * Assign all elements of the given vector to 0.
 *
 * @param x Vector to be updated
 * @param N Size of the vectors
 */
inline void _zero128_i(int x[], const uint32_t N) {
    uint32_t i = 0;
    const __m128i c = _mm_setzero_si128();
    for(; i < ROUND_DOWN(N, 4); i += 4) {
        _mm_store_si128((__m128i *)(x + i), c);
    }
    for(; i < N; i++) {
        x[i] = 0;
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector assignment to scalar (low-level)
 *
 * Low level vector assignment to a scalar for SSE2 intrinsics.
 * Assign all elements of the given vector to the given scalar.
 *
 * @param x Vector to be updated
 * @param value Scalar
 * @param N Size of the vectors
 */
inline void _assign128_d(double x[], double const value, const uint32_t N) {
    uint32_t i = 0;
    const __m128d c = _mm_set1_pd(value);
    for(; i < ROUND_DOWN(N, 2); i += 2) {
        _mm_store_pd(x + i, c);
    }
    for(; i < N; i++) {
        x[i] = value;
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector copy (low-level)
 *
 * Low level vector copy for SSE2 intrinsics.
 * Assign one vector to another in coefficient-wise manner.
 *
 * @param x Vector to be updated
 * @param y Vector to be copied
 * @param N Size of the vectors
 */
inline void _setto128_d(double x[], const double y[], const uint32_t N) {
    uint32_t i = 0;
    for(; i < ROUND_DOWN(N, 2); i += 2) {
        __m128d b2 = _mm_load_pd(y + i);
        _mm_store_pd(x + i, b2);
    }
    for(; i < N; i++) {
        x[i] = y[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Scaled vector copy (low-level)
 *
 * Low level scaled vector copy for SSE2 intrinsics.
 * Assign one vector to another in coefficient-wise manner.
 *
 * @param x Vector to be updated
 * @param y Vector to be copied
 * @param a Scale factor
 * @param N Size of the vectors
 */
inline void _setto128_d(double x[], const double y[], const double a, const uint32_t N) {
    uint32_t i = 0;
    __m128d factor = _mm_set1_pd(a);
    for(; i < ROUND_DOWN(N, 2); i += 2) {
        __m128d b2 = _mm_load_pd(y + i);
        _mm_store_pd(x + i, _mm_mul_pd(b2, factor));
    }
    for(; i < N; i++) {
        x[i] = a * y[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Scales vector with a scalar (low-level)
 *
 * Low level scaled vector for SSE2 intrinsics.
 * Assign one vector to another in coefficient-wise manner.
 *
 * @param x Vector to be updated
 * @param a Scale factor
 * @param N Size of the vectors
 */
inline void _scale128_d(double x[], const double a, const uint32_t N) {
    uint32_t i = 0;
    __m128d factor = _mm_set1_pd(a);
    for(; i < ROUND_DOWN(N, 2); i += 2) {
        __m128d b2 = _mm_load_pd(x + i);
        _mm_store_pd(x + i, _mm_mul_pd(b2, factor));
    }
    for(; i < N; i++) {
        x[i] *= a;
    }
}
}

/*!
 * \ingroup SSE2
 * \brief Vector assignment to zero (high-level) (64-bit doubles)
 *
 * Calls for wrp::low::_zero128_d() function in OpenMP loop to assign
 * all vector's coefficients to zero.
 *
 * @param x Vector to be updated
 * @param N Size of the vectors
 */
inline void _zero_d(double x[], const uint32_t N) {
#ifdef BUMBLEBEE_USE_OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    int offset = ROUND_DOWN(N / nthreads, nthreads);
    if (IS_ODD(offset)) --offset;
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for(int i = 0; i < nthreads; ++i) {
        low::_zero128_d(&x[i * offset], offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] = 0.0;
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector assignment to zero (high-level) (32-bit integers)
 *
 * Calls for wrp::low::_zero128_d() function in OpenMP loop to assign
 * all vector's coefficients to zero.
 *
 * @param x Vector to be updated
 * @param N Size of the vectors
 */
inline void _zero_i(int x[], const uint32_t N) {
#ifdef BUMBLEBEE_USE_OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    int offset = ROUND_DOWN(N / nthreads, nthreads);
    if (IS_ODD(offset)) --offset;
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for(int i = 0; i < nthreads; ++i) {
        low::_zero128_i(&x[i * offset], offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] = 0;
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector assignment to scalar (high-level)
 *
 * Calls for wrp::low::_assign128_d() function in OpenMP loop to assign
 * all vector's coefficients to the given scalar.
 *
 * @param x Vector to be updated
 * @param value Scalar
 * @param N Size of the vectors
 */
inline void _assign_d(double x[], double const value, const uint32_t N) {
#ifdef BUMBLEBEE_USE_OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    int offset = ROUND_DOWN(N / nthreads, nthreads);
    if (IS_ODD(offset)) --offset;
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for(int i = 0; i < nthreads; ++i) {
        low::_assign128_d(&x[i * offset], value, offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] = value;
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector copy (high-level)
 *
 * Calls for wrp::low::_setto128_d() function in OpenMP loop to copy
 * coefficients of one vector to another
 *
 * @param x Vector to be updated
 * @param y Vector to be copied
 * @param N Size of the vectors
 */
inline void _setto_d(double x[], const double y[], const uint32_t N) {
#ifdef BUMBLEBEE_USE_OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    int offset = ROUND_DOWN(N / nthreads, nthreads);
    if (IS_ODD(offset)) --offset;
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for(int i = 0; i < nthreads; ++i) {
        low::_setto128_d(&x[i * offset], &y[i * offset], offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] = y[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Scaled vector copy (high-level)
 *
 * Calls for wrp::low::_setto128_d() function in OpenMP loop to copy
 * scaled coefficients of one vector to another
 *
 * @param x Vector to be updated
 * @param y Vector to be copied
 * @param a Scale factor
 * @param N Size of the vectors
 */
inline void _setto_d(double x[], const double y[], const double a, const uint32_t N) {
#ifdef BUMBLEBEE_USE_OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    int offset = ROUND_DOWN(N / nthreads, nthreads);
    if (IS_ODD(offset)) --offset;
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for(int i = 0; i < nthreads; ++i) {
        low::_setto128_d(&x[i * offset], &y[i * offset], a, offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] = a * y[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Scales vector with provided scalar
 *
 * Calls for wrp::low::_scale128_d() function in OpenMP loop to scale
 * coefficients of the vector
 *
 * @param x Vector to be updated
 * @param a Scale factor
 * @param N Size of the vectors
 */
inline void _scale_d(double x[], const double a, const uint32_t N) {
#ifdef BUMBLEBEE_USE_OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    int offset = ROUND_DOWN(N / nthreads, nthreads);
    if (IS_ODD(offset)) --offset;
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for(int i = 0; i < nthreads; ++i) {
        low::_scale128_d(&x[i * offset], a, offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] *= a;
    }
}
}

#endif /* ASSIGN_H_ */
