/* ****************************************************************************** *
 * MIT License                                                                    *
 *                                                                                *
 * Copyright (c) 2018 Maxim Masterov                                              *
 *                                                                                *
 * Permission is hereby granted, free of charge, to any person obtaining a copy   *
 * of this software and associated documentation files (the "Software"), to deal  *
 * in the Software without restriction, including without limitation the rights   *
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      *
 * copies of the Software, and to permit persons to whom the Software is          *
 * furnished to do so, subject to the following conditions:                       *
 *                                                                                *
 * The above copyright notice and this permission notice shall be included in all *
 * copies or substantial portions of the Software.                                *
 *                                                                                *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    *
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  *
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  *
 * SOFTWARE.                                                                      *
 * ****************************************************************************** */

#ifndef DOT_H_
#define DOT_H_

/*!
 * \file Dot.h
 * \brief Contains SSE2 optimized vector dot product
 *
 * Contains SSE2 and OpenMP optimized methods for vector dot product, such as
 *  - \f$ \alpha = (x, y) \f$
 */

#include "../Macros.h"
#ifdef BUMBLEBEE_USE_OPENMP
#include <omp.h>
#endif

namespace wrp_mpi {
namespace low {
/*!
 * \ingroup SSE2
 * \brief Vector dot product (low-level)
 *
 * Low level vector dot product for SSE2 intrinsics.
 * Calculates scalar of the form: \f$ \alpha = (x, y) \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param N Size of the vectors
 */
inline double _dot128_d(const double x[], const double y[], const uint32_t N) {
    __m128d sum2 = _mm_set1_pd(0.0);
    uint32_t i = 0;
    for(; i < ROUND_DOWN(N, 2); i += 2) {
        sum2 = _mm_add_pd(sum2, _mm_mul_pd(_mm_load_pd(x + i), _mm_load_pd(y + i)));
    }
    __m128d temp = _mm_hadd_pd(sum2, sum2);
    double sum = _mm_cvtsd_f64(temp);
    for(; i < N; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

/*!
 * \ingroup SSE2
 * \brief Vector dot product (low-level)
 *
 * Low level vector dot product for SSE2 intrinsics.
 * Calculates scalar of the form: \f$ \alpha = (x, y) \f$ and returns long double
 * as a result
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param N Size of the vectors
 */
inline long double _dot128_ld(const double x[], const double y[], const uint32_t N) {
    __m128d sum2 = _mm_set1_pd(0.0);
    uint32_t i = 0;
    for(; i < ROUND_DOWN(N, 2); i += 2) {
        sum2 = _mm_add_pd(sum2, _mm_mul_pd(_mm_load_pd(x + i), _mm_load_pd(y + i)));
    }
    __m128d temp = _mm_hadd_pd(sum2, sum2);
    long double sum = _mm_cvtsd_f64(temp);
    for(; i < N; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}
}

/*!
 * \ingroup SSE2
 * \brief Vector dot product (high-level)
 *
 * Calls for wrp::low::_dot128_d() function in OpenMP loop to calculate
 * scalar of the form: \f$ \alpha = (x, y) \f$
 *
 * @param x Vector
 * @param y Vector
 * @param N Size of the vectors
 * @param nthreads Number of OpenMP threads
 * @return Result of dot product as a scalar
 */
inline double _dot_d(const double x[], const double y[], const uint32_t N) { //, const int offset) {
#ifdef BUMBLEBEE_USE_OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    int offset = ROUND_DOWN(N / nthreads, nthreads);
    if (IS_ODD(offset)) --offset;
    double suma[nthreads]; // = {0.0f};
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for(int i = 0; i < nthreads; ++i) {
        suma[i] = low::_dot128_d(&x[i * offset], &y[i * offset], offset);
    }
    double sum = 0.0f;
    for(int i = 0; i < nthreads; ++i) {
        sum += suma[i];
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

/*!
 * \ingroup SSE2
 * \brief Vector dot product (high-level)
 *
 * Calls for wrp::low::_dot128_d() function in OpenMP loop to calculate
 * scalar of the form: \f$ \alpha = (x, y) \f$ and returns long double
 * as a result
 *
 * @param x Vector
 * @param y Vector
 * @param N Size of the vectors
 * @param nthreads Number of OpenMP threads
 * @return Result of dot product as a scalar
 */
inline long double _dot_ld(const double x[], const double y[], const uint32_t N) { //, const int offset) {
#ifdef BUMBLEBEE_USE_OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    int offset = ROUND_DOWN(N / nthreads, nthreads);
    if (IS_ODD(offset)) --offset;
    long double suma[nthreads]; // = {0.0f};
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for(int i = 0; i < nthreads; ++i) {
        suma[i] = low::_dot128_d(&x[i * offset], &y[i * offset], offset);
    }
    long double sum = 0.0f;
    for(int i = 0; i < nthreads; ++i) {
        sum += suma[i];
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}
}

#endif /* DOT_H_ */
