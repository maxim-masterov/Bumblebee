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

#ifndef ADD_H_
#define ADD_H_

/*!
 * \file Add.h
 * \brief Contains SSE2 optimized vector addition
 *
 * Contains SSE2 and OpenMP optimized methods for vector sum, such as
 *  - \f$ x = x + y \f$
 */

#include "../Macros.h"
#ifdef BUMBLEBEE_USE_OPENMP
#include <omp.h>
#endif

namespace wrp_mpi {
namespace low {
/*!
 * \ingroup SSE2
 * \brief Vector sum (low-level)
 *
 * Low level vector sum for SSE2 intrinsics.
 * Calculates vector of the form: \f$ x = x + y \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param N Size of the vectors
 */
inline void _add128_d(double x[], const double y[], const uint32_t N) {
    uint32_t i = 0;
    for(; i < ROUND_DOWN(N, 2); i += 2) {
        __m128d x2 = _mm_load_pd(x + i);
        __m128d y2 = _mm_load_pd(y + i);
        _mm_store_pd(x + i, _mm_add_pd(x2, y2));
    }
    for(; i < N; i++) {
        x[i] += y[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector sum (low-level)
 *
 * Low level vector sum for AVX(256) intrinsics.
 * Calculates vector of the form: \f$ x = x + y \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param N Size of the vectors
 */
inline void _add256_d(double x[], const double y[], const uint32_t N) {
    uint32_t i = 0;
    for(; i < ROUND_DOWN(N, 4); i += 4) {
        __m256d x4 = _mm256_load_pd(x + i);
        __m256d y4 = _mm256_load_pd(y + i);
        x4 = _mm256_add_pd(x4, y4);
        _mm256_store_pd(x + i, y4);
    }
    for(; i < N; i++) {
        x[i] += y[i];
    }
}
} /* END namespace low */

/*!
 * \ingroup SSE2
 * \brief Vector update (high-level)
 *
 * Calls for wrp::low::_add128_d() function in OpenMP loop to calculate
 * vector of the form: \f$ x = x + y \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param N Size of the vectors
 * @param nthreads Number of OpenMP threads
 */
inline void _add_d(double x[], const double y[], const uint32_t N) {
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
        low::_add128_d(&x[i * offset], &y[i * offset], offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] += y[i];
    }
}
}

#endif /* ADD_H_ */
