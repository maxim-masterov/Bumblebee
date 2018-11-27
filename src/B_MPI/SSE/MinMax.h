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

#ifndef MINMAX_H_
#define MINMAX_H_

#include <x86intrin.h>
#include "../Macros.h"
#ifdef BUMBLEBEE_USE_OPENMP
#include <omp.h>
#endif

namespace wrp_mpi {
/*!
 * \brief Searches for a maximum element in the array
 * @param a Input array
 * @param N Size of the array
 * @return Result
 */
inline double find_max_simd128(const double a[], const uint32_t N) {
    double res;
    __m128d max = _mm_load_pd(&a[0]);
    uint32_t n = 2;
    uint32_t end = ROUND_DOWN(N, 2);

    for(n = 2; n < end; n += 2) {
        __m128d cur = _mm_load_pd(&a[n]);
        max = _mm_max_pd(max, cur);
    }

    for(int i = 0; i < 2; i++) {
        max = _mm_max_pd(max, _mm_shuffle_pd(max, max, 0x93));
    }
    _mm_store_sd(&res, max);

    for(; n < N; ++n) {
        if (a[n] > res) res = a[n];
    }

    return res;
}

/*!
 * \brief Searches for a minimum element in the array
 * @param a Input array
 * @param N Size of the array
 * @return Result
 */
inline double find_min_simd128(const double a[], const uint32_t N) {
    double res;
    __m128d min = _mm_load_pd(&a[0]);
    uint32_t n = 2;
    uint32_t end = ROUND_DOWN(N, 2);

    for(n = 2; n < end; n += 2) {
        __m128d cur = _mm_load_pd(&a[n]);
        min = _mm_min_pd(min, cur);
    }

    for(int i = 0; i < 2; i++) {
        min = _mm_min_pd(min, _mm_shuffle_pd(min, min, 0x93));
    }
    _mm_store_sd(&res, min);

    for(; n < N; ++n) {
        if (a[n] < res) res = a[n];
    }

    return res;
}

inline double find_max_simd128(const double a[], const double b[], const uint32_t N) {
    double res;
    __m128d max_a = _mm_load_pd(&a[0]);
    __m128d max_b = _mm_load_pd(&b[0]);
    uint32_t n = 2;
    uint32_t end = ROUND_DOWN(N, 2);

    for(n = 2; n < end; n += 2) {
        __m128d cur_a = _mm_load_pd(&a[n]);
        max_a = _mm_max_pd(max_a, cur_a);

        __m128d cur_b = _mm_load_pd(&b[n]);
        max_b = _mm_max_pd(max_b, cur_b);
    }

    max_a = _mm_max_pd(max_a, max_b);

    for(int i = 0; i < 2; i++) {
        max_a = _mm_max_pd(max_a, _mm_shuffle_pd(max_a, max_a, 0x93));
    }

    _mm_store_sd(&res, max_a);

    for(; n < N; ++n) {
        if (a[n] > res) res = a[n];

        if (b[n] > res) res = b[n];
    }

    return res;
}

inline double find_min_simd128(const double a[], const double b[], const uint32_t N) {
    double res;
    __m128d min_a = _mm_load_pd(&a[0]);
    __m128d min_b = _mm_load_pd(&b[0]);
    uint32_t n = 2;
    uint32_t end = ROUND_DOWN(N, 2);

    for(n = 2; n < end; n += 2) {
        __m128d cur_a = _mm_load_pd(&a[n]);
        min_a = _mm_min_pd(min_a, cur_a);

        __m128d cur_b = _mm_load_pd(&b[n]);
        min_b = _mm_min_pd(min_b, cur_b);
    }

    min_a = _mm_min_pd(min_a, min_b);

    for(int i = 0; i < 2; i++) {
        min_a = _mm_min_pd(min_a, _mm_shuffle_pd(min_a, min_a, 0x93));
    }

    _mm_store_sd(&res, min_a);

    for(; n < N; ++n) {
        if (a[n] < res) res = a[n];

        if (b[n] < res) res = b[n];
    }

    return res;
}
}

#endif /* MINMAX_H_ */
