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

/*
 * Rock'n'Roll starts here!
 */
#ifndef UPDATE_H_
#define UPDATE_H_

/*!
 * \file Update.h
 * \brief Contains SSE2 optimized vector update
 *
 * Contains SSE2 and OpenMP optimized methods for vector updates, such as
 *  - \f$ x = \alpha x + \beta y + \gamma z + \delta q \f$
 *  - \f$ x = \alpha x + \beta y + \gamma z \f$
 *  - \f$ x = \alpha x + \beta y \f$
 *  - \f$ x = \beta y + \delta z \f$
 *  - \f$ x = x + \beta y \f$
 *  - \f$ x = \alpha x + \beta y\f$, \f$ z = \gamma z + \delta q \f$ (invoked simultaneously)
 */

#include "../Macros.h"
#ifdef BUMBLEBEE_USE_OPENMP
#include <omp.h>
#endif

namespace wrp_mpi {
namespace low {
/*!
 * \ingroup SSE2
 * \brief Vector update (low-level)
 *
 * Low level vector update.
 * Calculates vector of the form: \f$ x = x + \beta y \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param beta Scale factor of the vector \e y
 * @param N Size of the vectors
 */
inline void _upd128_d(double x[], const double y[], const double beta, const uint32_t N) {
    uint32_t i = 0;
    __m128d bc = _mm_set1_pd(beta);
    for(; i < ROUND_DOWN(N, 2); i += 2) {
        __m128d x2 = _mm_load_pd(x + i);
        __m128d y2 = _mm_load_pd(y + i);
        _mm_store_pd(x + i, _mm_add_pd(x2, _mm_mul_pd(y2, bc)));
    }
    for(; i < N; i++) {
        x[i] += beta * y[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector update (low-level)
 *
 * Low level vector update.
 * Calculates vector of the form: \f$ x = \alpha x + \beta y \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param alpha Scale factor of the vector \e x
 * @param beta Scale factor of the vector \e y
 * @param N Size of the vectors
 */
inline void _upd128_d(double x[], const double y[], const double alpha, const double beta,
        const uint32_t N) {
    uint32_t i = 0;
    __m128d ac = _mm_set1_pd(alpha);
    __m128d bc = _mm_set1_pd(beta);
    for(; i < ROUND_DOWN(N, 2); i += 2) {
        __m128d x2 = _mm_load_pd(x + i);
        __m128d y2 = _mm_load_pd(y + i);
        _mm_store_pd(x + i, _mm_add_pd(_mm_mul_pd(x2, ac), _mm_mul_pd(y2, bc)));
    }
    for(; i < N; i++) {
        x[i] = alpha * x[i] + beta * y[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector update (low-level)
 *
 * Low level vector update.
 * Calculates vector of the form: \f$ x = \alpha x + \beta y + \gamma z \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param z Vector to be added
 * @param alpha Scale factor of the vector \e x
 * @param beta Scale factor of the vector \e y
 * @param gamma Scale factor of the vector \e z
 * @param N Size of the vectors
 */
inline void _upd128_d(double x[], const double y[], const double z[], const double alpha,
        const double beta, const double gamma, const uint32_t N) {

    uint32_t i = 0;
    const __m128d ac = _mm_set1_pd(alpha);
    const __m128d bc = _mm_set1_pd(beta);
    const __m128d gc = _mm_set1_pd(gamma);
    __m128d temp = _mm_set1_pd(0.0f);
    for(; i < ROUND_DOWN(N, 2); i += 2) {
        __m128d x2 = _mm_load_pd(x + i);
        __m128d y2 = _mm_load_pd(y + i);
        __m128d z2 = _mm_load_pd(z + i);
        temp = _mm_add_pd(_mm_mul_pd(y2, bc), _mm_mul_pd(z2, gc));
        _mm_store_pd(x + i, _mm_add_pd(_mm_mul_pd(x2, ac), temp));
    }
    for(; i < N; i++) {
        x[i] = alpha * x[i] + beta * y[i] + gamma * z[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector update (low-level)
 *
 * Low level vector update.
 * Calculates vector of the form: \f$ x = \beta y + \gamma z \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param z Vector to be added
 * @param beta Scale factor of the vector \e y
 * @param gamma Scale factor of the vector \e z
 * @param N Size of the vectors
 */
inline void _upd128_d(double x[], const double y[], const double z[], const double beta,
        const double gamma, const uint32_t N) {
    uint32_t i = 0;
    __m128d bc = _mm_set1_pd(beta);
    __m128d gc = _mm_set1_pd(gamma);
    for(; i < ROUND_DOWN(N, 2); i += 2) {
//				__m128d y2 = _mm_loadu_pd(y+i);
//				__m128d z2 = _mm_loadu_pd(z+i);
        __m128d y2 = _mm_load_pd(y + i);
        __m128d z2 = _mm_load_pd(z + i);
        _mm_store_pd(x + i, _mm_add_pd(_mm_mul_pd(y2, bc), _mm_mul_pd(z2, gc)));
    }
    for(; i < N; i++) {
        x[i] = beta * y[i] + gamma * z[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector update (low-level)
 *
 * Low level vector update.
 * Calculates vector of the form: \f$ x = \alpha x + \beta y + \gamma z + \delta q \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param z Vector to be added
 * @param q Vector to be added
 * @param alpha Scale factor of the vector \e x
 * @param beta Scale factor of the vector \e y
 * @param gamma Scale factor of the vector \e z
 * @param delta Scale factor of the vector \e q
 * @param N Size of the vectors
 */
inline void _upd128_d(double x[], const double y[], const double z[], const double q[],
        const double alpha, const double beta, const double gamma, const double delta,
        const uint32_t N) {
    uint32_t i = 0;
    __m128d ac = _mm_set1_pd(alpha);
    __m128d bc = _mm_set1_pd(beta);
    __m128d gc = _mm_set1_pd(gamma);
    __m128d dc = _mm_set1_pd(delta);
    __m128d temp = _mm_set1_pd(0.0f);
    for(; i < ROUND_DOWN(N, 2); i += 2) {
//				__m128d x2 = _mm_loadu_pd(x+i);
//				__m128d y2 = _mm_loadu_pd(y+i);
//				__m128d z2 = _mm_loadu_pd(z+i);
//				__m128d q2 = _mm_loadu_pd(q+i);
        __m128d x2 = _mm_load_pd(x + i);
        __m128d y2 = _mm_load_pd(y + i);
        __m128d z2 = _mm_load_pd(z + i);
        __m128d q2 = _mm_load_pd(q + i);
        temp = _mm_add_pd(_mm_mul_pd(y2, bc), _mm_add_pd(_mm_mul_pd(z2, gc), _mm_mul_pd(q2, dc)));
        _mm_store_pd(x + i, _mm_add_pd(_mm_mul_pd(x2, ac), temp));
    }
    for(; i < N; i++) {
        x[i] = alpha * x[i] + beta * y[i] + gamma * z[i] + delta * q[i];
    }
}
} /* END namespace low */

/*!
 * \ingroup SSE2
 * \brief Vector update (high-level)
 *
 * Calls for one of hte wrp::low::_upd128_d() functions in OpenMP loop to calculate
 * vector of the form: \f$ x = x + \beta y \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param beta Scale factor of the vector \e y
 * @param N Size of the vectors
 * @param nthreads Number of OpenMP threads
 */
inline void _upd_d(double x[], const double y[], const double beta, const uint32_t N) {

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
        _mm_prefetch((char*)&x[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&y[i * offset], _MM_HINT_T1);
        low::_upd128_d(&x[i * offset], &y[i * offset], beta, offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] += beta * y[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector update (high-level)
 *
 * Calls for one of hte wrp::low::_upd128_d() functions in OpenMP loop to calculate
 * vector of the form: \f$ x = \alpha x + \beta y \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param alpha Scale factor of the vector \e x
 * @param beta Scale factor of the vector \e y
 * @param N Size of the vectors
 * @param nthreads Number of OpenMP threads
 */
inline void _upd_d(double x[], const double y[], const double alpha, const double beta,
        const uint32_t N) {

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
        _mm_prefetch((char*)&x[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&y[i * offset], _MM_HINT_T1);
        low::_upd128_d(&x[i * offset], &y[i * offset], alpha, beta, offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] = alpha * x[i] + beta * y[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector update (high-level)
 *
 * Calls for one of hte wrp::low::_upd128_d() functions in OpenMP loop to calculate
 * vector of the form: \f$ x = \alpha x + \beta y + \gamma z \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param z Vector to be added
 * @param alpha Scale factor of the vector \e x
 * @param beta Scale factor of the vector \e y
 * @param gamma Scale factor of the vector \e z
 * @param N Size of the vectors
 * @param nthreads Number of OpenMP threads
 */
inline void _upd_d(double x[], const double y[], const double z[], const double alpha,
        const double beta, const double gamma, const uint32_t N) {

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
        _mm_prefetch((char*)&x[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&y[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&z[i * offset], _MM_HINT_T1);
        low::_upd128_d(&x[i * offset], &y[i * offset], &z[i * offset], alpha, beta, gamma, offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] = alpha * x[i] + beta * y[i] + gamma * z[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector update (high-level)
 *
 * Calls for one of hte wrp::low::_upd128_d() functions in OpenMP loop to calculate
 * vector of the form: \f$ x = \beta y + \gamma z \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param z Vector to be added
 * @param beta Scale factor of the vector \e y
 * @param gamma Scale factor of the vector \e z
 * @param N Size of the vectors
 * @param nthreads Number of OpenMP threads
 */
inline void _upd_d(double x[], const double y[], const double z[], const double beta,
        const double gamma, const uint32_t N) {

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
        _mm_prefetch((char*)&x[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&y[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&z[i * offset], _MM_HINT_T1);
        low::_upd128_d(&x[i * offset], &y[i * offset], &z[i * offset], beta, gamma, offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] = beta * y[i] + gamma * z[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Two vectors update (high-level)
 *
 * Simultaneously Calls for one of hte wrp::low::_upd128_d() functions in OpenMP loop
 * to calculate two vectors of the form: \f$ x = \alpha x + \beta y \f$
 *
 * @param x1 Vector to be updated
 * @param y1 Vector to be added to \e x1
 * @param alpha1 Scale factor of the vector \e x1
 * @param beta1 Scale factor of the vector \e y1
 * @param x2 Vector to be updated
 * @param y2 Vector to be added to \e x2
 * @param alpha2 Scale factor of the vector \e x2
 * @param beta2 Scale factor of the vector \e y2
 * @param N Size of the vectors
 * @param nthreads Number of OpenMP threads
 */
inline void _upd2_d(double x1[], const double y1[], const double alpha1, const double beta1,
        double x2[], const double y2[], const double alpha2, const double beta2, const uint32_t N) {

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
        _mm_prefetch((char*)&x1[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&y1[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&x2[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&y2[i * offset], _MM_HINT_T1);
        low::_upd128_d(&x1[i * offset], &y1[i * offset], alpha1, beta1, offset);
        low::_upd128_d(&x2[i * offset], &y2[i * offset], alpha2, beta2, offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x1[i] = alpha1 * x1[i] + beta1 * y1[i];
        x2[i] = alpha2 * x2[i] + beta2 * y2[i];
    }
}

/*!
 * \ingroup SSE2
 * \brief Vector update (high-level)
 *
 * Calls for one of hte wrp::low::_upd128_d() functions in OpenMP loop to calculate
 * vector of the form: \f$ x = \alpha x + \beta y + \gamma z + \delta q \f$
 *
 * @param x Vector to be updated
 * @param y Vector to be added
 * @param z Vector to be added
 * @param q Vector to be added
 * @param alpha Scale factor of the vector \e x
 * @param beta Scale factor of the vector \e y
 * @param gamma Scale factor of the vector \e z
 * @param delta Scale factor of the vector \e q
 * @param N Size of the vectors
 * @param nthreads Number of OpenMP threads
 */
inline void _upd_d(double x[], const double y[], const double z[], const double q[],
        const double alpha, const double beta, const double gamma, const double delta,
        const uint32_t N) {

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
        _mm_prefetch((char*)&x[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&y[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&z[i * offset], _MM_HINT_T1);
        _mm_prefetch((char*)&q[i * offset], _MM_HINT_T1);
        low::_upd128_d(&x[i * offset], &y[i * offset], &z[i * offset], &q[i * offset], alpha, beta,
                gamma, delta, offset);
    }
    for(uint32_t i = nthreads * offset; i < N; ++i) {
        x[i] = alpha * x[i] + beta * y[i] + gamma * z[i] + delta * q[i];
    }
}

} /* END namespace wrp */

#endif /* DOT_H_ */
