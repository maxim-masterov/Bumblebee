/*
 * Warppers.h
 *
 *  Created on: Mar 7, 2016
 *      Author: maxim
 */

/*!
 * \file Wrappers.h
 * \brief Contains some wrappers over Eigen library
 *
 * Functions in the file show better performance in comparison with Eigen' implementation.
 * Moreover some of them are parallelized with OpenMP.
 */

#ifndef WRAPPERS_H_
#define WRAPPERS_H_

#include "../Macros.h"
#include "SSE2.h"
#include <typeinfo>
#include <vector>
#include <cmath>
#include <mpi.h>

#ifdef BUMBLEBEE_USE_OPENMP
#include <omp.h>
#endif

/*!
 * \brief Wrappers over Eigen library
 *
 * \warning All methods presented in the namespace do not have any checks and assertions. If you provide wrong data
 * your program may stuck or be terminated. They were written only to accelerate internal calculations and should
 * work good and pay off. If one wants to use them with other data - read, understand, understand all possible
 * pitfalls and only after use it. For instance, SpMMP function was tested only with square matrices and definitely
 * will fell down if one of input matrices won't be filled in but only declared (checked).
 *
 * \note Namespace contains low level Eigen' methods for CSR matrices. They can be easily changed onto own implementation
 * or methods from other libraries if one will find them convenient.
 */
namespace wrp {

/*!
 * \brief Dot product of two dense vectors
 *
 * Function calls for parallelized SSE2 version of vector-vector multiplication if flag
 * USE_MAGIC_POWDER has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 First vector
 * @param v2 Second vector
 * @param size Size of provided vector
 * @return Dot product
 */
template<class Vector>
inline long double Dot(Vector &v1, Vector &v2, uint32_t size) {
    /* Original code */
#ifndef USE_MAGIC_POWDER
    long double res = 0.0L;
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for reduction(+:res)
#endif
    for(uint32_t i = 0; i < size; ++i)
        res += v1[i] * v2[i];
    return res;
#else
    /* Second optimized code */
    long double res = 0.0L;
    uint32_t i = 0;
    int lvl = 4;									// unrolling level
#ifdef USE_MAGIC_POWDER
    _mm_prefetch((char*)v1.Values(), _MM_HINT_T1);
    _mm_prefetch((char*)v2.Values(), _MM_HINT_T1);
#endif
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for reduction(+:res)
#endif
    for(uint32_t i = 0; i < ROUND_DOWN(size, lvl); i+=lvl) {
        __m128d r1 = _mm_mul_pd(_mm_load_pd(v1.Values()+i), _mm_load_pd(v2.Values()+i));
        __m128d r2 = _mm_mul_pd(_mm_load_pd(v1.Values()+i+2), _mm_load_pd(v2.Values()+i+2));
        r1 = _mm_add_pd(r1, r2);
        res += _mm_cvtsd_f64(_mm_hadd_pd(r1, r1));
    }
    for(i = ROUND_DOWN(size, lvl); i < size; ++i)
    res += v1.Values()[i] * v2.Values()[i];

    long double res_comm = res;
    MPI_Allreduce(&res, &res_comm, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return res_comm;
#endif
}

/*!
 * \brief Vector update
 *
 * Updates \e v1 values with scaled values of \e v2, \e v3 and \e v4,
 * \e v1 = a*\e v1 + b*\e v2 + c*\e v3 + d*\e v4.
 *
 * Function calls for parallelized SSE2 version of vector update if flag USE_MAGIC_POWDER
 * has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Vector to be updated
 * @param v2 Vector to be scaled and added
 * @param v3 Vector to be scaled and added
 * @param v4 Vector to be scaled and added
 * @param a Scale factor of \e v1
 * @param b Scale factor of \e v2
 * @param c Scale factor of \e v3
 * @param d Scale factor of \e v4
 * @param size Size of the vectors
 */
template<class Vector>
inline void Update(Vector &v1, Vector &v2, Vector &v3, Vector &v4, double a, double b, double c,
    double d, uint32_t size) {

#ifdef USE_MAGIC_POWDER
    _upd_d(v1.Values(), v2.Values(), v3.Values(), v4.Values(), a, b, c, d, size);
#else
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
        v1[i] = a * v1[i] + b * v2[i] + c * v3[i] + d * v4[i];
#endif
}

/*!
 * \brief Vector update
 *
 * Updates \e v1 values with scaled values of \e v2 and \e v3, \e v1 = a*\e v1 + b*\e v2 + c*\e v3.
 *
 * Function calls for parallelized SSE2 version of vector update if flag USE_MAGIC_POWDER
 * has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Vector to be updated
 * @param v2 Vector to be added and scaled
 * @param size Size of provided vector
 *
 * @param v1 Vector to be updated
 * @param v2 Vector to be scaled and added
 * @param v3 Vector to be scaled and added
 * @param a Scale factor of \e v1
 * @param b Scale factor of \e v2
 * @param c Scale factor of \e v3
 * @param size Size of the vectors
 */
template<class Vector>
inline void Update(Vector &v1, Vector &v2, Vector &v3, double a, double b, double c,
    uint32_t size) {

#ifdef USE_MAGIC_POWDER
    _upd_d(v1.Values(), v2.Values(), v3.Values(), a, b, c, size);
#else
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for OPENMP_SCHEDULE
#endif
    for(int i = 0; i < size; ++i)
        v1[i] = a * v1[i] + b * v2[i] + c * v3[i];
#endif
}

/*!
 * \brief Vector update
 *
 * Updates \e v1 values with scaled values of \e v2, \e v1 = a*\e v1 + b*\e v2.
 *
 * Function calls for parallelized SSE2 version of vector update if flag USE_MAGIC_POWDER
 * has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Vector to be updated
 * @param v2 Vector to be added and scaled
 * @param a Scale factor of \e v1
 * @param b Scale factor of \e v2
 * @param size Size of provided vector
 */
template<class Vector>
inline void Update(Vector &v1, Vector &v2, double a, double b, uint32_t size) {

#ifdef USE_MAGIC_POWDER
    _upd_d(v1.Values(), v2.Values(), a, b, size);
#else
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
        v1[i] = a * v1[i] + b * v2[i];
#endif
}

/*!
 * \brief Vector update
 *
 * Updates \e v1 values with scaled values of \e v2, \e v1 = \e v1 + b*\e v2.
 *
 * Function calls for parallelized version of vector update if flag USE_MAGIC_POWDER
 * has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Vector to be updated
 * @param v2 Vector to be added and scaled
 * @param b Scale factor of \e v2
 * @param size Size of provided vector
 *
 * FIXME: do it properly!
 */
template<class Vector>
inline void Update(Vector &v1, Vector &v2, double b, uint32_t size) {

#ifdef USE_MAGIC_POWDER
    _upd_d(v1.Values(), v2.Values(), b, size);
#else
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
        v1[i] += b * v2[i];
#endif
}

/*!
 * \brief Vector update
 *
 * Updates \e v1 values with scaled values of \e v2 and \e v3, \e v1 = b*\e v2 + c*\e v3.
 *
 * Function calls for parallelized version of vector update if flag USE_MAGIC_POWDER
 * has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Vector to be updated
 * @param v2 Vector to be scaled and added
 * @param v3 Vector to be scaled and added
 * @param b Scale factor of \e v2
 * @param c Scale factor of \e v3
 * @param size Size of provided vector
 */
template<class Vector>
inline void Update(Vector &v1, Vector &v2, Vector &v3, double b, double c, uint32_t size) {

#ifdef USE_MAGIC_POWDER
    _upd_d(v1.Values(), v2.Values(), v3.Values(), b, c, size);
#else
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for OPENMP_SCHEDULE
#endif
    for(int i = 0; i < size; ++i)
        v1[i] = b * v2[i] + c * v3[i];
#endif
}

/*!
 * \brief Two vectors update
 *
 * Updates \e v1 values with scaled values of \e v2, \e v1 = a1*\e v1 + b1*\e v2 and
 * does the same for vector \e v3: \e v3 = a2*\e v3 + b2*\e v4
 *
 * Function calls for parallelized version of vector update if flag USE_MAGIC_POWDER
 * has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Vector to be updated
 * @param v2 Vector to be scaled and added to \e v1
 * @param v3 Vector to be updated
 * @param v4 Vector to be scaled and added to \e v3
 * @param a Scale factor of \e v1
 * @param b Scale factor of \e v2
 * @param c Scale factor of \e v3
 * @param d Scale factor of \e v4
 * @param size Size of the vectors
 */
template<class Vector>
inline void Update2(Vector &v1, Vector &v2, double a, double b, Vector &v3, Vector &v4, double c,
    double d, uint32_t size) {

#ifdef USE_MAGIC_POWDER
    _upd2_d(
        v1.Values(), v2.Values(), a, b,
        v3.Values(), v4.Values(), c, d,
        size);
#else
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for OPENMP_SCHEDULE
#endif
    for(int i = 0; i < size; ++i) {
        v1[i] = a * v1[i] + b * v2[i];
        v3[i] = c * v3[i] + d * v4[i];
    }
#endif
}

/*!
 * \brief Add one vector to another
 *
 * Adds \e v2 vector to \e v1 vector, \e v1 = \e v1 + v2.
 *
 * Function calls for parallelized version of vector addition if flag USE_MAGIC_POWDER
 * has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Vector to be updated
 * @param v2 Vector to be added and scaled
 * @param size Size of provided vector
 */
template<class Vector>
inline void Add(Vector &v1, Vector &v2, uint32_t size) {

#ifdef USE_MAGIC_POWDER
    _add_d(v1.Values(), v2.Values(), size);
#else
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for OPENMP_SCHEDULE
#endif
    for(int i = 0; i < size; ++i)
        v1[i] += v2[i];
#endif
}

/*!
 * \brief L2 norm
 *
 * Function calls for a parallelized version of dot product if flag USE_MAGIC_POWDER
 * has been defined. Function returns square root of result, i.e. L2 norm.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Input vector
 * @param size Size of provided vector
 * @return L2 norm
 */
template<class Vector>
inline double Norm2(Vector &v1, uint32_t size) {
    /*
     * Code below helps to avoid overflows. The idea is quite simple: take logarithm of whole
     * expression, calculate logarithmic values and than obtain true result through exp:
     * log(a0 + a1 + a2 + ...) = log(a0) + log (1 + sum(ai/a0)) = log(a0 * (1 + sum(ai/a0))) =
     * log (a0 + a0 * sum(ai/a0))
     * where a0 = max(abs(ai)). Thus, obtained expression under log we can easily and safely obtain
     * desired value of L2 norm.
     */
    long double xmax = 0., scale;
    long double sum = 0.;

#ifdef USE_MAGIC_POWDER
    xmax = find_max_simd128(v1.Values(), size);
#else
    for(uint32_t i = 0; i < size; ++i) {
        long double xabs = fabs(v1[i]);
        if (xabs > xmax) xmax = xabs;
    }
#endif

    if (xmax == 0.) return 0.;
    scale = 1.0 / xmax;

#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for reduction(+:sum) num_threads(USE_MAGIC_POWDER)
#endif
    for(uint32_t i = 0; i < size; ++i) {
        double xs = scale * v1[i];
        sum += xs * xs;
    }

    long double res = xmax * sqrt(sum);
    long double res_comm = res;
    MPI_Allreduce(&res, &res_comm, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return res_comm;
}

/*!
 * \brief Assigns given value to whole vector
 *
 * Assigns a given value to all coefficients in the vector
 *
 * Function calls for parallelized version of vector addition if flag USE_MAGIC_POWDER
 * has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Vector to be updated
 * @param value Value
 * @param size Size of provided vector
 */
template<class Vector>
inline void Assign(Vector &v1, double const value, uint32_t size) {

#ifdef USE_MAGIC_POWDER
    _assign_d(v1.Values(), value, size);
#else
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for OPENMP_SCHEDULE
#endif
    for(int i = 0; i < size; ++i)
        v1[i] = value;
#endif
}

/*!
 * \brief Sets all values in a vector to zero
 *
 * Assigns a zero to all coefficients in the vector
 *
 * Function calls for parallelized version of vector addition if flag USE_MAGIC_POWDER
 * has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Vector to be updated
 * @param size Size of provided vector
 */
template<class Vector>
inline void Zero(Vector &v1, uint32_t size) {

#ifdef USE_MAGIC_POWDER
    _zero_d(v1.Values(), size);
#else
#ifdef BUMBLEBEE_USE_OPENMP
    #pragma omp parallel for OPENMP_SCHEDULE
#endif
    for(int i = 0; i < size; ++i)
        v1[i] = 0.;
#endif
}

/*!
 * \brief Assigns one vector to another
 *
 * Copies coefficients of \e v2 to vector \e v1
 *
 * Function calls for parallelized version of vector addition if flag USE_MAGIC_POWDER
 * has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Vector to be updated
 * @param v2 Vector to be copied
 * @param size Size of provided vector
 */
template<class Vector>
inline void Copy(Vector &v1, Vector &v2, uint32_t size) {

#ifdef USE_MAGIC_POWDER
    _setto_d(v1.Values(), v2.Values(), size);
#else
#ifdef BUMBLEBEE_USE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    v1[i] = v2[i];
#endif
}

/*!
 * \brief Assigns one scaled vector to another
 *
 * Copies coefficients of \e v2 to vector \e v1 as follows: \e v1 = a * \e v2
 *
 * Function calls for parallelized version of vector addition if flag USE_MAGIC_POWDER
 * has been defined. Otherwise standard Eigen's method will be called.
 *
 * If flag BUMBLEBEE_USE_OPENMP was defined instead of flag USE_MAGIC_POWDER regular OpenMP
 * implementation will be invoked (i.e. without explicit SSE2 usage).
 *
 * @param v1 Vector to be updated
 * @param v2 Vector to be copied
 * @param size Size of provided vector
 */
template<class Vector>
inline void Copy(Vector &v1, Vector &v2, double a, uint32_t size) {

#ifdef USE_MAGIC_POWDER
    _setto_d(v1.Values(), v2.Values(), a, size);
#else
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
        v1[i] = a * v2[i];
#endif
}

} /* END namespace wrap*/

#endif /* WRAPPERS_H_ */
