/*
 * SSE2.h
 *
 *  Created on: Jul 19, 2016
 *      Author: maxim
 */

#ifndef SSE2_H_
#define SSE2_H_

/*!
 * \file SSE2.h
 * All magic comes from here. File contains headers of OpenMP and SSE2 optimized
 * operations on vectors, such as:
 *  - vector update
 *  - vector-vector dot product
 *  - addition of two vectors
 *  - element-wise division of two vectors
 *  - assigning vector to constant value
 *  - copy of vector into another one
 *
 * \note In order to allow code to use these optimized functions one must define flag
 * #USE_MAGIC_POWDER in the file Macros.h.
 *
 * \note Remember, these APIs are low-level. See description for #USE_MAGIC_POWDER for
 * more information on usage.
 */

/*!
 * \defgroup SSE2 SSE2
 * \brief Group of SSE2 optimized functions
 */

#include <x86intrin.h>

#include "MinMax.h"
#include "Update.h"
#include "Add.h"
#include "Assign.h"
#include "Divide.h"
#include "Dot.h"

namespace wrp_mpi {
/*!
 * \brief Contains low-level APIs for vectors
 *
 * Is a namespace for low-level API which allows manipulations
 * with vectors. It contains SSE2 implementation of most important
 * operations, such as vector update, dot product, addition of two
 * vectors and many others (see SSE2.h)
 */
namespace low {

}
}

#endif /* SSE2_H_ */
