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
