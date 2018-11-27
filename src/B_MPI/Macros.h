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

/*!
 * \file Macros.h
 * \brief Contains macros and directives
 *
 * File contains all macros and directives for the library. Also contains debugging symbols which allow
 * to print out main information and time profiling. For more information see explanations below.
 */
#ifndef MACROS_H_
#define MACROS_H_

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))
#define IS_POWER_OF_TWO(x) ((x != 0) && !(x & (x - 1)))
#define IS_EVEN(x) (!(x & 1))
#define IS_ODD(x) (x & 1)

// Compile with BUMBLEBEE_USE_OPENMP to switch on hybrid parallelization
//#define BUMBLEBEE_USE_OPENMP

// Compile with USE_MAGIC_POWDER to use buiil-in utilization of low-level intrinsics
//#define USE_MAGIC_POWDER

/*!
 * Type of convergence criteria
 */
enum ConvergenceCriteria {
	RNORM,									//!< L2-norm of residual vector
	RBNORM,									//!< L2-norm of residual vector normalized by L2-norm of right hand side
	RWNORM,									//!< Weighted L2-norm of residual vector
	RRNORM,									//!< L2-norm of residual vector normalized by L2-norm of initial residual
	INTERN									//!< Default criteria determined for each solver separately (see implementation)
};

#endif /* MACROS_H_ */
