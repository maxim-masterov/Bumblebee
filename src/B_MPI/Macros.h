/*
 * Macros.h
 *
 *  Created on: Dec 16, 2015
 *      Author: maxim
 */

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
