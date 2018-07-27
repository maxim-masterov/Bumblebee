/*
 * Utils.h
 *
 *  Created on: Jul 23, 2018
 *      Author: maxim
 */

#ifndef SRC_B_MPI_UTILS_H_
#define SRC_B_MPI_UTILS_H_

#include <sys/time.h>

namespace slv {
/*!
 * \returns real time (used for debugging).
 *
 * \todo: Should be changed, since it is not too smart to look at data and year to calculate seconds
 */
inline double getRealTime()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double) tv.tv_sec + 1.0e-6 * (double) tv.tv_usec;
}
}


#endif /* SRC_B_MPI_UTILS_H_ */
