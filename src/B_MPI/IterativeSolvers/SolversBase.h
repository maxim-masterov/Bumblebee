/*
 * SolversBase.h
 *
 *  Created on: Apr 4, 2016
 *      Author: maxim
 */

/*!
 * \file SolversBase.h
 *
 * \brief This file consists of base class for all iterative solvers
 *
 * Class Base contains basic methods for any iterative solver. All derived classes are based on
 * methods which work with highly optimized Eigen library. As a storege format for sparse matrices
 * a Compressed Sparse Row (CSR) format is used.
 * CSR storage format helps to represent matrix as a group of three arrays: coA, jcoA, begA. First
 * one contains all nonzero elements of matrix in sequential row-wise order, i.e. if we consider matrix
 * A of the next view
 *
 *                      [1	0	0]
 *                  A = [0	9	2]
 *                      [3	0	5]
 *
 * it will be written as
 * 			\f[ coA = {1, 9, 2, 3, 5} \f]
 * Next array jcoA consist of columns numbers for each nonzero entry, i.e.
 *			\f[ jcoA = {0, 1, 2, 0, 2} \f]
 * Obviously sizes of both arrays are equivalent to the total number of non zeros in the matrix. Third
 * array, begA, stores offsets of nonzero elements for each row. If we will number all non zeros in the
 * matrix A row-wise then begA can be written as
 * 			\f[ begA = {0, 1, 3, 5} \f]
 * Here the last element is an incremented sequential number of last non-zero entry and helps to loop
 * matrix rows.
 *
 * \note Most iterative methods are implemented in a such way that one can provide either dense or sparse
 * Eigen' matrices.
 */

#ifndef SOLVERSBASE_H_
#define SOLVERSBASE_H_

#include "../Macros.h"
#include <iostream>
#include <cfloat>
#include <cmath>
#include <deque>

//using namespace Eigen;

/*!
 * \defgroup KrylovSolvers Krylov solvers
 * \brief KrylovSolvers Krylov subspace solvers for sparse and dense matrices
 */

/*!
 * \defgroup CommonSolvers Common solvers
 * \brief Common iterative solvers for sparse matrices
 */

namespace slv_mpi {

	/*!
	 * \class Base
	 * \brief Basic class for all iterative solvers
	 *
	 * Base class for iterative solvers. Contains main method to set up solvers, store main data,
	 * such as number of iterations. Also contains some debug methods. This class should be used
	 * if one decided to write additional solvers. Just be sure that your new class is a friend
	 * to Base one
	 */
	class Base {

	protected:
		double		stab_criteria;		//!< Stabilization factor for additional stabilization in some iterative solvers
		double  	eps;            	//!< Stores tolerance for solvers
		double		weight;				//!< Weight of residual' L2-norm
		double		residual_norm;		//!< Residual' L2 norm returned by a solver;
		size_t  	MemoryInUsage;  	//!< Stores total amount of memory in usage (debug variable, should be deleted)
		int     	MaxIter;        	//!< Stores maximum number of iterations for solvers
		int			iterations_num;		//!< Number of iterations returned by a solver
		int			stop_criteria;		//!< Determines convergence criteria (by default \f$ ||r||_2 \f$ is used)
		int			print_each;			//!< Indicates that every n'th iteration will be printed
		uint32_t	queue_length;		//!< Stores size of the queue of residuals which helps to determine stalling of the solvers
		int		check_stalling_each;	//!< Determines frequency of \a res_queue updating
		bool		ifprint;			//!< Indicates if convergence history and other data should be printed or not
		bool		check_stalling;		//!< Indicates if convergence of solvers should be checked for stalling or not (false by default)
		bool		stalled;			//!< True if stalling has been detected, false otherwise
		bool		use_add_stab;		//!< True if additional stabilization should be used (works not for all methods)

		std::deque<double> res_queue;	//!< Queue (FIFO) of norms of the last \a queue_length residuals

		MPI_Comm communicator;



		/*!
		 * \brief Add new element to queue and keep its maximum size
		 *
		 * Method adds new element at the end of queue and remove first if necessary. Since queue is
		 * used to detect stalling of convergence element in queue are presented as absolute value of
		 * ratio between current residual and residual from the previous iteration as follows
		 * \f[
		 * 		el_{new} = \left | \frac {||r||_old - ||r||_new} {||r||_old} \right |
		 * \f]
		 *
		 * @param q Queue to be modified
		 * @param data_new New residual
		 * @param data_old Previous residual
		 */
		inline	void	addToResQueue(double data_new, double data_old) {

			if (data_old != 0) {				// Prevent NaN
				/* r = (d_old - d_new) / d_old */
				data_new -= data_old;
				data_new = data_new / data_old;	// Get percentage
			}
			else
				data_new = 1.;

			if (res_queue.size() == queue_length) {
				res_queue.pop_front();
			}
				res_queue.push_back( fabs(data_new) );
		}

		/*!
		 * Calculates mean of elements in a queue and returns true if it is less than 1 and
		 * false otherwise
		 *
		 * @param q Incoming queue
		 * @return Result
		 */
		inline	bool	checkResQueue( ) {

			/*
			 * Check only if queue is fully filled (otherwise it will be useless)
			 */
			if (res_queue.size() == queue_length) {
				std::deque<double>::iterator it = res_queue.begin();
				double sum = 0.;

				while (it != res_queue.end())
					sum += *it++;

				/*
				 * Return true if difference is less than 1%. Since all elements in queue
				 * are ratios (not percents) we compare with 0.01 (= 1/100)
				 */
				return sum < (0.01*queue_length) ? true : false;
			}

			return false;
		}

		/*!
		 * Cleans queue for further use
		 */
		inline	void	cleanResQueue( ) {
			res_queue.clear();
		}

		/*!
		 * Debug method. Prints out queue length and elements
		 */
		inline	void	printResQueue( ) {

			std::deque<double>::iterator it = res_queue.begin();

			std::cout << queue_length << ": ";
			while (it != res_queue.end())
				std::cout << ' ' << *it++;
			std::cout << std::endl;
		}

	public:

		/*!
		 * Constructor initializes basic parameters as
		 *
		 * @para, _stab_criteria 0.7
		 * @param _EPS 1e-7
		 * @param _MaxIter 100
		 * @param _MemoryInUsage 0.0
		 * @param _iterations_num 0
		 * @param _residual_norm 0.0
		 * @param _stop_criteria \f$ ||r||_2 \f$
		 * @param _weight 1.0
		 * @param _print_each 1
		 * @param _ifprint false
		 * @param _queue_length 0
		 * @param _check_stalling false
		 * @param _stalled false
		 * @param _use_add_stab false
		 */
		Base (MPI_Comm _comm) :
					stab_criteria(0.7),
					eps(1e-7),
					weight(1.),
					residual_norm(0.),
					MemoryInUsage(0),
					MaxIter(100),
					iterations_num(0),
					stop_criteria(RNORM),
					print_each(1),
					queue_length(0),
					check_stalling_each(6),
					ifprint(false),
					check_stalling(false),
					stalled(false),
					use_add_stab(false),
					communicator(_comm) { }

		virtual ~Base() {MemoryInUsage = 0; res_queue.clear(); stalled = false;}

		/*!
		 * \defgroup Solver Solvers preferences
		 * \brief Common solver's preferences
		 */

		/*!
		 * \ingroup Solvers
		 * \brief Switches on and off additional stabilization in some solvers
		 *
		 * Method switches on and off additional stabilization in some Krylov sub-space solvers. It
		 * may improve robustness and convergence rate as well as vice versa, so be aware
		 *
		 * \note \a stab_criteria is 0.7 by default. It may be replaced by any other non-small constant
		 * less than 1. If \a stab_criteria is set to 0 additional stablilization will lead to regular
		 * algorithm.
		 *
		 * For more information see G. Sleijpen, H. van Der Vorst "Hybrid bi-conjugate gradient methods
		 * for CFD problems", Computational Fluid Dynamics REVIEW - 1995, Universiteit Utrecht
		 */
		inline	void	SetAdditionalStabilizaation(bool _switcher, double _stab_criteria = 0.7) {
			use_add_stab = _switcher;
			stab_criteria = _stab_criteria;
		}

		/*!
		 * \ingroup Solver
		 * Low-level method to specify number of residual norms that should be taken into account to
		 * prevent stalling of convergence. In other words it determines the length of queue which
		 * stores norms. If \a _queue_length set to zero no previous data will be stored and checker
		 * will be switched off.
		 *
		 * \note Minimum size of queue is 2. If smaller number was specified queue will use default size,
		 * i.e. 2.
		 *
		 * \note It is very-very simple algorithm. It will not detect staling of oscillatory residuals
		 *
		 * \warning Be very careful with this option. It may prevent convergence if specified wrong
		 * \warning If queue is too long it can significantly affect on solver performance, in the same
		 * time it queue length is too short it will not be able to predict stalling of convergence.
		 * Length of 4-6 is usually quite enough.
		 */
		inline	void	SetResQueueDepth(uint32_t _queue_length) {

			/*
			 * Since _queue_length = 0 means that we switch off residual tracking
			 * and 2 is a minimum required elements in queue to make it workable
			 */
			if (_queue_length == 1)
				_queue_length = 2;

			queue_length = _queue_length;

			cleanResQueue();	// Just to be sure

			/* Switch on/off global boolean variable */
			if (_queue_length != 0)
				check_stalling = true;
			else
				check_stalling = false;

			stalled = false;
		}

		/*!
		 * \ingroup Solver
		 * High-level method to switch on and off checking for convergence stalling based of a history
		 * of residuals
		 *
		 * \note Method switch on stalling and set queue size to be 6. If one wants to explicitly
		 * specify queue length use Base::SetQueueDepth() method
		 *
		 * \note It is very-very simple algorithm. It will not detect staling of oscillatory residuals
		 *
		 * @param _check_stalling Switcher
		 */
		inline	void	SetStallingChecker(bool _switcher) {
			check_stalling = _switcher;
			queue_length = 6;
			stalled = false;
		}

		/*!
		 * \ingroup Solver
		 * \returns True if stalling of convergence was detected
		 */
		inline	bool	IsStalled( ) {
			return stalled;
		}
		/*!
		 * \ingroup Solver
		 * Switch on and off printing of convergence data and set that only each n'th
		 * iteration should be printed
		 */
		inline	void	PrintHistory(bool _switcher, int n = 1)
			{ ifprint = _switcher; print_each = n; }

		/*!
		 * \ingroup Solver
		 * Set frequency of queue update in terms of iterations (default is 6, see constructor), i.e.
		 *  \a res_queue will be updated each \a _freq iteration of the called solver
		 */
		inline	void	SetResQueueFrequency(int _freq)
			{ check_stalling_each = _freq; }

		/*!
		 * \ingroup Solver
		 * Set convergence criteria for solvers
		 *
		 * \note Works only for solvers with Eigen' implementation
		 *
		 * @param _type	Type of convergence criteria
		 * @param _weight Residual norm weight (optional, is 1 by default)
		 */
		inline	void	SetStopCriteria(int _type, double _weight = 1.)
			{ stop_criteria = _type; weight = _weight;}

		/*!
		 * \ingroup Solver
		 * \returns Number of iterations taken by a solver
		 */
		inline	int		Iterations()
			{ return iterations_num; }

		/*!
		 * \ingroup Solver
		 * \returns L2 norm of final residual
		 * \note Result depends on chosen type of stop criteria
		 */
		inline	double	Residual()
			{ return residual_norm; }

		/*!
		 * \ingroup Solver
		 * Set solver tolerance
		 * @param _EPS incoming parameter
		 */
		inline	void    SetTolerance(double _EPS)
			{ eps = _EPS; }

		/*!
		 * \ingroup Solver
		 * Set maximum number of iterations
		 * @param _MaxIter incoming parameter
		 */
		inline	void    SetMaxIter(int _MaxIter)
			{ MaxIter = _MaxIter; }

		/*!
		 * \defgroup Debug Debug methods
		 * \brief Are used for debugging
		 */

		/*!
		 * \ingroup Debug
		 * Set memory in usage (for debug)
		 * @param _MemoryInUsage (input)
		 */
		inline	void    SetMemoryInUsage(size_t _MemoryInUsage)
			{ MemoryInUsage = _MemoryInUsage; }

		/*!
		 * \ingroup Debug
		 * Add memory in usage (for debug)
		 * @param _MemoryInUsage (input)
		 */
		inline	void    AddMemoryInUsage(size_t _MemoryInUsage)
			{ MemoryInUsage += _MemoryInUsage; }

		/*!
		 * \ingroup Debug
		 * Reduce (subtract) memory in usage (for debug)
		 * @param _MemoryInUsage (input)
		 */
		inline	void    SubMemoryInUsage(size_t _MemoryInUsage)
			{ MemoryInUsage -= _MemoryInUsage; }

		/*!
		 * \ingroup Debug
		 * Print total memory in usage and ask if program should be continued
		 * @return Total memory in usage
		 */
		inline	size_t  CheckMemoryInUsage() {

			char    in;

			std::cout << "Total memory in usage:\t" << (double)MemoryInUsage/1024/1024 << " Mb" << std::endl;

			std::cout << "Continue? (y/n) ";
			std::cin  >> in;

			if (in == 'n')
				return 0;
			else
				return MemoryInUsage;
		}

		/*!
		 * \ingroup Debug
		 * Print out matrix in CSR storage format as a dense one by using three arrays of non zeros, columns indices and offsets
		 *
		 * @param Size Number of rows
		 * @param NonZero Non-zero elements of incoming matrix (input)
		 * @param ColNonZero Column numbers of non-zero elements (input)
		 * @param PNonZero Pointers to the first non-zero entries in each row of matrix (input)
		 *
		 * /note Works correct only with matrices without zero rows! (should be fixed)
		 */
		template <class NNZ, class COLS, class OFFS>
		inline	void    PrintMatrix(
					int		Size,
					NNZ		*NonZero,
					COLS	*ColNonZero,
					OFFS	*PNonZero)
		{
			int i, j, n = 0;

			std::cout << "\t";   // columns indices
			for(j = 0; j < Size; ++j) {
				std::cout << j << "\t";
			}
			std::cout << std::endl;

			for(i = 0; i < Size; ++i) {
				n = 0;
				std::cout << i << "\t";  // row indices
				for(j = 0; j < Size; ++j) {
					if ( ColNonZero[PNonZero[i]+n] == j ) {
						std::cout << NonZero[PNonZero[i] + n] << "\t";
						n++;
					}
					else
						std::cout << "0\t";
				}
				std::cout << std::endl;
			}
		}
	};
}

#endif /* SOLVERSBASE_H_ */
