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

#ifndef ASSEMBLY_POISSON_H_
#define ASSEMBLY_POISSON_H_

/*!
 * File contains methods to build standard matrices in CSR format
 */

namespace slv_mpi {
	/*!
	 * \brief Class to assemble matrix for Poisson' problem
	 *
	 * Provides fast matrix assembling for 2d or 3d structured Poisson' problem
	 *
	 * 5-point and 7-point stencils are used in 2d and 3d cases correspondingly. To assemble matrix for
	 * 2d problem \a nz can either be set to 0 or can be omitted
	 *
	 *             [   -1    ]               [       -1        ]
	 *         A = [-1  4  -1]     or    A = [-1  -1  6  -1  -1]
	 *             [   -1    ]               [       -1        ]
	 */
	class Poisson {
	public:
		Poisson() {};
		~Poisson() {};

		/*!
		 * The method takes an uninitialized matrix resize it and fill with non zeros
		 *
		 * An example of usage is as follows:
		 * \code
		 * ...
		 * Epetra_MpiComm comm (MPI_COMM_WORLD);
		 * ...
		 * int nx, ny;
		 * int Size;
		 *
		 * nx = ny = 4;
		 *
		 * Epetra_CrsMatrix *Matrix;
		 * slv::Poisson poisson;
		 * poisson.AssemblePoisson(comm, Matrix, nx, ny);
		 *
		 * // Print matrix out
		 * Matrix.Print(std::cout);
		 * \endcode
		 *
		 * @param Matrix Outgoing filled matrix
		 * @param nx Number of nodes along x direction
		 * @param ny Number of nodes along y direction
		 * @param nz Number of nodes along z direction (optional)
		 */
		void	AssemblePoisson(
					const Epetra_MpiComm &comm,
					Epetra_CrsMatrix *&Matrix,
					int nx,
					int ny,
					int nz = 0) {

			int Size;					// Number of rows of Matrix
			int nynz = 0;				// Number of elements in one YZ-plane
			bool is3D;					// Helps to assemble matrix for 3d problem
			double diag;				// Diagonal coefficient
			int dimension = 1;			// Number of neighbors
			double off_diag = -1.;		// Off-diagonal coefficient
			double l = 1.;				// Number of YZ plane

			/*
			 * Here we reserve memory correspondingly to estimated number of non zeros in matrix.
			 */
			if (nz == 0) {	// If problem is 2d
				dimension = 5;								// 5-point stencil in 2d
				Size = nx * ny;								// Number of matrix' rows
				diag = 4.;							// Weight of diagonal coefficient

				nz = ny;									// This assignment helps to fill in "nearest" neighbors. In 3d case we first
															// go through z direction, then along y direction and only at the end along
															// x direction. In 2d the role of z direction if fulfilled by y direction,
															// so instead of putting additional if statements in the body of loop below
															// we can simply set number of nodes along "fastest" (i.e. z) direction to
															// number of nodes along y direction

				nynz = ny;									// Since we have only one XY plane and there is no additional offset due to
															// the 3rd direction the product of ny*nz is set to be ny

				is3D = false;								// Specify that current case is not 3d
			}
			else {			// If problem is 3d
				dimension = 7;								// 7-point stencil in 3d
				Size = nx * ny * nz;						// Number of matrix' rows
				diag = 6.;							// Weight of diagonal coefficient
				nynz = ny * nz;								// Number of nodes in one YZ plane
				is3D = true;								// Specify that current case is 3d
			}

			Epetra_Map Map(Size, 0, comm);

			/*
			 * Allocate memory for a matrix
			 */
			Matrix = new Epetra_CrsMatrix(Copy, Map, dimension, false);

			double *Values = new double[dimension];				// Array of values in a row (excluding diagonal)
			int *Indices = new int[dimension];					// Array of column indices (excluding diagonal)
			int NumEntries;										// Number of non zeros in a row
			int NumMyElements = Map.NumMyElements();			// Number of local elements
			int *MyGlobalElements = Map.MyGlobalElements( );	// Global index of local elements

			/*
			 * Fill off-diagonal elements
			 */
			Values[0] = -1.0; Values[1] = -1.0;
			Values[2] = -1.0; Values[3] = -1.0;
			if (is3D) {		// two more in 3d case
				Values[4] = -1.0; Values[5] = -1.0;
			}

			/*
			 * Low-level matrix assembly (row-by-row from left to right)
			 */
			for(int i = 0; i < NumMyElements; ++i) {

				NumEntries = 0;

				if (MyGlobalElements[i] >= nynz && MyGlobalElements[i] < Size) {
					Indices[NumEntries] = MyGlobalElements[i] - nynz;
					Values[NumEntries] = off_diag;
					++NumEntries;
				}

				if (is3D) {
					if (MyGlobalElements[i] >= nz && MyGlobalElements[i] < Size) {
						/*
						 * Notice: when doing an integer division in c/c++, the result will always be rounded down,
						 * so no need to use floor() or kind of std::modf(row / nynz, &l);
						 */
						l = i / nynz;
						if (i >= (nynz * l + nz)) {
							Indices[NumEntries] = MyGlobalElements[i] - nz;
							Values[NumEntries] = off_diag;
							++NumEntries;
						}
					}
				}

				if ( MyGlobalElements[i] % nz ) {
					Indices[NumEntries] = MyGlobalElements[i] - 1;
					Values[NumEntries] = off_diag;
					++NumEntries;
				}

				Indices[NumEntries] = MyGlobalElements[i];
				Values[NumEntries] = diag;
				++NumEntries;

				if ( (MyGlobalElements[i]+1) % nz ) {
					Indices[NumEntries] = MyGlobalElements[i] + 1;
					Values[NumEntries] = off_diag;
					++NumEntries;
				}

				if (is3D) {
					if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < Size-nz) {
						/*
						 * Notice: when doing an integer division in c/c++, the result will always be rounded down,
						 * so no need to use floor() or kind of std::modf(row / nynz, &l);
						 */
						l = i / nynz;
						++l;
						if (i < (nynz * l - nz)) {
							Indices[NumEntries] = MyGlobalElements[i] + nz;
							Values[NumEntries] = off_diag;
							++NumEntries;
						}
					}
				}

				if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < (Size-nynz)) {
					Indices[NumEntries] = MyGlobalElements[i] + nynz;
					Values[NumEntries] = off_diag;
					++NumEntries;
				}

				// Put in non zero entries
				Matrix->InsertGlobalValues(MyGlobalElements[i], NumEntries, Values, Indices);
			}

			/*
			 * By default next method will optimize storage format
			 */
			Matrix->FillComplete();

			delete [] Values;
			delete [] Indices;
		}

		/*!
		 * Method takes prepared matrix and simply fill it in
		 *
		 * An example of usage is as follows:
		 * \code
		 * ...
		 * Epetra_MpiComm comm (MPI_COMM_WORLD);
		 * ...
		 * int nx, ny;
		 * int Size;
		 * int stencil = 5;
		 *
		 * nx = ny = 4;
		 * GlobalSize = nx * ny;
		 *
		 * // Build matrix
		 * Epetra_Map Map(GlobalSize, 0, comm);
		 * Epetra_CrsMatrix Matrix(Copy, Map, stencil-1, false);
		 *
		 * // Fill in matrix
		 * poisson.AssemblePoisson(Matrix, nx, ny);
		 *
		 * // Print matrix out
		 * Matrix.Print(std::cout);
		 * \endcode
		 *
		 * \warning Before calling of method matrix constructor should be called
		 * \warning Be aware to have same dimensions in provided \a Map and arguments
		 * \a nx, \a ny, \a nz. Notice that Map.NumMyElements() should be equal to
		 * \f$ nx * ny * nz\f$
		 *
		 * @param Matrix Reference to incoming matrix
		 * @param nx Number of elements along x direction
		 * @param ny Number of elements along y direction
		 * @param nz Number of elements along z direction (optional)
		 */
		void	AssemblePoisson(
					Epetra_CrsMatrix &Matrix,
					int nx,
					int ny,
					int nz = 0) {

			Epetra_Map Map = Matrix.RowMap();

			int Size;					// Number of rows of Matrix
			int nynz = 0;				// Number of elements in one YZ-plane
			bool is3D;					// Helps to assemble matrix for 3d problem
			double diag;				// Diagonal coefficient
			int dimension = 1;			// Number of neighbors
			double off_diag = -1.;		// Off-diagonal coefficient
			double l = 1.;				// Number of YZ plane

			/*
			 * Here we check if the problem is 2d or 3d
			 */
			if (nz == 0) {	// If problem is 2d
				dimension = 4;								// Only 4 neighbors for 5-point stencil
				Size = nx * ny;								// Number of matrix' rows
				diag = dimension;							// Weight of diagonal coefficient

				nz = ny;									// This assignment helps to fill in "nearest" neighbors. In 3d case we first
															// go through z direction, then along y direction and only at the end along
															// x direction. In 2d the role of z direction if fulfilled by y direction,
															// so instead of putting additional if statements in the body of loop below
															// we can simply set number of nodes along "fastest" (i.e. z) direction to
															// number of nodes along y direction

				nynz = ny;									// Since we have only one XY plane and there is no additional offset due to
															// the 3rd direction the product of ny*nz is set to be ny

				is3D = false;								// Specify that current case is not 3d
			}
			else {			// If problem is 3d
				dimension = 6;								// Only 6 neighbors for 5-point stencil
				Size = nx * ny * nz;						// Number of matrix' rows
				diag = dimension;							// Weight of diagonal coefficient
				nynz = ny * nz;								// Number of nodes in one YZ plane
				is3D = true;								// Specify that current case is 3d
			}

			double *Values = new double[dimension];				// Array of values in a row (excluding diagonal)
			int *Indices = new int[dimension];					// Array of column indices (excluding diagonal)
			int NumEntries;										// Number of non zeros in a row
			int NumMyElements = Map.NumMyElements();			// Number of local elements
			int *MyGlobalElements = Map.MyGlobalElements( );	// Global index of local elements

			/*
			 * Fill off-diagonal elements
			 */
			Values[0] = -1.0; Values[1] = -1.0;
			Values[2] = -1.0; Values[3] = -1.0;
			if (is3D) {		// two more in 3d case
				Values[4] = -1.0; Values[5] = -1.0;
			}

			/*
			 * Low-level matrix assembly (row-by-row from left to right)
			 */
			for(int i = 0; i < NumMyElements; ++i) {

				NumEntries = 0;

				if (MyGlobalElements[i] >= nynz && MyGlobalElements[i] < Size) {
					Indices[NumEntries] = MyGlobalElements[i] - nynz;
					++NumEntries;
				}

				if (is3D) {
					if (MyGlobalElements[i] >= nz && MyGlobalElements[i] < Size) {
						/*
						 * Notice: when doing an integer division in c/c++, the result will always be rounded down,
						 * so no need to use floor() or kind of std::modf(row / nynz, &l);
						 */
						l = i / nynz;
						if (i >= (nynz * l + nz)) {
							Indices[NumEntries] = MyGlobalElements[i] - nz;
							Values[NumEntries] = off_diag;
							++NumEntries;
						}
					}
				}

				if ( MyGlobalElements[i] % nz ) {
					Indices[NumEntries] = MyGlobalElements[i] - 1;
					++NumEntries;
				}

				if ( (MyGlobalElements[i]+1) % nz ) {
					Indices[NumEntries] = MyGlobalElements[i] + 1;
					++NumEntries;
				}

				if (is3D) {
					if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < Size-nz) {
						/*
						 * Notice: when doing an integer division in c/c++, the result will always be rounded down,
						 * so no need to use floor() or kind of std::modf(row / nynz, &l);
						 */
						l = i / nynz;
						++l;
						if (i < (nynz * l - nz)) {
							Indices[NumEntries] = MyGlobalElements[i] + nz;
							++NumEntries;
						}
					}
				}

				if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < (Size-nynz)) {
					Indices[NumEntries] = MyGlobalElements[i] + nynz;
					++NumEntries;
				}

				// Put in off-diagonal entries
				Matrix.InsertGlobalValues(MyGlobalElements[i], NumEntries, Values, Indices);
				// Put in the diagonal entry
				Matrix.InsertGlobalValues(MyGlobalElements[i], 1, &diag, MyGlobalElements+i);
			}

			/*
			 * By default next method will optimize storage format
			 */
			Matrix.FillComplete();

			delete [] Values;
			delete [] Indices;
		}
	};
}

#endif /* ASSEMBLY_POISSON_H_ */
