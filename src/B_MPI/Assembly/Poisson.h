/*
 * Poisson.h
 *
 *  Created on: Apr 5, 2016
 *      Author: maxim
 */

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
    Poisson() { }
    ~Poisson() { }

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
    template<typename SpMatrix>
    void AssemblePoisson(
                        SpMatrix &Matrix,
                        int nx,
                        int ny,
                        int nz = 0) {

        int Size;                   // Number of rows of Matrix
        int nynz = 0;               // Number of elements in one YZ-plane
        bool is3D;                  // Helps to assemble matrix for 3d problem
        double diag;                // Diagonal coefficient
        int dimension = 1;          // Number of neighbors
        double off_diag = -1.;      // Off-diagonal coefficient
        double l = 1.;              // Number of YZ plane

        /*
         * Here we check if the problem is 2d or 3d
         */
        if (nz == 0) {  // If problem is 2d
            dimension = 4;                              // Only 4 neighbors for 5-point stencil
            Size = nx * ny;                             // Number of matrix' rows
            diag = dimension;                           // Weight of diagonal coefficient

            nz = ny;                                    // This assignment helps to fill in "nearest" neighbors. In 3d case we first
                                                        // go through z direction, then along y direction and only at the end along
                                                        // x direction. In 2d the role of z direction if fulfilled by y direction,
                                                        // so instead of putting additional if statements in the body of loop below
                                                        // we can simply set number of nodes along "fastest" (i.e. z) direction to
                                                        // number of nodes along y direction

            nynz = ny;                                  // Since we have only one XY plane and there is no additional offset due to
                                                        // the 3rd direction the product of ny*nz is set to be ny

            is3D = false;                               // Specify that current case is not 3d
        }
        else {          // If problem is 3d
            dimension = 6;                              // Only 6 neighbors for 5-point stencil
            Size = nx * ny * nz;                        // Number of matrix' rows
            diag = dimension;                           // Weight of diagonal coefficient
            nynz = ny * nz;                             // Number of nodes in one YZ plane
            is3D = true;                                // Specify that current case is 3d
        }

        double *Values = new double[dimension + 1];             // Array of values in a row (excluding diagonal)
        int *Indices = new int[dimension + 1];                  // Array of column indices (excluding diagonal)
        int NumEntries;                                     // Number of non zeros in a row
        int NumMyElements = Matrix.getMap()->getNodeNumElements();      // Number of local elements
        auto MyGlobalElements = Matrix.getMap()->getMyGlobalIndices();// Global index of local elements

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

            Matrix.insertGlobalValues(MyGlobalElements[i], NumEntries, Values, Indices);
        }

        /*
         * By default next method will optimize storage format
         */
        Matrix.fillComplete();

        delete [] Values;
        delete [] Indices;
    }
};
}

#endif /* ASSEMBLY_POISSON_H_ */
