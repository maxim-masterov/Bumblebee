//============================================================================
// Name        : BumblebeeMPI.cpp
// Author      : Maxim Masterov
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <mpi.h>
#include <stdio.h>
#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <sys/time.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Time.h>
#include <Epetra_CrsMatrix.h>
#include <AztecOO_config.h>
#include <AztecOO.h>
#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"

#include "IterativeSolvers.h"

inline double getRealTime();
int StripeDecomposition(Epetra_Map *&Map, int _nx, int _ny, int _nz, int _size,
    const Epetra_MpiComm &comm);

int main(int argc, char** argv) {

    //  int NumMyElements = 1e+5;
    int _nx;
    int _ny;
    int _nz;
    int NumGlobalElements;

    if (argc > 1) {
        if (argc == 2) {
            _nx = atoi(argv[1]);
            _ny = _nz = 1;
        }
        else if (argc == 3) {
            _nx = atoi(argv[1]);
            _ny = atoi(argv[2]);
            _nz = 1;
        }
        else if (argc == 4) {
            _nx = atoi(argv[1]);
            _ny = atoi(argv[2]);
            _nz = atoi(argv[3]);
        }
        else {
            std::cout << "Too many arguments..." << std::endl;
            return 0;
        }
        NumGlobalElements = _nx * _ny * _nz;
    }
    else {
        _nx = _ny = 64;
        _nz = 64;
        NumGlobalElements = _nx * _ny * _nz;
        std::cout << "Runned with 1 thread..." << std::endl;
    }

    MPI_Init(&argc, &argv);
    Epetra_MpiComm comm(MPI_COMM_WORLD);

    // Epetra_Comm has methods that wrap basic MPI functionality.
    // MyPID() is equivalent to MPI_Comm_rank, and NumProc() to
    // MPI_Comm_size.
    //
    // With a "serial" communicator, the rank is always 0, and the
    // number of processes is always 1.
    const int myRank = comm.MyPID();
    const int numProcs = comm.NumProc();
    Epetra_Time time(comm);
    double min_time, max_time;
    double time1, time2, full;

    if (myRank == 0) std::cout << "Problem size: " << NumGlobalElements << std::endl;

    Epetra_Map Map(NumGlobalElements, 0, comm);

    int NumMyElements = Map.NumMyElements();            // Number of local elements
    int *MyGlobalElements = Map.MyGlobalElements();    // Global index of local elements

    std::cout << "NumMyElements: " << NumMyElements << " " << myRank << "\n";
    /*
     * Sparse matrix. 3 - is a guessed number of non-zeros per row
     */
    int dimension;
    if (_nz == 1)
        dimension = 4;
    else
        dimension = 6;

//  Epetra_Map *myMap;  // if create do not forget do delete!
//  StripeDecomposition(myMap, _nx, _ny, _nz, NumGlobalElements, comm);

//  cout << *myMap << endl;

//  Epetra_CrsMatrix A(Copy, *myMap, dimension+1, false);
    Epetra_CrsMatrix A(Copy, Map, dimension+1, false);

    double *Values = new double[dimension];
    int *Indices = new int[dimension];
    double cent = dimension;
    int NumEntries;

    Values[0] = -1.0; Values[1] = -1.0;
    Values[2] = -1.0; Values[3] = -1.0;
    if (_nz > 1) {
        Values[4] = -1.0; Values[5] = -1.0;
    }

    /*
     * Low-level matrix assembly (row-by-row from left to right)
     */
    int nynz = _ny * _nz;
    int y_offset = _ny * _nz - _nz;
    int counter_l = 0, counter_r = 0;
    int l = 1;
    int coeff;

    if (_nz == 1)
        coeff = _ny;
    else
        coeff = _nz;

    for(int i = 0; i < NumMyElements; ++i) {

        NumEntries = 0;

        if (MyGlobalElements[i] >= nynz && MyGlobalElements[i] < NumGlobalElements) {
            Indices[NumEntries] = MyGlobalElements[i] - nynz;
            ++NumEntries;
        }

        if (_nz > 1) {
            if (MyGlobalElements[i] >= _nz && MyGlobalElements[i] < NumGlobalElements) {
                l = i / nynz;
                if (i >= (nynz * l + _nz)) {
                    Indices[NumEntries] = MyGlobalElements[i] - _nz;
                    ++NumEntries;
                }
            }
        }

        if ( MyGlobalElements[i] % coeff ) {
            Indices[NumEntries] = MyGlobalElements[i] - 1;
            ++NumEntries;
        }

        if ( (MyGlobalElements[i] + 1) % coeff ) {
            Indices[NumEntries] = MyGlobalElements[i] + 1;
            ++NumEntries;
        }

        if (_nz > 1) {
            if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < NumGlobalElements - _nz) {
                l = i / nynz;
                ++l;
                if (i < (nynz * l - _nz)) {
                    Indices[NumEntries] = MyGlobalElements[i] + _nz;
                    ++NumEntries;
                }
            }
        }

        if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < NumGlobalElements - nynz) {
            Indices[NumEntries] = MyGlobalElements[i] + nynz;
            ++NumEntries;
        }

        // Put in off-diagonal entries
        A.InsertGlobalValues(MyGlobalElements[i], NumEntries, Values, Indices);
        // Put in the diagonal entry
        A.InsertGlobalValues(MyGlobalElements[i], 1, &cent, MyGlobalElements+i);
    }
    delete [] Values;
    delete [] Indices;

    A.FillComplete();

    /*
     * Epetra vectors for Unknowns and RHS
     */
//    Epetra_Vector x(*myMap);
//    Epetra_Vector b(*myMap, false);
    Epetra_Vector x(Map);
    Epetra_Vector b(Map, false);

    double dx = 1./(_nx-1);
    b.PutScalar(1000. *dx * dx);

//    slv::Precond p;
//    p.build(*balanced_matrix, Jacobi_p);
//  p.print();

    time1 = time.WallTime();

    /* **************************** */
    /* **************************** */
    /* **************************** */

    // create a parameter list for ML options
    Teuchos::ParameterList MLList;
    // Sets default parameters for classic smoothed aggregation. After this
    // call, MLList contains the default values for the ML parameters,
    // as required by typical smoothed aggregation for symmetric systems.
    // Other sets of parameters are available for non-symmetric systems
    // ("DD" and "DD-ML"), and for the Maxwell equations ("maxwell").
    ML_Epetra::SetDefaults("SA",MLList);
    // overwrite some parameters. Please refer to the user's guide
    // for more information
    // some of the parameters do not differ from their default value,
    // and they are here reported for the sake of clarity
    // output level, 0 being silent and 10 verbose
    MLList.set("ML output", 10);
    // maximum number of levels
    MLList.set("max levels",5);
    // set finest level to 0
    MLList.set("increasing or decreasing","increasing");
    // use Uncoupled scheme to create the aggregate
    MLList.set("aggregation: type", "Uncoupled");
    // smoother is Chebyshev. Example file
    // `ml/examples/TwoLevelDD/ml_2level_DD.cpp' shows how to use
    // AZTEC's preconditioners as smoothers
    MLList.set("smoother: type","Chebyshev");
    MLList.set("smoother: sweeps",1);
    // use both pre and post smoothing
    MLList.set("smoother: pre or post", "both");
  #ifdef HAVE_ML_AMESOS
    // solve with serial direct solver KLU
    MLList.set("coarse: type","Amesos-KLU");
  #else
    // this is for testing purposes only, you should have
    // a direct solver for the coarse problem (either Amesos, or the SuperLU/
    // SuperLU_DIST interface of ML)
    MLList.set("coarse: type","Jacobi");
  #endif
    // Creates the preconditioning object. We suggest to use `new' and
    // `delete' because the destructor contains some calls to MPI (as
    // required by ML and possibly Amesos). This is an issue only if the
    // destructor is called **after** MPI_Finalize().
    ML_Epetra::MultiLevelPreconditioner* MLPrec =
      new ML_Epetra::MultiLevelPreconditioner(A, MLList);
    // verify unused parameters on process 0 (put -1 to print on all
    // processes)
    MLPrec->PrintUnused(0);
  #ifdef ML_SCALING
    timeVec[precBuild].value = MPI_Wtime() - timeVec[precBuild].value;
  #endif
    // ML allows the user to cheaply recompute the preconditioner. You can
    // simply uncomment the following line:
    //
    // MLPrec->ReComputePreconditioner();
    //
    // It is supposed that the linear system matrix has different values, but
    // **exactly** the same structure and layout. The code re-built the
    // hierarchy and re-setup the smoothers and the coarse solver using
    // already available information on the hierarchy. A particular
    // care is required to use ReComputePreconditioner() with nonzero
    // threshold.
    // =========================== end of ML part =============================
    // tell AztecOO to use the ML preconditioner, specify the solver
    // and the output, then solve with 500 maximum iterations and 1e-12
    // of tolerance (see AztecOO's user guide for more details)

    /* **************************** */
    /* **************************** */
    /* **************************** */

    // Create Linear Problem
    Epetra_LinearProblem problem(&A, &x, &b);

    // Create AztecOO instance
    AztecOO solver(problem);

    solver.SetPrecOperator(MLPrec);
    solver.SetAztecOption(AZ_conv, AZ_noscaled);
    solver.SetAztecOption(AZ_solver, AZ_bicgstab);
//    solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
//    solver.SetAztecOption(AZ_subdomain_solve, AZ_icc);
    solver.Iterate(1000, 1.0E-8);

//    slv::BiCGSTAB solver;
//    solver.SetStopCriteria(RNORM);
//    solver.SetMaxIter(1000);
//    solver.SetTolerance(1e-8);
//    solver.PrintHistory(true, 1);
//    solver.solve(A, x, b, x);

    /* time2 */
    time2 = time.WallTime();
    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());

    if (myRank == 0) {
        full = max_time - min_time;
        std::cout << numProcs << "\t" << full << std::endl;
    }
#ifdef HAVE_MPI
    // Since you called MPI_Init, you are responsible for calling
    // MPI_Finalize after you are done using MPI.
    (void)MPI_Finalize();
#endif // HAVE_MPI

    //  delete myMap;

    return 0;
}

inline double getRealTime() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

int StripeDecomposition(Epetra_Map *&Map, int _nx, int _ny, int _nz, int _size,
    const Epetra_MpiComm &comm) {

    int myRank = comm.MyPID();
    int numProc = comm.NumProc();
    int chunk_size;
    int chunk_start;
    int chunk_end;
    int nynz;

    /*
     * First check if this kind of decomposition is possible at all
     */
    if (numProc > _nx) {
        if (myRank == 0)
            std::cout << "ERROR: Map for stripe decomposition can't be performed, since number "
                "of cores is greater than number of nodes along x direction.\n"
                "Standard Epetra Map will be create instead...\t" << std::endl;
        Map = new Epetra_Map(_size, 0, comm);
        return 1;
    }

    /*
     * Treat 2d case
     */
    if (_nz == 0) _nz = 1;

    nynz = _ny * _nz;

    chunk_size = _nx / numProc; // because c always round to the lower boundary
    chunk_start = myRank * chunk_size;
    chunk_end = chunk_start + chunk_size;

    /*
     * Assign last process with the end of domain, so it will contain last stripe
     */
    if (myRank == (numProc - 1)) chunk_end = _nx;

    chunk_size = (chunk_end - chunk_start) * nynz;

    Map = new Epetra_Map(_size, chunk_size, 0, comm);

    return 0;
}
