//============================================================================
// Name        : BumblebeeMPI.cpp
// Author      : Maxim Masterov
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <sys/time.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Time.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Export.h>
#include <Epetra_Import.h>
#include <AztecOO_config.h>
#include <AztecOO.h>
#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"
#include "ml_epetra_utils.h"
#include "Teuchos_ParameterList.hpp"

#include "B_MPI/IterativeSolvers.h"
#include "B_MPI/AMG/AMG.h"
#include "B_MPI/Assembly/Poisson.h"

int main(int argc, char** argv) {

    int _nx;
    int _ny;
    int _nz;
    int NumGlobalElements;
    int omp_nthreads = 1;

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
        else if (argc == 5) {
            _nx = atoi(argv[1]);
            _ny = atoi(argv[2]);
            _nz = atoi(argv[3]);
#ifdef BUMBLEBEE_USE_OPENMP
            omp_set_num_threads(atoi(argv[4]));
            omp_nthreads = omp_get_max_threads();
#endif
        }
        else {
            std::cout << "Too many arguments..." << std::endl;
            return 0;
        }
        NumGlobalElements = _nx * _ny * _nz;
    }
    else {
        _nx = 4;
        _ny = 4;
        _nz = 4;
        NumGlobalElements = _nx * _ny * _nz;
    }

    MPI_Init(&argc, &argv);
    Epetra_MpiComm comm(MPI_COMM_WORLD);

    const int myRank = comm.MyPID();
    const int numProcs = comm.NumProc();
    Epetra_Time time(comm);
    double min_time, max_time;
    double time1, time2, full;

    if (myRank == 0) {
        std::cout << "Problem size: " << NumGlobalElements << std::endl;
        std::cout << numProcs << " processes running with " << omp_nthreads << " threads each.\n" << std::endl;
    }

    /*
     * Sparse matrix. 3 - is a guessed number of non-zeros per row
     */
    int dimension;
    if (_nz == 1)
        dimension = 4;
    else
        dimension = 6;

    Epetra_Map *myMap = nullptr;
    myMap = new Epetra_Map(NumGlobalElements, 0, comm);     // Or use any other decomposition

    /*
     * Assemble matrix (row-by-row from left to right)
     */
    Epetra_CrsMatrix *A;
    A = new Epetra_CrsMatrix(Copy, *myMap, dimension+1, false);

    slv_mpi::Poisson poisson;
    time1 = time.WallTime();
    poisson.AssemblePoisson(*A, _nx, _ny, _nz);
    time2 = time.WallTime();
    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
    if (myRank == 0) {
        full = max_time - min_time;
        std::cout << "Assembly time: " << full << std::endl;
    }

    /*
     * Epetra vectors for Unknowns and RHS
     */
    Epetra_Vector x(*myMap);
    Epetra_Vector b(*myMap, false);

    double dx = 1./(_nx-1);
#ifdef BUMBLEBEE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int n = 0; n < myMap->NumMyElements(); ++n) {
        b[n] = 1000. * dx * dx;
        x[n] = 0.;
    }

    time1 = time.WallTime();

    /* **************************** */
    /* **************************** */
    /* **************************** */

    // create a parameter list for ML options
    Teuchos::ParameterList MLList;
    slv_mpi::AMG amg;
    slv_mpi::BiCGSTAB2<Epetra_CrsMatrix, Epetra_Vector> solver(comm.Comm(), false);
    ML_Epetra::SetDefaults("DD",MLList);

    MLList.set("max levels",7);
    MLList.set("aggregation: type", "Uncoupled-MIS");
    MLList.set("smoother: type","Chebyshev");
    MLList.set("smoother: sweeps",1);
    MLList.set("smoother: pre or post", "both");

    amg.SetParameters(MLList);
    time1 = time.WallTime();
    amg.Coarse(*A);
    time2 = time.WallTime();

    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
    if (myRank == 0) {
        full = max_time - min_time;
        std::cout << "Coarsening time: (" << numProcs << ") " << full << std::endl;
    }

    solver.SetStopCriteria(RNORM);
    solver.SetMaxIter(20);
    solver.SetTolerance(1e-8);
    solver.PrintHistory(true, 1);

    time1 = time.WallTime();
    for(int n = 0; n < 5; ++n) {
        x.PutScalar(0.);
        solver.solve(amg, *A, x, b, x);
//        solver.solve(*A, x, b, x);
    }
    time2 = time.WallTime();

    amg.Destroy();

    /* time2 */
    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
    if (myRank == 0) {
        full = max_time - min_time;
        std::cout << "Solving time: (" << numProcs << ") " << full << std::endl;
    }

    /* **************************** */
    /* **************************** */
    /* **************************** */

    delete myMap;
    delete A;

    MPI_Finalize();

    return 0;
}

