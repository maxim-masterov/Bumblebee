//============================================================================
// Name        : BumblebeeMPI.cpp
// Author      : Maxim Masterov
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <mpi.h>
#include <stdio.h>
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
#include <SMesh.h>
#include <Decomposer.h>
#include <Distributor.h>

union _neighb {
    struct {
        double West;
        double South;
        double Bottom;
        double Cent;
        double Top;
        double North;
        double East;

        double do_not_touch_me;     // Dummy member to enforce alignment
    } name;

    double data[8];

    void setZero() {
        for(int n = 0; n < 8; ++n)
            data[n] = 0.;
    }
};

inline double getRealTime();
int Full3dDecomposition(Epetra_Map *&Map, int _Imax, int _Jmax,
    int _Kmax, geo::SMesh &grid, const Epetra_MpiComm &comm);

/*
 * Mimics build of coefficients
 */
void BuildCoefficients(int i, int _ny, int _nz, int *MyGlobalElements, int NumGlobalElements, _neighb &weights) {

    double diag = 4.;
    double offd = -1.;

    if (_nz > 1)
        diag = 6.;

    int nynz = _ny * _nz;
    int l = 1;
    int coeff;

    if (_nz == 1)
        coeff = _ny;
    else
        coeff = _nz;

    weights.setZero();

    if (MyGlobalElements[i] >= nynz && MyGlobalElements[i] < NumGlobalElements) {
        weights.name.West = offd;
    }

    if (_nz > 1) {
        if (MyGlobalElements[i] >= _nz && MyGlobalElements[i] < NumGlobalElements) {
            l = i / nynz;
            if (i >= (nynz * l + _nz)) {
                weights.name.South = offd;
            }
        }
        if ( MyGlobalElements[i] % coeff ) {
            weights.name.Bottom = offd;
        }
    }
    else {
        if ( MyGlobalElements[i] % coeff ) {
            weights.name.South = offd;
        }
    }


    weights.name.Cent = diag;

    if (_nz > 1) {
        if ( (MyGlobalElements[i] + 1) % coeff ) {
            weights.name.Top = offd;
        }

        if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < NumGlobalElements - _nz) {
            l = i / nynz;
            ++l;
            if (i < (nynz * l - _nz)) {
                weights.name.North = offd;
            }
        }
    }
    else {
        if ( (MyGlobalElements[i] + 1) % coeff ) {
            weights.name.North = offd;
        }
    }


    if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < NumGlobalElements - nynz) {
        weights.name.East = offd;
    }

//    double West;
//    double South;
//    double Bottom;
//    double Cent;
//    double Top;
//    double North;
//    double East;
//    std::cout << i << ": ";
//    if (_nz > 1) {
//        for(int n = 0; n < 7; ++n)
//            std::cout << weights.data[n] << " ";
//    }
//    else {
//        std::cout << weights.name.West << " " << weights.name.South << " " << weights.name.Cent << " "
//            << weights.name.North << " " << weights.name.East << " ";
//    }
//    std::cout << "\n";
}

void AssembleMatrixGlob(int dimension, Epetra_Map *Map, int _nx, int _ny, int _nz, Epetra_CrsMatrix *A) {

    int NumMyElements = A->Map().NumMyElements();           // Number of local elements
    int *MyGlobalElements = A->Map().MyGlobalElements();    // Global index of local elements
    int NumGlobalElements = A->Map().NumGlobalElements();   // Number of global elements

    std::vector<double> Values(dimension + 1);
    std::vector<int> Indices(dimension + 1);
    _neighb weights;

    int nynz = _ny * _nz;

    for(int i = 0; i < NumMyElements; ++i) {

        int NumEntries = 0;

        BuildCoefficients(i, _ny, _nz, MyGlobalElements, NumGlobalElements, weights);

        if (fabs(weights.name.West) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - nynz;
            Values[NumEntries] = weights.name.West;
            ++NumEntries;
        }

        if (fabs(weights.name.South) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - _nz;
            Values[NumEntries] = weights.name.South;
            ++NumEntries;
        }

        if (fabs(weights.name.Bottom) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - 1;
            Values[NumEntries] = weights.name.Bottom;
            ++NumEntries;
        }

        Indices[NumEntries] = MyGlobalElements[i];
        Values[NumEntries] = weights.name.Cent;
        ++NumEntries;

        if (fabs(weights.name.Top) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + 1;
            Values[NumEntries] = weights.name.Top;
            ++NumEntries;
        }

        if (fabs(weights.name.North) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + _nz;
            Values[NumEntries] = weights.name.North;
            ++NumEntries;
        }

        if (fabs(weights.name.East) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + nynz;
            Values[NumEntries] = weights.name.East;
            ++NumEntries;
        }

        // Put in off-diagonal entries
        A->InsertGlobalValues(MyGlobalElements[i], NumEntries, Values.data(), Indices.data());
    }

    Values.clear();
    Indices.clear();

    A->FillComplete();
}

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

    if (myRank == 0) std::cout << "Problem size: " << NumGlobalElements << std::endl;

    /*
     * Sparse matrix. 3 - is a guessed number of non-zeros per row
     */
    int dimension;
    if (_nz == 1)
        dimension = 4;
    else
        dimension = 6;

    Epetra_Map *myMap = nullptr;  // if create do not forget do delete!
    geo::SMesh grid;
    Full3dDecomposition(myMap, _nx + 1, _ny + 1, _nz + 1, grid, comm);

    /*
     * Lowlevel matrix assembly (row-by-row from left to right)
     */
    Epetra_CrsMatrix *A;
    A = new Epetra_CrsMatrix(Copy, *myMap, dimension+1, false);

    time1 = time.WallTime();
//    AssembleMatrixGlob(dimension, myMap, _nx, _ny, _nz, A);
    slv_mpi::Poisson poisson;
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
    b.PutScalar(1000. *dx * dx);

    time1 = time.WallTime();

    /* **************************** */
    /* **************************** */
    /* **************************** */

    // create a parameter list for ML options
    Teuchos::ParameterList MLList;
    slv_mpi::AMG amg;
//    slv_mpi::BiCGSTAB2<Epetra_CrsMatrix, Epetra_Vector> solver(comm.Comm(), false);
    slv_mpi::IBiCGSTAB2<Epetra_CrsMatrix, Epetra_Vector> solver(comm.Comm(), false);
    ML_Epetra::SetDefaults("DD",MLList);

//    MLList.set("ML output", 0);
    MLList.set("max levels",7);
//    MLList.set("increasing or decreasing","increasing");
    MLList.set("aggregation: type", "Uncoupled-MIS");
    MLList.set("smoother: type","Chebyshev");
//    MLList.set("smoother: Chebyshev alpha", 0.9);
//    MLList.set("smoother: damping factor", 0.7);
    MLList.set("smoother: sweeps",1);
    MLList.set("smoother: pre or post", "both");
//    MLList.set("coarse: type","Amesos-KLU");
//    MLList.set("eigen-analysis: type", "cg");
//    MLList.set("eigen-analysis: iterations", 15);
//
//    MLList.set("coarse: max size", 2000);
//    MLList.set("aggregation: threshold", 1e-3);

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
    solver.SetMaxIter(100);
    solver.SetTolerance(1e-8);
    solver.PrintHistory(true, 1);

    time1 = time.WallTime();
    for(int n = 0; n < 1; ++n) {
//        solver.solve(amg, *A, x, b, x);
        x.PutScalar(0.);
        solver.solve(*A, x, b, x);
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

#ifdef HAVE_MPI
    // Since you called MPI_Init, you are responsible for calling
    // MPI_Finalize after you are done using MPI.
    (void)MPI_Finalize();
#endif // HAVE_MPI

    return 0;
}

inline double getRealTime() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

int Full3dDecomposition(Epetra_Map *&Map, int _Imax, int _Jmax,
    int _Kmax, geo::SMesh &grid, const Epetra_MpiComm &comm) {

    int my_rank = 0;

    MPI_Comm_rank(comm.Comm(), &my_rank);

    dcp::Decomposer decomp;
    double sizes[3];
    fb::Index3 nodes;

    nodes.i = _Imax;
    nodes.j = _Jmax;
    nodes.k = _Kmax;
    if (_Kmax <= 1)
        nodes.k = 1;

    /* Get distributed scalar grid */
    grid.GetDistributedScalarGrid(
            MPI_COMM_WORLD,
            0,
            1,
            sizes,
            nodes,
            decomp);

    int num_loc_elements = grid.GetDistributor().GetMapLocToGlob().size();
    std::vector<int> list_global_elements(num_loc_elements);

    for(size_t n = 0; n < num_loc_elements; ++n)
        list_global_elements[n] = grid.GetDistributor().GetMapLocToGlob().data()[n];

    Map = new Epetra_Map(
                            -1,
                            num_loc_elements,
                            list_global_elements.data(),
                            0,
                            comm);

    return 0;
}
