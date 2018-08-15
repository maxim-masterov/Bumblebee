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
int StripeDecomposition(Epetra_Map *&Map, int _nx, int _ny, int _nz, int _size,
    const Epetra_MpiComm &comm);
int Decomposition2(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm);
int Decomposition3(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm);
int Decomposition4(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm);
int Full3dDecomposition(Epetra_Map *&Map, double L, double H, double D, int _Imax, int _Jmax,
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
        _nx = 20;
        _ny = 20;
        _nz = 80;
        NumGlobalElements = _nx * _ny * _nz;
//        std::cout << "Runned with 1 thread..." << std::endl;
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

//    Epetra_Map Map(NumGlobalElements, 0, comm);

//    int NumMyElements = Map.NumMyElements();            // Number of local elements
//    int *MyGlobalElements = Map.MyGlobalElements();    // Global index of local elements

    /*
     * Sparse matrix. 3 - is a guessed number of non-zeros per row
     */
    int dimension;
    if (_nz == 1)
        dimension = 4;
    else
        dimension = 6;

    Epetra_Map *myMap = nullptr;  // if create do not forget do delete!
//    StripeDecomposition(myMap, _nx, _ny, _nz, NumGlobalElements, comm);
//    Decomposition2(myMap, NumGlobalElements, comm);
    geo::SMesh grid;
    Full3dDecomposition(myMap, 0.05, 0.05, 1.4, _nx + 1, _ny + 1, _nz + 1, grid, comm);

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
//    Epetra_Vector x(Map);
//    Epetra_Vector b(Map, false);

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
//    ML_Epetra::SetDefaults("SA",MLList);
//    // overwrite some parameters. Please refer to the user's guide
//    // for more information
//    // some of the parameters do not differ from their default value,
//    // and they are here reported for the sake of clarity
//    // output level, 0 being silent and 10 verbose
//    MLList.set("ML output", 0);
//    // maximum number of levels
//    MLList.set("max levels",5);
//    // set finest level to 0
//    MLList.set("increasing or decreasing","increasing");
//    // use Uncoupled scheme to create the aggregate
//    MLList.set("aggregation: type", "Uncoupled");
//    // smoother is Chebyshev. Example file
//    // `ml/examples/TwoLevelDD/ml_2level_DD.cpp' shows how to use
//    // AZTEC's preconditioners as smoothers
//    MLList.set("smoother: type","Chebyshev");
//    MLList.set("smoother: sweeps",1);
//    // use both pre and post smoothing
//    MLList.set("smoother: pre or post", "both");
//  #ifdef HAVE_ML_AMESOS
//    // solve with serial direct solver KLU
//    MLList.set("coarse: type","Amesos-KLU");
//  #else
//    // this is for testing purposes only, you should have
//    // a direct solver for the coarse problem (either Amesos, or the SuperLU/
//    // SuperLU_DIST interface of ML)
//    MLList.set("coarse: type","Jacobi");
//  #endif
    // Creates the preconditioning object. We suggest to use `new' and
    // `delete' because the destructor contains some calls to MPI (as
    // required by ML and possibly Amesos). This is an issue only if the
    // destructor is called **after** MPI_Finalize().
    double time3 = time.WallTime();
    ML_Epetra::MultiLevelPreconditioner* MLPrec = nullptr;
//      new ML_Epetra::MultiLevelPreconditioner(*A, MLList);

    double time4 = time.WallTime();
    MPI_Reduce(&time3, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
    MPI_Reduce(&time4, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
    if (myRank == 0) {
        full = max_time - min_time;
        std::cout << "ML time: " << "\t" << full << std::endl;
    }

    // verify unused parameters on process 0 (put -1 to print on all
    // processes)
    if (MLPrec != nullptr)
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
    Epetra_LinearProblem problem;

    problem.SetOperator(A);
    problem.SetLHS(&x);
    problem.SetRHS(&b);

    // Create AztecOO instance
//#define AZ_r0               0 /* ||r||_2 / ||r^{(0)}||_2                      */
//#define AZ_rhs              1 /* ||r||_2 / ||b||_2                            */
//#define AZ_Anorm            2 /* ||r||_2 / ||A||_infty                        */
//#define AZ_sol              3 /* ||r||_infty/(||A||_infty ||x||_1+||b||_infty)*/
//#define AZ_weighted         4 /* ||r||_WRMS                                   */
//#define AZ_expected_values  5 /* ||r||_WRMS with weights taken as |A||x0|     */
//#define AZ_noscaled         6 /* ||r||_2                                      */
//#define AZTECOO_conv_test   7 /* Convergence test will be done via AztecOO    */
//#define AZ_inf_noscaled     8 /* ||r||_infty

//    AztecOO solver(problem);
//    solver.SetAztecOption(AZ_conv, AZ_r0);
//    solver.SetAztecOption(AZ_solver, AZ_bicgstab);
//    solver.SetAztecOption(AZ_precond, AZ_none);
//    solver.SetAztecOption(AZ_output, 1);
//    solver.Iterate(50, 1.0E-8);

    std::cout << "====== " << myMap->NumMyElements() << " " << myMap->NumGlobalElements() << "\n";

//    // Create AztecOO instance
//    AztecOO solver(problem);
//
////    ML_Epetra::MultiLevelPreconditioner MLPrec2(*A, MLList);
//
//    double time5 = time.WallTime();
//    solver.SetPrecOperator(MLPrec);
//    double time6 = time.WallTime();
//    MPI_Reduce(&time5, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
//    MPI_Reduce(&time6, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
//    if (myRank == 0) {
//        full = max_time - min_time;
//        std::cout << "SetPrecOperator() time: " << "\t" << full << std::endl;
//    }
//    solver.SetAztecOption(AZ_conv, AZ_noscaled);
//    solver.SetAztecOption(AZ_solver, AZ_bicgstab);
//    solver.SetAztecOption(AZ_output, 1);
////    solver.SetAztecOption(AZ_precond, AZ_dom_decomp);//AZ_ilu);//AZ_dom_decomp);
////    solver.SetAztecOption(AZ_subdomain_solve, AZ_icc);
//    solver.SetAztecOption(AZ_precond, AZ_Jacobi);
//    solver.SetAztecOption(AZ_omega, 0.72);
//    solver.Iterate(30, 1.0E-8);

    slv_mpi::AMG amg;
    slv_mpi::BiCGSTAB2 solver(comm.Comm());
    ML_Epetra::SetDefaults("DD",MLList);

    MLList.set("ML output", 0);
    MLList.set("max levels",5);
    MLList.set("increasing or decreasing","increasing");
    MLList.set("aggregation: type", "Uncoupled");
    MLList.set("smoother: type","Chebyshev");
    MLList.set("smoother: damping factor", 0.72);
    MLList.set("smoother: sweeps",1);
    MLList.set("smoother: pre or post", "both");
    MLList.set("coarse: type","Amesos-KLU");
    MLList.set("eigen-analysis: type", "cg");
    MLList.set("eigen-analysis: iterations", 7);

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
    solver.solve(amg, *A, x, b, x);
//    solver.solve(*A, x, b, x);
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
    delete MLPrec;

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

int Full3dDecomposition(Epetra_Map *&Map, double L, double H, double D, int _Imax, int _Jmax,
    int _Kmax, geo::SMesh &grid, const Epetra_MpiComm &comm) {

    int my_rank = 0;

    MPI_Comm_rank(comm.Comm(), &my_rank);

    dcp::Decomposer decomp;
    double sizes[3];
    fb::Index3 nodes;

    sizes[0] = L;
    sizes[1] = H;
    sizes[2] = D;

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

int Decomposition4(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm) {

    int my_pid = comm.MyPID();
    int MyElements = 0;
    int *MyGlobalElements;
    switch(my_pid) {
        case 0:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 0;
            MyGlobalElements[1] = 1;
            MyGlobalElements[2] = 4;
            MyGlobalElements[3] = 5;
            break;

        case 1:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 2;
            MyGlobalElements[1] = 3;
            MyGlobalElements[2] = 6;
            MyGlobalElements[3] = 7;
            break;

        case 2:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 8;
            MyGlobalElements[1] = 9;
            MyGlobalElements[2] = 12;
            MyGlobalElements[3] = 13;
            break;

        case 3:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 10;
            MyGlobalElements[1] = 11;
            MyGlobalElements[2] = 14;
            MyGlobalElements[3] = 15;
            break;
    }

    Map = new Epetra_Map(-1, MyElements, MyGlobalElements, 0, comm);

    delete [] MyGlobalElements;
    return 0;
}

int Decomposition3(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm) {

    int my_pid = comm.MyPID();
    int MyElements = 0;
    int *MyGlobalElements;
    switch(my_pid) {
        case 0:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 0;
            MyGlobalElements[1] = 1;
            MyGlobalElements[2] = 4;
            MyGlobalElements[3] = 5;
            break;

        case 1:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 2;
            MyGlobalElements[1] = 3;
            MyGlobalElements[2] = 6;
            MyGlobalElements[3] = 7;
            break;

        case 2:
            MyElements = 8;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 8;
            MyGlobalElements[1] = 9;
            MyGlobalElements[2] = 10;
            MyGlobalElements[3] = 11;
            MyGlobalElements[4] = 12;
            MyGlobalElements[5] = 13;
            MyGlobalElements[6] = 14;
            MyGlobalElements[7] = 15;
            break;
    }

    Map = new Epetra_Map(-1, MyElements, MyGlobalElements, 0, comm);

    delete [] MyGlobalElements;
    return 0;
}


int Decomposition2(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm) {

    int my_pid = comm.MyPID();
    int MyElements = 0;
    int *MyGlobalElements;
    switch(my_pid) {
        case 0:
            MyElements = 8;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 0;
            MyGlobalElements[1] = 1;

            MyGlobalElements[2] = 5;
            MyGlobalElements[3] = 6;

            MyGlobalElements[4] = 10;
            MyGlobalElements[5] = 11;

            MyGlobalElements[6] = 15;
            MyGlobalElements[7] = 16;
            break;

        case 1:
            MyElements = 12;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 2;
            MyGlobalElements[1] = 3;
            MyGlobalElements[2] = 4;

            MyGlobalElements[3] = 7;
            MyGlobalElements[4] = 8;
            MyGlobalElements[5] = 9;

            MyGlobalElements[6] = 12;
            MyGlobalElements[7] = 13;
            MyGlobalElements[8] = 14;

            MyGlobalElements[9] = 17;
            MyGlobalElements[10] = 18;
            MyGlobalElements[11] = 19;
            break;
    }

    Map = new Epetra_Map(-1, MyElements, MyGlobalElements, 0, comm);

    delete [] MyGlobalElements;
    return 0;
}

