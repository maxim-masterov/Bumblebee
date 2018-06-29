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
#include <Epetra_Export.h>
#include <Epetra_Import.h>
#include <AztecOO_config.h>
#include <AztecOO.h>
#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"
#include "ml_epetra_utils.h"

#include "IterativeSolvers.h"

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
    }

    if ( MyGlobalElements[i] % coeff ) {
        weights.name.Bottom = offd;
    }

    weights.name.Cent = diag;

    if ( (MyGlobalElements[i] + 1) % coeff ) {
        weights.name.Top = offd;
    }

    if (_nz > 1) {
        if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < NumGlobalElements - _nz) {
            l = i / nynz;
            ++l;
            if (i < (nynz * l - _nz)) {
                weights.name.North = offd;
            }
        }
    }

    if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < NumGlobalElements - nynz) {
        weights.name.East = offd;
    }

//    std::cout << i << ": ";
//    for(int n = 0; n < 7; ++n)
//        std::cout << weights.data[n] << " ";
//    std::cout << "\n";
}

void AssembleMatrixGlob(int dimension, Epetra_Map *Map, int _nx, int _ny, int _nz, Epetra_CrsMatrix *A) {

    int NumMyElements = Map->NumMyElements();           // Number of local elements
    int *MyGlobalElements = Map->MyGlobalElements();    // Global index of local elements
    int NumGlobalElements = Map->NumGlobalElements();   // Number of global elements

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

void UpdateMatrixGlob(int dimension, Epetra_Map *Map, int _nx, int _ny, int _nz, Epetra_CrsMatrix *A) {

    int NumMyElements = Map->NumMyElements();           // Number of local elements
    int *MyGlobalElements = Map->MyGlobalElements();    // Global index of local elements
    int NumGlobalElements = Map->NumGlobalElements();   // Number of global elements

    std::vector<double> Values(dimension + 1);
    std::vector<int> Indices(dimension + 1);
    _neighb weights;

    int nynz = _ny * _nz;


//    double *values = A->ExpertExtractValues();
//    Epetra_IntSerialDenseVector *indices = &A->ExpertExtractIndices();
//    Epetra_IntSerialDenseVector *offsets = &A->ExpertExtractIndexOffset();

    for(int i = 0; i < NumMyElements; ++i) {

        int NumEntries = 0;

        BuildCoefficients(i, _ny, _nz, MyGlobalElements, NumGlobalElements, weights);

        if (fabs(weights.name.West) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - nynz;
            Values[NumEntries] = weights.name.West / 2.;
            ++NumEntries;
        }

        if (fabs(weights.name.South) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - _nz;
            Values[NumEntries] = weights.name.South / 2.;
            ++NumEntries;
        }

        if (fabs(weights.name.Bottom) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - 1;
            Values[NumEntries] = weights.name.Bottom / 2.;
            ++NumEntries;
        }

        Indices[NumEntries] = MyGlobalElements[i];
        Values[NumEntries] = weights.name.Cent / 2.;
        ++NumEntries;

        if (fabs(weights.name.Top) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + 1;
            Values[NumEntries] = weights.name.Top / 2.;
            ++NumEntries;
        }

        if (fabs(weights.name.North) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + _nz;
            Values[NumEntries] = weights.name.North / 2.;
            ++NumEntries;
        }

        if (fabs(weights.name.East) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + nynz;
            Values[NumEntries] = weights.name.East / 2.;
            ++NumEntries;
        }

        // Put in off-diagonal entries
        A->ReplaceGlobalValues(MyGlobalElements[i], NumEntries, Values.data(), Indices.data());
//        memcpy(values+offsets->operator ()(i), Values.data(), sizeof(double)*NumEntries);
    }

    Values.clear();
    Indices.clear();
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
        _nx = _ny = 4;
        _nz = 1;
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

    /*
     * Lowlevel matrix assembly (row-by-row from left to right)
     */
    Epetra_CrsMatrix *A;
    A = new Epetra_CrsMatrix(Copy, Map, dimension+1, false);

    time1 = time.WallTime();
    AssembleMatrixGlob(dimension, &Map, _nx, _ny, _nz, A);
    time2 = time.WallTime();
    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
    if (myRank == 0) {
        full = max_time - min_time;
        std::cout << "Assembly time: " << full << std::endl;
    }

//    time1 = time.WallTime();
//    UpdateMatrixGlob(dimension, &Map, _nx, _ny, _nz, &A);
//    time2 = time.WallTime();
//    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
//    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
//    if (myRank == 0) {
//        full = max_time - min_time;
//        std::cout << "Update time: " << full << std::endl;
//    }

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
    double time3 = time.WallTime();
    ML_Epetra::MultiLevelPreconditioner* MLPrec =
      new ML_Epetra::MultiLevelPreconditioner(*A, MLList);

    double time4 = time.WallTime();
    MPI_Reduce(&time3, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
    MPI_Reduce(&time4, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
    if (myRank == 0) {
        full = max_time - min_time;
        std::cout << "ML time: " << "\t" << full << std::endl;
    }

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
    Epetra_LinearProblem problem;

    problem.SetOperator(A);
    problem.SetLHS(&x);
    problem.SetRHS(&b);

    // Create AztecOO instance
    AztecOO solver(problem);

    solver.SetPrecOperator(MLPrec);
    solver.SetAztecOption(AZ_conv, AZ_noscaled);
    solver.SetAztecOption(AZ_solver, AZ_bicgstab);
//    solver.SetAztecOption(AZ_output, 0);
//    solver.SetAztecOption(AZ_precond, AZ_ilu);//AZ_dom_decomp);
//    solver.SetAztecOption(AZ_subdomain_solve, AZ_icc);
    solver.SetAztecOption(AZ_precond, AZ_Jacobi);
    solver.SetAztecOption(AZ_omega, 0.72);
    solver.Iterate(30, 1.0E-8);

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
        std::cout << "Iterations: " << solver.NumIters() << "\n";
        std::cout << "Residual: " << solver.TrueResidual() << "\n";
        std::cout << "Residual: " << solver.ScaledResidual() << "\n";
    }

    std::cout << "\n\n\n" << std::endl;

    /* *************************************** */
    int entries = 0;
    double *values;
//    int *indices;

    int rows = A->NumMyRows();
    values = A->ExpertExtractValues();
    Epetra_IntSerialDenseVector *indices = &A->ExpertExtractIndices();
    Epetra_IntSerialDenseVector *offsets = &A->ExpertExtractIndexOffset();

    for(int i = 0; i < rows; ++i) {
        for(int j = offsets->operator ()(i); j < offsets->operator ()(i+1); ++j) {
            if (i == indices->operator ()(j)) {
                values[j] = i;
                std::cout << "diag : " << values[j] << "\n";
            }
        }
    }

    /* ************************************************** */
    /* 0) Create a map
     * 1) Extract diagonal
     * 2) import diagonal elements */
    /* ************************************************** */
//    int glob_el[2];
//    if (comm.MyPID() == 0) {
//        glob_el[0] = 15;
//        glob_el[1] = 12;
//    }
//    else {
//        glob_el[0] = 0;
//        glob_el[1] = 2;
//    }
//    Epetra_Map TargetMap(-1, 2, glob_el, 0, comm);
//    Epetra_Import Import(TargetMap, A->Map());
//    Epetra_CrsMatrix B(Copy, TargetMap, dimension+1);
//
//    Epetra_Vector diagA(A->Map());
//    Epetra_Vector diagB(TargetMap);
//
//    A->ExtractDiagonalCopy(diagA);
//    diagB.Import(diagA, Import, Insert);
//
//    std::cout << "diagB: " << "\n";
//    for(int n = 0; n < diagB.MyLength(); ++n)
//        std::cout << diagB[n] << "\n";
    /* ************************************************** */


//    std::cout << "error: " << A->ExtractGlobalRowView(1, entries, values, indices) << "\n";
//
//    A->ExtractDiagonalCopy(Diagonal)
//
//    std::cout << "\n";
//    std::cout << "entries: " << entries << "\n";
//    for(int n = 0; n < entries; ++n) {
//        std::cout << values[n] << " " << "\n";
//    }
//    std::cout << *A << "\n";
//    values[0] = -99999.;
    std::cout << *A << "\n";

    int glob_el[2];
    if (comm.MyPID() == 0) {
        glob_el[0] = 15;
        glob_el[1] = 12;
    }
    else {
        glob_el[0] = 0;
        glob_el[1] = 2;
    }
    Epetra_Map TargetMap(-1, 2, glob_el, 0, comm);

//    std::cout << Map << "\n";
//    std::cout << TargetMap << "\n";

//    Epetra_Import Import(TargetMap, Map);
//    Epetra_Vector y(TargetMap);
//
//    for(int n = 0; n < y.MyLength(); ++n)
//        y[n] = 0;
//
//    for(int n = 0; n < x.MyLength(); ++n)
//        x[n] = n;

//    A->ExtractDiagonalCopy(x);

    if (A->Filled())
        std::cout << "************** Matrix A is filled\n";
    else
        std::cout << "-------------- Matrix A is not filled\n";
//    Epetra_Import Import(TargetMap, A->Map());
//    Epetra_CrsMatrix B(Copy, TargetMap, dimension+1);
//
//    Epetra_Vector diagA(A->Map());
//    Epetra_Vector diagB(TargetMap);
//
//    A->ExtractDiagonalCopy(diagA);
//    diagB.Import(diagA, Import, Insert);

//    std::cout << "diagB: " << "\n";
//    for(int n = 0; n < diagB.MyLength(); ++n)
//        std::cout << diagB[n] << "\n";

//    /* How to get a map of ghost elements */
//    const Epetra_Import *imp = A->Importer();
//    if (imp != nullptr) {
//        std::cout << "(" << myRank << ") nums: " << imp->NumSend() << " " << imp->NumRecv() << "\n";
//        std::cout << "(" << myRank << ") NumExportIDs: " << imp->NumExportIDs() << "\n";
//        int numexids = imp->NumExportIDs();
//        int *procs, *ids;
//        ids = imp->ExportLIDs();                // get ids which should be exported
//        procs = imp->ExportPIDs();              // get process ids to which those ids should be exported
//        std::cout << "(" << myRank << ") procs: ";
//        if (procs != nullptr)
//            for(int n = 0; n < numexids; ++n) {
//                std::cout << procs[n] << " ";
//            }
//        std::cout << "\n";
//        std::cout << "(" << myRank << ") ids: ";
//        if (ids !=  nullptr)
//            for(int n = 0; n < numexids; ++n) {
//                std::cout << ids[n] << " ";
//            }
//        std::cout << "\n";
//    }
//    /* ********************************* */

    if (A->Importer() != nullptr)
        std::cout << "(" << myRank << ") A nums: " << A->Importer()->NumSend() << " " << A->Importer()->NumRecv() << "\n";


    /* How to get a map of ghost elements */
    const Epetra_Import *imp = A->Importer();
    const Epetra_Export *exp = A->Exporter();
    if (imp != nullptr) {
        int numexids;
        int *procs, *ids;
        numexids = imp->NumExportIDs();
        ids = imp->ExportLIDs();                // get ids which should be exported
        procs = imp->ExportPIDs();              // get process ids to which those ids should be exported
        std::cout << "(" << myRank << ") procs: ";
        if (procs != nullptr) {
            for(int n = 0; n < numexids; ++n) {
                std::cout << procs[n] << " ";
            }
        }
        std::cout << "\n";
        std::cout << "(" << myRank << ") ids: ";
        if (ids !=  nullptr) {
            for(int n = 0; n < numexids; ++n) {
                std::cout << ids[n] << " ";
            }
        }
        std::cout << "\n";

        std::vector<double> snd(numexids);
        std::vector<int> pid(numexids);
        values = A->ExpertExtractValues();
        indices = &A->ExpertExtractIndices();
        offsets = &A->ExpertExtractIndexOffset();
        std::cout << "(" << myRank << ") : ";
        for(int i = 0; i < numexids; ++i) {
            int loc_row = ids[i];
            for(int j = offsets->operator ()(loc_row); j < offsets->operator ()(loc_row+1); ++j) {
                if (loc_row == indices->operator ()(j)) {
                    snd[i] = values[j];
                    pid[i] = procs[i];
                }
            }

            std::cout << snd[i] << " ";
        }
        std::cout << "\n";

//        int buf = 2;
//        MPI_Status status;
//        if (myRank != 0)
//            // receive message from any source
//            MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
//        else
//            // send reply back to sender of the message received above
//            MPI_Send(&buf, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);

        std::cout << "(" << myRank << ") NumRemoteIDs: " << imp->NumRemoteIDs() << "\n";
//        int numinids = imp->NumRecv();
//        std::vector<double> rcv(numexids);
        MPI_Status status;
        for(uint32_t n = 0; n < snd.size(); ++n) {
            double buf;
            MPI_Recv(&buf, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, comm.Comm(), &status);
            std::cout << "received: '" << buf  << "' from process " << status.MPI_SOURCE << "\n";
        }
        for(uint32_t n = 0; n < snd.size(); ++n) {
            MPI_Send(&snd[0], 1, MPI_DOUBLE, status.MPI_SOURCE, MPI_ANY_TAG, comm.Comm());
        }
    }
    /* ********************************* */

//    std::cout << Map << "\n";
//    std::cout << "error Import: " << B.Import(*A, Import, Insert) << "\n";
//    B.FillComplete();
//    std::cout << "error FillComplete: " << B.FillComplete() << "\n";
//    std::cout << B << "\n";
////    rows = B.NumMyRows();
//
//    if (B.Filled())
//        std::cout << "************** Matrix B is filled\n";
//    else
//        std::cout << "-------------- Matrix B is not filled\n";

//    const Epetra_Import *Importer = A->Importer();

//    B.Import(*A, *Importer, Insert);
//    std::cout << B << "\n";
//    Importer->Print(std::cout);

//    Epetra_Vector diag(B.Map());
//    B.ExtractDiagonalCopy(diag);
//
//    std::cout << "diag: " << "\n";
//    for(int n = 0; n < diag.MyLength(); ++n)
//        std::cout << diag[n] << "\n";

//    double *val_loc = nullptr;
//    int *ind_loc = nullptr;
//    int *off_loc = nullptr;
//    int length = B.NumMyNonzeros();
//    B.ExtractCrsDataPointers(off_loc, ind_loc, val_loc);
//
//    if (off_loc == nullptr)
//        std::cout << "off_loc" << "\n";
//    if (ind_loc == nullptr)
//        std::cout << "ind_loc" << "\n";
//    if (val_loc == nullptr)
//        std::cout << "val_loc" << "\n";
//    std::cout << "rows: " << rows << "\n";
//    std::cout << "indices  values\n";
////    for(int n = 0; n < length; ++n) {
////        std::cout << ind_loc[n] << " " << val_loc[n] << "\n";
////    }
//    std::cout << "offsets\n";
//    for(int n = 0; n < B.NumMyRows(); ++n) {
//        std::cout << off_loc[n] << "\n";
//    }

//    for(int i = 0; i < rows; ++i) {
//        for(int j = offsets->operator ()(i); j < offsets->operator ()(i+1); ++j) {
//            if (i == indices->operator ()(j)) {
//                std::cout << "diag : " << values[j] << "\n";
//            }
//        }
//    }


//    y.Import(x, Import, Insert);
//    std::cout << "NumExportIDs: " << A->Importer()->NumExportIDs() << "\n";
//    std::cout << "NumSameIDs: " << A->Importer()->NumSameIDs() << "\n";
//    std::cout << "NumRemoteIDs: " << A->Importer()->NumRemoteIDs() << "\n";

//    double *t;
//
//    t = y.Values();
//
//    for(int n = 0; n < y.MyLength(); ++n)
//        std::cout << n << ": " << t[n] << "\n";
//
////    std::cout << x << "\n";
////
//    std::cout << y << "\n";

//    delete [] values;
//    delete [] indices;

    //  delete myMap;
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





