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

#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Version.hpp>

#include <MueLu.hpp>
//#include <MueLu_Level.hpp>
//#include <MueLu_ParameterListInterpreter.hpp>
//#include <MueLu_MLParameterListInterpreter.hpp>
//#include <MueLu_ML2MueLuParameterTranslator.hpp>

#include "B_MPI/IterativeSolvers.h"
#include "B_MPI/AMG/AMG.h"
#include "B_MPI/Assembly/Poisson.h"
#include <SMesh.h>
#include <Decomposer.h>
#include <Distributor.h>

typedef Tpetra::Map<> SpMap;
typedef Tpetra::CrsMatrix<> SpMatrix;
typedef Tpetra::Vector<> Vector;

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
int Full3dDecomposition(Teuchos::RCP<SpMap> &Map, double L, double H, double D, int _Imax, int _Jmax,
    int _Kmax, geo::SMesh &grid, Teuchos::RCP<Teuchos::MpiComm<int> > &comm);

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

//void
//exampleRoutine (const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
//                std::ostream& out)
//{
//  using std::endl;
//  using Teuchos::Array;
//  using Teuchos::ArrayRCP;
//  using Teuchos::ArrayView;
//  using Teuchos::outArg;
//  using Teuchos::RCP;
//  using Teuchos::rcp;
//  using Teuchos::REDUCE_SUM;
//  using Teuchos::reduceAll;
//  const int myRank = comm->getRank ();
//  // Print out the Tpetra software version information.
//  if (myRank == 0) {
//    out << Tpetra::version () << endl << endl;
//  }
//  // Type of the Tpetra::Map specialization to use.
//  using map_type = Tpetra::Map<>;
//  using vector_type = Tpetra::Vector<double>;
//  using global_ordinal_type = vector_type::global_ordinal_type;
//
//  // Create a Tpetra Map
//  // The total (global, i.e., over all MPI processes) number of
//  // entries in the Map.
//  //
//  // For this example, we scale the global number of entries in the
//  // Map with the number of MPI processes.  That way, you can run this
//  // example with any number of MPI processes and every process will
//  // still have a positive number of entries.
//  const Tpetra::global_size_t numGlobalEntries = 2;//comm->getSize () * 5;
//  const global_ordinal_type indexBase = 0;
//
//  // Construct a Map that puts the same number of equations on each
//  // MPI process.
//
////  numGlobalEntries = 2;
//  global_ordinal_type list[2] = {0, 1};
//
//  RCP<const map_type> contigMap =
//    rcp (new map_type (numGlobalEntries, list, numGlobalEntries, indexBase, comm));
//
////  RCP<const map_type> contigMap =
////    rcp (new map_type (numGlobalEntries, indexBase, comm));
//
////  const global_size_t numGlobalElements,
////           const GlobalOrdinal indexList[],
////           const LocalOrdinal indexListSize,
////           const GlobalOrdinal indexBase,
////           const Teuchos::RCP<const Teuchos::Comm<int> >& comm
//
//  vector_type x (contigMap);
//
//  x.putScalar (42.0);
//}
////
//// The same main() driver routine as in the first Tpetra lesson.
////
//int
//main (int argc, char *argv[])
//{
//  MPI_Init(&argc, &argv);
//  {
//    auto comm = Tpetra::getDefaultComm ();
//    exampleRoutine (comm, std::cout);
//    // Tell the Trilinos test framework that the test passed.
//    if (comm->getRank () == 0) {
//      std::cout << "End Result: TEST PASSED" << std::endl;
//    }
//  }
//  MPI_Finalize();
//  return 0;
//}

//void AssembleMatrixGlob(int dimension, SpMap *Map, int _nx, int _ny, int _nz, SpMatrix *A) {
//
//    int NumMyElements = A->getMap().NumMyElements();           // Number of local elements
//    int *MyGlobalElements = A->getMap().MyGlobalElements();    // Global index of local elements
//    int NumGlobalElements = A->getMap().NumGlobalElements();   // Number of global elements
//
//    std::vector<double> Values(dimension + 1);
//    std::vector<int> Indices(dimension + 1);
//    _neighb weights;
//
//    int nynz = _ny * _nz;
//
//    for(int i = 0; i < NumMyElements; ++i) {
//
//        int NumEntries = 0;
//
//        BuildCoefficients(i, _ny, _nz, MyGlobalElements, NumGlobalElements, weights);
//
//        if (fabs(weights.name.West) > DBL_EPSILON) {
//            Indices[NumEntries] = MyGlobalElements[i] - nynz;
//            Values[NumEntries] = weights.name.West;
//            ++NumEntries;
//        }
//
//        if (fabs(weights.name.South) > DBL_EPSILON) {
//            Indices[NumEntries] = MyGlobalElements[i] - _nz;
//            Values[NumEntries] = weights.name.South;
//            ++NumEntries;
//        }
//
//        if (fabs(weights.name.Bottom) > DBL_EPSILON) {
//            Indices[NumEntries] = MyGlobalElements[i] - 1;
//            Values[NumEntries] = weights.name.Bottom;
//            ++NumEntries;
//        }
//
//        Indices[NumEntries] = MyGlobalElements[i];
//        Values[NumEntries] = weights.name.Cent;
//        ++NumEntries;
//
//        if (fabs(weights.name.Top) > DBL_EPSILON) {
//            Indices[NumEntries] = MyGlobalElements[i] + 1;
//            Values[NumEntries] = weights.name.Top;
//            ++NumEntries;
//        }
//
//        if (fabs(weights.name.North) > DBL_EPSILON) {
//            Indices[NumEntries] = MyGlobalElements[i] + _nz;
//            Values[NumEntries] = weights.name.North;
//            ++NumEntries;
//        }
//
//        if (fabs(weights.name.East) > DBL_EPSILON) {
//            Indices[NumEntries] = MyGlobalElements[i] + nynz;
//            Values[NumEntries] = weights.name.East;
//            ++NumEntries;
//        }
//
//        // Put in off-diagonal entries
//        A->InsertGlobalValues(MyGlobalElements[i], NumEntries, Values.data(), Indices.data());
//    }
//
//    Values.clear();
//    Indices.clear();
//
//    A->FillComplete();
//}

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
    }

    MPI_Init(&argc, &argv);
    {
        Teuchos::RCP<Teuchos::MpiComm<int> > comm;

        /*
         * Sparse matrix. 3 - is a guessed number of non-zeros per row
         */
        int dimension;
        if (_nz == 1)
            dimension = 4;
        else
            dimension = 6;

        geo::SMesh grid;
        Teuchos::RCP<SpMap> myMap;
        Full3dDecomposition(myMap, 0.05, 0.05, 1.4, _nx + 1, _ny + 1, _nz + 1, grid, comm);

        const int myRank = comm->getRank();
        const int numProcs = comm->getSize();

        double min_time, max_time;
        double time1, time2, full;

        if (myRank == 0) std::cout << "Problem size: " << NumGlobalElements << std::endl;

        /*
         * Lowlevel matrix assembly (row-by-row from left to right)
         */
        Teuchos::RCP<SpMatrix> A(new SpMatrix (myMap, dimension+1));

        time1 = MPI_Wtime();
    //    AssembleMatrixGlob(dimension, myMap, _nx, _ny, _nz, A);
        slv_mpi::Poisson poisson;
        poisson.AssemblePoisson(*A, _nx, _ny, _nz);
        time2 = MPI_Wtime();
        MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, *comm->getRawMpiComm());
        MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, *comm->getRawMpiComm());
        if (myRank == 0) {
            full = max_time - min_time;
            std::cout << "Assembly time: " << full << std::endl;
        }


        /*
         * Create vectors for Unknowns and RHS
         */
        Vector x(myMap);
        Vector b(myMap, false);

        double dx = 1./(_nx-1);
        b.putScalar(1000. *dx * dx);

        /*
         * Create
         */
        time1 = MPI_Wtime();
        // create a parameter list for ML options

    //    RCP<MueLu::TpetraOperator<> > M = MueLu::CreateTpetraPreconditioner((RCP<operator_type>)A, mueluParams);
        slv_mpi::AMG amg;
        slv_mpi::BiCGSTAB2 solver(*comm->getRawMpiComm());

        Teuchos::ParameterList MLList;
        ML_Epetra::SetDefaults("SA",MLList);
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


//        Teuchos::RCP<MueLu::TpetraOperator> mueLuPreconditioner;
        Teuchos::ParameterList paramList;
        paramList.set("verbosity", "medium");
        paramList.set("multigrid algorithm", "sa");
        paramList.set("aggregation: type", "uncoupled");
        paramList.set("smoother: type", "CHEBYSHEV");
        paramList.set("coarse: max size", 500);
//        mueLuPreconditioner = Teuchos::rcp(MueLu::CreateTpetraPreconditioner(*A, paramList));

        amg.SetParameters(MLList);
        time1 = MPI_Wtime();
//        amg.Coarse(*A);
        time2 = MPI_Wtime();

        MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, *comm->getRawMpiComm());
        MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, *comm->getRawMpiComm());
        if (myRank == 0) {
            full = max_time - min_time;
            std::cout << "Coarsening time: (" << numProcs << ") " << full << std::endl;
        }

        solver.SetStopCriteria(RNORM);
        solver.SetMaxIter(100);
        solver.SetTolerance(1e-8);
        solver.PrintHistory(true, 1);

        time1 = MPI_Wtime();
    //    solver.solve(amg, *A, x, b, x);
//        solver.solve(*A, x, b, x);
        time2 = MPI_Wtime();

        amg.Destroy();

        /* time2 */
        MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, *comm->getRawMpiComm());
        MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, *comm->getRawMpiComm());
        if (myRank == 0) {
            full = max_time - min_time;
            std::cout << "Solving time: (" << numProcs << ") " << full << std::endl;
        }
    }
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

int Full3dDecomposition(Teuchos::RCP<SpMap> &Map, double L, double H, double D, int _Imax, int _Jmax,
    int _Kmax, geo::SMesh &grid, Teuchos::RCP<Teuchos::MpiComm<int> > &comm) {

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

    comm = Teuchos::rcp(new Teuchos::MpiComm<int> (grid.GetLocMPIComm()));

    Map = Teuchos::rcp(new SpMap(grid.GetDistributor().GetMapGlobToLoc().size(),
                                 list_global_elements.data(),
                                 num_loc_elements,
                                 0,
                                 comm));
    return 0;
}
