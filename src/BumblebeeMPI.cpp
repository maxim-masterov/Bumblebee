//============================================================================
// Name        : BumblebeeMPI.cpp
// Author      : Maxim Masterov
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

// Tpetra provides distributed sparse linear algebra
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>

// Belos provides Krylov solvers
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>

// Galeri
#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraProblemFactory.hpp>

// MueLu main header: include most common header files in one line
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>

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
int Full3dDecomposition(Teuchos::RCP<SpMap> &Map, int _Imax, int _Jmax,
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
        _nx = 4;
        _ny = 4;
        _nz = 4;
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
        Full3dDecomposition(myMap, _nx + 1, _ny + 1, _nz + 1, grid, comm);

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

        std::cout << (*A) << "\n";

        /*
         * Create vectors for Unknowns and RHS
         */
        Vector x(myMap);
        Vector b(myMap, false);

        double dx = 1./(_nx-1);
        b.putScalar(1000. * dx * dx);

        /*
         * Create
         */
        time1 = MPI_Wtime();
        // create a parameter list for ML options

    //    RCP<MueLu::TpetraOperator<> > M = MueLu::CreateTpetraPreconditioner((RCP<operator_type>)A, mueluParams);
        slv_mpi::AMG amg;
        slv_mpi::CG solver(*comm->getRawMpiComm());

        Teuchos::ParameterList MLList;
//        ML_Epetra::SetDefaults("SA",MLList);
        MLList.set("multigrid algorithm", "sa");
//        MLList.set("ML output", 0);
        MLList.set("max levels", 5);
        MLList.set("aggregation: type", "uncoupled");
        MLList.set("smoother: type", "chebyshev");
//        MLList.set("chebyshev: degree", 1);
        MLList.set("smoother: pre or post", "both");
        MLList.set("coarse: type", "KLU2");
//        MLList.set("eigen-analysis: type", "cg");
//        MLList.set("eigen-analysis: iterations", 7);


//        Teuchos::RCP<MueLu::TpetraOperator> mueLuPreconditioner;
        Teuchos::ParameterList paramList;
        paramList.set("verbosity", "medium");
        paramList.set("multigrid algorithm", "sa");
        paramList.set("aggregation: type", "uncoupled");
        paramList.set("smoother: type", "CHEBYSHEV");
        paramList.set("coarse: max size", 500);
//        mueLuPreconditioner = Teuchos::rcp(MueLu::CreateTpetraPreconditioner(*A, paramList));

        std::string solverOptionsFile = "amg.xml";
        Teuchos::ParameterList mueluParams;
        Teuchos::updateParametersFromXmlFile(solverOptionsFile, Teuchos::inoutArg(mueluParams));

        amg.SetParameters(mueluParams);
        time1 = MPI_Wtime();
        typedef Tpetra::Operator<double, int, int, KokkosClassic::DefaultNode::DefaultNodeType> operator_type;
        amg.Coarse((Teuchos::RCP<operator_type>)A);
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
        solver.solve(amg, *A, x, b, x);
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

int Full3dDecomposition(Teuchos::RCP<SpMap> &Map, int _Imax, int _Jmax,
    int _Kmax, geo::SMesh &grid, Teuchos::RCP<Teuchos::MpiComm<int> > &comm) {

    dcp::Decomposer decomp;
    double sizes[3];
    fb::Index3 nodes;

    sizes[0] = sizes[1] = sizes[2] = 1.;
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


//// @HEADER
////
//// ***********************************************************************
////
////        MueLu: A package for multigrid based preconditioning
////                  Copyright 2012 Sandia Corporation
////
//// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//// the U.S. Government retains certain rights in this software.
////
//// Redistribution and use in source and binary forms, with or without
//// modification, are permitted provided that the following conditions are
//// met:
////
//// 1. Redistributions of source code must retain the above copyright
//// notice, this list of conditions and the following disclaimer.
////
//// 2. Redistributions in binary form must reproduce the above copyright
//// notice, this list of conditions and the following disclaimer in the
//// documentation and/or other materials provided with the distribution.
////
//// 3. Neither the name of the Corporation nor the names of the
//// contributors may be used to endorse or promote products derived from
//// this software without specific prior written permission.
////
//// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
//// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
//// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////
//// Questions? Contact
////                    Jonathan Hu       (jhu@sandia.gov)
////                    Andrey Prokopenko (aprokop@sandia.gov)
////                    Ray Tuminaro      (rstumin@sandia.gov)
////
//// ***********************************************************************
////
//// @HEADER
////
//// To compile and run this example, Trilinos must be configured with
//// Tpetra, Amesos2, MueLu, Ifpack2, and Belos.
////
//// This example will only work with Tpetra, not Epetra.
////
//// Commonly used options are
////   --matrixType
////   --nx
////   --ny
////   --xmlFile
////
//// "./MueLu_Simple.exe --help" shows all supported command line options.
////
//
//#include <iostream>
//
//// Tpetra provides distributed sparse linear algebra
//#include <Tpetra_CrsMatrix.hpp>
//#include <Tpetra_Vector.hpp>
//
//// Belos provides Krylov solvers
//#include <BelosConfigDefs.hpp>
//#include <BelosLinearProblem.hpp>
//#include <BelosBlockCGSolMgr.hpp>
//#include <BelosPseudoBlockCGSolMgr.hpp>
//#include <BelosBlockGmresSolMgr.hpp>
//#include <BelosTpetraAdapter.hpp>
//
//// Galeri
//#include <Galeri_XpetraParameters.hpp>
//#include <Galeri_XpetraProblemFactory.hpp>
//
//// MueLu main header: include most common header files in one line
//#include <MueLu.hpp>
//#include <MueLu_TpetraOperator.hpp>
//#include <MueLu_CreateTpetraPreconditioner.hpp>
//#include <MueLu_Utilities.hpp>
//
//#include "B_MPI/IterativeSolvers.h"
//#include "B_MPI/AMG/AMG.h"
//#include "B_MPI/Assembly/Poisson.h"
//#include <SMesh.h>
//#include <Decomposer.h>
//#include <Distributor.h>
//
//int main(int argc, char *argv[]) {
//
//  // Define default types
//  typedef double                                      scalar_type;
//  typedef int                                         local_ordinal_type;
//  typedef int                                         global_ordinal_type;
//  typedef KokkosClassic::DefaultNode::DefaultNodeType node_type;
//
//  // Convenient typedef's
//  typedef Tpetra::Operator<scalar_type,local_ordinal_type,global_ordinal_type,node_type>    operator_type;
//  typedef Tpetra::CrsMatrix<scalar_type,local_ordinal_type,global_ordinal_type,node_type>   crs_matrix_type;
//  typedef Tpetra::Vector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>      vector_type;
//  typedef Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type> multivector_type;
//  typedef Tpetra::Map<local_ordinal_type,global_ordinal_type,node_type>                     driver_map_type;
//
//  typedef MueLu::TpetraOperator<scalar_type, local_ordinal_type, global_ordinal_type, node_type> muelu_tpetra_operator_type;
//  typedef MueLu::Utilities<scalar_type,local_ordinal_type,global_ordinal_type,node_type> MueLuUtilities;
//
//  typedef Belos::LinearProblem<scalar_type, multivector_type, operator_type> linear_problem_type;
//  typedef Belos::SolverManager<scalar_type, multivector_type, operator_type> belos_solver_manager_type;
//  typedef Belos::PseudoBlockCGSolMgr<scalar_type, multivector_type, operator_type> belos_pseudocg_manager_type;
//  typedef Belos::BlockGmresSolMgr<scalar_type, multivector_type, operator_type> belos_gmres_manager_type;
//
//  //MueLu_UseShortNames.hpp wants these typedefs.
//  typedef scalar_type         Scalar;
//  typedef local_ordinal_type  LocalOrdinal;
//  typedef global_ordinal_type GlobalOrdinal;
//  typedef node_type           Node;
//# include <MueLu_UseShortNames.hpp>
//
//  typedef Galeri::Xpetra::Problem<Map,CrsMatrixWrap,MultiVector> GaleriXpetraProblem;
//
//  using Teuchos::RCP; // reference count pointers
//  using Teuchos::rcp; // reference count pointers
//
//  //
//  // MPI initialization using Teuchos
//  //
//
//  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
//  RCP< const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
//  int mypid = comm->getRank();
//
//  Teuchos::CommandLineProcessor clp(false);
//
//  // Parameters
//
//  global_ordinal_type nx = 100*100*100;
//  Galeri::Xpetra::Parameters<GO> matrixParameters(clp, nx); // manage parameters of the test case
//  Xpetra::Parameters             xpetraParameters(clp);     // manage parameters of xpetra
//
//  global_ordinal_type maxIts            = 25;
//  scalar_type tol                       = 1e-8;
//  std::string solverOptionsFile         = "amg.xml";
//  std::string krylovSolverType          = "cg";
//
//  clp.setOption("xmlFile",    &solverOptionsFile, "XML file containing MueLu solver parameters");
//  clp.setOption("maxits",     &maxIts,            "maximum number of Krylov iterations");
//  clp.setOption("tol",        &tol,               "tolerance for Krylov solver");
//  clp.setOption("krylovType", &krylovSolverType,  "cg or gmres solver");
//
//  switch (clp.parse(argc, argv)) {
//    case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:        return EXIT_SUCCESS;
//    case Teuchos::CommandLineProcessor::PARSE_ERROR:
//    case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: return EXIT_FAILURE;
//    case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:          break;
//  }
//
//  if (xpetraParameters.GetLib() == Xpetra::UseEpetra) {
//    throw std::invalid_argument("This example only supports Tpetra.");
//  }
//
//  Teuchos::ParameterList mueluParams;
//  Teuchos::updateParametersFromXmlFile(solverOptionsFile, Teuchos::inoutArg(mueluParams));
//
//  //
//  // Construct the problem
//  //
//  std::cout << "problem size: " << matrixParameters.GetNumGlobalElements() << "\n";
//
//  global_ordinal_type indexBase = 0;
//  RCP<const Map> xpetraMap = MapFactory::Build(Xpetra::UseTpetra, matrixParameters.GetNumGlobalElements(), indexBase, comm);
//  RCP<GaleriXpetraProblem> Pr = Galeri::Xpetra::BuildProblem<scalar_type, local_ordinal_type, global_ordinal_type, Map, CrsMatrixWrap, MultiVector>(matrixParameters.GetMatrixType(), xpetraMap, matrixParameters.GetParameterList());
//  RCP<Matrix>  xpetraA = Pr->BuildMatrix();
//  RCP<crs_matrix_type> A = MueLuUtilities::Op2NonConstTpetraCrs(xpetraA);
//  RCP<const driver_map_type> map = MueLuUtilities::Map2TpetraMap(*xpetraMap);
//
//  //
//  // Construct a multigrid preconditioner
//  //
//
//  // Multigrid Hierarchy
//  RCP<muelu_tpetra_operator_type> M = MueLu::CreateTpetraPreconditioner((RCP<operator_type>)A, mueluParams);
//
//  //
//  // Set up linear problem Ax = b and associate preconditioner with it.
//  //
//
//  RCP<multivector_type> X = rcp(new multivector_type(map,1));
//  RCP<multivector_type> B = rcp(new multivector_type(map,1));
//
//  X->putScalar((scalar_type) 0.0);
//  B->randomize();
//
//  RCP<linear_problem_type> Problem = rcp(new linear_problem_type(A, X, B));
//  Problem->setRightPrec(M);
//  Problem->setProblem();
//
//  //
//  // Set up Krylov solver and iterate.
//  //
//  RCP<Teuchos::ParameterList> belosList = rcp(new Teuchos::ParameterList());
//  belosList->set("Maximum Iterations",    maxIts); // Maximum number of iterations allowed
//  belosList->set("Convergence Tolerance", tol);    // Relative convergence tolerance requested
//  belosList->set("Verbosity",             Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
//  belosList->set("Output Frequency",      1);
//  belosList->set("Output Style",          Belos::Brief);
//  belosList->set("Implicit Residual Scaling", "None");
//  RCP<belos_solver_manager_type> solver;
//  if (krylovSolverType == "cg")
//    solver = rcp(new belos_pseudocg_manager_type(Problem, belosList));
//  else if (krylovSolverType == "gmres")
//    solver = rcp(new belos_gmres_manager_type(Problem, belosList));
//  else
//    throw std::invalid_argument("bad Krylov solver type");
//
//  solver->solve();
//  int numIterations = solver->getNumIters();
//
//  Teuchos::Array<typename Teuchos::ScalarTraits<scalar_type>::magnitudeType> normVec(1);
//  multivector_type Ax(B->getMap(),1);
//  multivector_type residual(B->getMap(),1);
//  A->apply(*X, residual);
//  residual.update(1.0, *B, -1.0);
//  residual.norm2(normVec);
//  if (mypid == 0) {
//    std::cout << "number of iterations = " << numIterations << std::endl;
//    std::cout << "||Residual|| = " << normVec[0] << std::endl;
//  }
//
//  return EXIT_SUCCESS;
//}
