/*
 * AMG.h
 *
 *  Created on: 21 Jul 2018
 *      Author: maxim
 */

#ifndef AMG_AMG_H_
#define AMG_AMG_H_

//#include "ml_include.h"
//#include "ml_MultiLevelPreconditioner.h"
//#include "ml_epetra_utils.h"

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>

namespace slv_mpi {

class AMG {
public:
    AMG() : is_built(false) { }
    ~AMG() {
        CleanMemory();
    }

    inline void Destroy() {
        CleanMemory();
    }
    /*!
     * \brief Sets parameters
     */
    inline void SetParameters(Teuchos::ParameterList &_list) {
        mueluParams = _list;
    }

    /*!
     * \brief Setup for AMG solver
     */
//    template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
    inline void Coarse(const Teuchos::RCP<Tpetra::Operator<double, int, int, KokkosClassic::DefaultNode::DefaultNodeType> > &Matrix) {
        CleanMemory();
        M = MueLu::CreateTpetraPreconditioner<double, int, int, KokkosClassic::DefaultNode::DefaultNodeType>(Matrix, mueluParams, Teuchos::null, Teuchos::null);
        is_built = true;
    }

    /*!
     * \brief Calls for internal test function (see Trilinos documentation)
     */
    inline void Test() {
//        if (M != nullptr)
//            M->TestSmoothers();
    }

    /*!
     * \brief Calls for a preconditioner
     */
    inline void solve(Tpetra::CrsMatrix<> &Matrix,
                     Tpetra::Vector<> &x,
                     Tpetra::Vector<> &b,
                     bool IsStandAlone = false) {

        /*
         * Check if coarsening procedure has been called
         */
        if (!is_built) {
            std::cerr << "Error! Coarsening procedure has not been called for the ML solver..." << std::endl;
            return;
        }

        M->apply(b, x);
    }

    inline bool IsBuilt() {
        return is_built;
    }

private:
    void CleanMemory() {
//        if (MLPrec != nullptr) {
//            delete MLPrec;
//            MLPrec = nullptr;
//            is_built = false;
//        }
    }

private:
    Teuchos::RCP<MueLu::TpetraOperator<double, int, int, KokkosClassic::DefaultNode::DefaultNodeType> > M;
    Teuchos::ParameterList mueluParams;
    bool is_built;
};
}



#endif /* AMG_AMG_H_ */
