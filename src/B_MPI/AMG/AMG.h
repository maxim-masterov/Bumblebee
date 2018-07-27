/*
 * AMG.h
 *
 *  Created on: 21 Jul 2018
 *      Author: maxim
 */

#ifndef AMG_AMG_H_
#define AMG_AMG_H_

#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"
#include "ml_epetra_utils.h"

namespace slv_mpi {

class AMG {
public:
    AMG() : MLPrec(nullptr), is_built(false) { }
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
        MLList = _list;
    }

    /*!
     * \brief Setup for AMG solver
     */
    inline void Coarse(Epetra_CrsMatrix  &Matrix) {
        CleanMemory();
        MLPrec = new ML_Epetra::MultiLevelPreconditioner(Matrix, MLList);
        is_built = true;
    }

    /*!
     * \brief Calls for internal test function (see Trilinos documentation)
     */
    inline void Test() {
        if (MLPrec != nullptr)
            MLPrec->TestSmoothers();
    }

    /*!
     * \brief Calls for a preconditioner
     */
    inline void solve(Epetra_CrsMatrix  &Matrix,
               Epetra_Vector    &x,
               Epetra_Vector    &b,
               bool        IsStandAlone = false) {

        /*
         * Check if coarsening procedure has been called
         */
        if (!is_built) {
            std::cerr << "Error! Coarsening procedure has not been called for the ML solver..." << std::endl;
            return;
        }

        MLPrec->ApplyInverse(b, x);
    }

    inline bool IsBuilt() {
        return is_built;
    }

private:
    void CleanMemory() {
        if (MLPrec != nullptr) {
            delete MLPrec;
            MLPrec = nullptr;
            is_built = false;
        }
    }

private:
    ML_Epetra::MultiLevelPreconditioner *MLPrec;
    Teuchos::ParameterList MLList;
    bool is_built;
};
}



#endif /* AMG_AMG_H_ */
