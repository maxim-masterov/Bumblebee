/*
 * CG.h
 *
 *  Created on: Apr 4, 2016
 *      Author: maxim
 */

/*!
 * \file CG.h
 *
 * \brief File contains class for Conjugate Gradient method
 */

#ifndef CG_H_
#define CG_H_

#include "../SolversBase.h"
#include "../../SSE/Wrappers.h"

namespace slv_mpi {

/*!
 * \ingroup KrylovSolvers
 * \class CG
 * \brief Conjugate Gradient solver
 *
 * Conjugate Gradient method works only with symmetric matrix. It can take non-symmetric
 * matrices as an input but result will be wrong. It is not due to the implementation, it's nature
 * of the method.
 *
 * Class contains two overloaded methods for solving original and preconditioned systems for both
 * dense and sparse incoming matrices
 *
 * The typical usage example is as follows:
 *
 * \code
 * int n = 100;
 * wrp_mpi::VectorD x(n);
 * wrp_mpi::VectorD b(n);
 * SparseMatrix<double, RowMajor> A(n, n);
 *
 * // Fill matrix and right hand side
 * ...
 *
 * CG cg;
 *
 * // Build preconditioner if necessary...
 *
 * // Set type of stopping criteria
 * cg.SetStopCriteria(RBNORM);
 *
 * // Set tolerance
 * cg.ChangeAcc(1e-8);
 *
 * // Set maximum number of iterations
 * cg.ChangeMaxIter(100);
 *
 * // Solve the system
 * cg.solve(A, x, b, x); // Or, in case of preconditioning: cg.solve(Preco, A, x, b, x);
 *
 * // Print out number of iterations and residual
 * std::cout << "Iterations: " << cg.Iterations() << std::endl;
 * std::cout << "Residual: " << cg.Residual() << std::endl;
 * \endcode
 */
class CG: public Base {

public:

    CG(MPI_Comm _comm) :
        Base(_comm) {
    }

    ~CG() {
    }


    /*!
     * \brief Conjugate Gradient method
     *
     * \par For developers
     * Method contains:
     *  - 3 Matrix-Vector Multiplication \n
     *      (2 in loop, 1 out of loop)
     *  - 5 Vector-Vector Multiplication \n
     *      (4 in loop, 1 out of loop)
     *
     * @param Matrix Incoming matrix
     * @param x Vector of unknowns
     * @param b Vector of RHS
     * @param x0 Vector of initial guess
     */
    template<class MatrixType, class VectorType>
    void solve(
                MatrixType &Matrix,
                VectorType &x,
                VectorType &b,
                VectorType &x0);

    /*!
     * \ingroup SparseSolversEigen
     * \brief Preconditioned Conjugate Gradient method
     *
     * Preconditioned Conjugate Gradients method uses right-preconditioned form, i.e.
     *                  \f{eqnarray*}{ A M^{-1} y =  b, \\ x = M^{-1} y \f}
     * where \f$ M^{-1}\f$ is a preconditioning matrix. Matrix \f$ M \f$ can be obtained via different ways, e.g.
     * Incomplete LU factorization, Incomplete Cholesky factorization etc.
     *
     * \par For developers
     * Method contains:
     *  - 3 Matrix-Vector Multiplication \n
     *      (2 in loop, 1 out of loop)
     *  - 5 Vector-Vector Multiplication \n
     *      (4 in loop, 1 out of loop)
     *
     * @param precond Object of preconditioner
     * @param Matrix Incoming matrix
     * @param x Vector of unknowns
     * @param b Vector of RHS
     * @param x0 Vector of initial guess
     */
    template<class Preco, class MatrixType, class VectorType>
    void solve(
                Preco &precond,
                MatrixType &Matrix,
                VectorType &x,
                VectorType &b,
                VectorType &x0);
};

template<class MatrixType, class VectorType>
void CG::solve(
                MatrixType &Matrix,
                VectorType &x,
                VectorType &b,
                VectorType &x0) {

    int k = 0;                              // iteration
    double alpha = 0.;
    double beta = 0.;
    double temp = 0.;
    double delta[2] = {0.};                 // delta[0] - delta_old, delta[1] - delta_new
    double convergence_check = 0.;          // keeps new residual
    double convergence_check_old = 0.;      // keeps old residual and used only is stalling checker is switched on
    double normalizer = 1.;                 // To normalize residual norm

    /*
     * MPI communicators
     */
    const int myRank = x.Comm().MyPID ();
    const Epetra_BlockMap _Map = x.Map();
    int size = _Map.NumMyElements();

    VectorType d(_Map);
    VectorType r(_Map);

    VectorType tmp(_Map);

    //! (1)  \f$ d_0 = r_0 = b - A x_0 \f$
    Matrix.Multiply(false, x0, tmp);
    r = b;
    r.Update(-1., tmp, 1.);
    d = r;

    //! Set \f$ \delta_{new} = \alpha = ||r||_2^2 \f$
    r.Dot(r, &convergence_check);
    delta[1] = alpha = convergence_check;

    /*!
     * Prepare stop criteria
     */
    switch (stop_criteria) {
        case RNORM:
            normalizer = 1.;
            break;
        case RRNORM:
        case INTERN:
            normalizer = sqrt(convergence_check);
            break;
        case RBNORM:
            normalizer = wrp_mpi::Norm2(b.Values(), size, communicator);
            break;
        case RWNORM:
            normalizer = weight;
            break;
        default:
            normalizer = 1.;
            break;
    }

    convergence_check /= normalizer;

    /*
     * Check residual. Stop if initial guess satisfies convergence criteria.
     */
    if (convergence_check < eps) {
        if (ifprint && !(k % print_each))
            if (myRank == 0)
                std::cout << k << '\t' << convergence_check << std::endl;
        iterations_num = k;
        residual_norm = convergence_check;
        return;
    }

    if (ifprint)
        if (myRank == 0)
            std::cout << k << '\t' << convergence_check / normalizer << std::endl;
    ++k;

    //! Start iterative loop
    while (1) {
        if (k > MaxIter) {
            break;
        }

        //! (2) \f$ \alpha = <r, r> / <d, A d_{old}> \f$
        wrp_mpi::Multiply(Matrix, d, tmp, false);
        temp = wrp_mpi::Dot(d.Values(), tmp.Values(), size, communicator);
        alpha /= temp;			// Possible break down if temp == 0.0

        //! (3) \f$ x_{new} = x_{old} + \alpha d_{old} \f$
        //! (4) \f$ r_{new} = r_{old} - \alpha A d_{old} \f$
        wrp_mpi::Update2(
                x.Values(), d.Values(), 1., alpha,
                r.Values(), tmp.Values(), 1., -alpha,
                size);

        //! (5)   \f$ \beta = <r_{new}, r_{new}> / <r_{old}, r_{old}> \f$
        alpha = wrp_mpi::Dot(r.Values(), r.Values(), size, communicator);       // Possible break down if alpha == 0.0
        delta[0] = delta[1];
        delta[1] = alpha;

        if (stop_criteria == INTERN)
            convergence_check = delta[1] / delta[0];
        else
            convergence_check = sqrt(delta[1]) / normalizer;

        /*!
         * Check convergence
         */
        if (ifprint && !(k % print_each)) {
            if (myRank == 0)
                std::cout << k << '\t' << convergence_check << std::endl;
        }

        if (convergence_check <= eps) {
            break;
        }

        beta = delta[1] / delta[0];

        //! (6)  \f$ d_{new} = r_{new} + \beta d_{old} \f$
        wrp_mpi::Update(d.Values(), r.Values(), beta, 1., size);

        /*
         * Check for convergence stalling
         */
        if (check_stalling && !(k % check_stalling_each)) {
            addToResQueue(convergence_check, convergence_check_old);
            convergence_check_old = convergence_check;

            stalled = checkResQueue();
            if (stalled) {
                if (ifprint)
                    if (myRank == 0)
                        std::cout << "Convergence stalling detected..." << std::endl;
                break;
            }
        }

        ++k;
    }
    if (ifprint && ((k - 1) % print_each))
        if (myRank == 0)
            std::cout << k - 1 << '\t' << convergence_check << std::endl;
    iterations_num = k;
    residual_norm = convergence_check;
}

template<class Preco, class MatrixType, class VectorType>
void CG::solve(
                Preco &precond,
                MatrixType &Matrix,
                VectorType &x,
                VectorType &b,
                VectorType &x0) {

    int k = 0;                              // iteration
    double alpha = 0.;
    double beta = 0.;
    double temp = 0.;
    double delta[2] = {0.};                 // delta[0] - delta_old, delta[1] - delta_new
    double convergence_check = 0.;          // keeps new residual
    double convergence_check_old = 0.;      // keeps old residual and used only is stalling checker is switched on
    double normalizer = 1.;                 // To normalize residual norm

    /*
     * MPI communicators
     */
    const int myRank = x.Comm().MyPID ();
    const Epetra_BlockMap _Map = x.Map();
    int size = _Map.NumMyElements();

    /*
     * First check if preconditioner has been built. If not - through a warning
     * and call for the unpreconditioned method
     */
    if (!precond.IsBuilt()) {
        if (myRank == 0) {
            std::cerr
                << "Warning! Preconditioner has not been built. Unpreconditioned method will be called instead..."
                << std::endl;
        }
        this->solve(Matrix, x, b, x0);
        return;
    }

    VectorType d(_Map);
    VectorType r(_Map);
    VectorType z(_Map);

    VectorType tmp(_Map);

    //! (1)	\f$ r_0 = b - A x_0 \f$
    Matrix.Multiply(false, x0, tmp);
    r = b;
    r.Update(-1., tmp, 1.);
    d = r;

    //! Apply preconditioner \f$ M z = r_0 \f$
    precond.solve(Matrix, z, r, false);
    //! Set d_0 = z_0
    d = z;

    //! Set \f$ \delta_{new} = \alpha = ||r||_2^2 \f$
    r.Dot(r, &convergence_check);
    delta[1] = alpha = convergence_check;

    /*!
     * Prepare stop criteria
     */
    switch (stop_criteria) {
        case RNORM:
            normalizer = 1.;
            break;
        case RRNORM:
        case INTERN:
            normalizer = sqrt(convergence_check);
            break;
        case RBNORM:
            b.Norm2(&normalizer);
//            normalizer = wrp_mpi::Norm2(b.Values(), size, communicator);
            break;
        case RWNORM:
            normalizer = weight;
            break;
        default:
            normalizer = 1.;
            break;
    }

    convergence_check /= normalizer;

    /*
     * Check residual. Stop if initial guess satisfies convergence criteria.
     */
    if (convergence_check < eps) {
        if (ifprint && !(k % print_each)) {
            if (myRank == 0)
                std::cout << k << '\t' << convergence_check << std::endl;
        }
        iterations_num = k;
        residual_norm = convergence_check;
        return;
    }

    if (ifprint) {
        if (myRank == 0)
            std::cout << k << '\t' << convergence_check / normalizer << std::endl;
    }
    ++k;

    //! Start iterative loop
    while (1) {
        if (k > MaxIter) {
            break;
        }

        //! (2)	\f$ \alpha = <r, r> / <d, A d_{old}> \f$
        Matrix.Multiply(false, d, tmp);
        d.Dot(tmp, &temp);
//        temp = wrp_mpi::Dot(d.Values(), tmp.Values(), size, communicator);
        alpha /= temp;		            // Possible break down if temp == 0.0

        //! (3) \f$ x_{new} = x_{old} + \alpha d_{old} \f$
        x.Update(alpha, d, 1.);

        //! (4) \f$ r_{new} = r_{old} - \alpha A d_{old} \f$
        r.Update(-alpha, tmp, 1.);

        //! Apply preconditioner \f$ M z = r_{new} \f$
        precond.solve(Matrix, z, r, false);

        //! (5)	\f$ \delta_{new} = <r_{new}, r_{new}> \f$
        r.Dot(z, &alpha);               // Possible break down if alpha == 0.0
//        alpha = wrp_mpi::Dot(r.Values(), z.Values(), size, communicator);

        //! (6)	\f$ \delta_{old} = <r_{old}, r_{old}> \f$
        delta[0] = delta[1];
        delta[1] = alpha;

        // means convergence_check = r.Norm2();
        if (stop_criteria == INTERN)
            convergence_check = delta[1] / delta[0];
        else {
            r.Norm2(&convergence_check);
//            convergence_check = wrp_mpi::Norm2(r.Values(), size, communicator);
//            convergence_check /= normalizer;
        }

        /*!
         * Check convergence
         */
        if (ifprint && !(k % print_each)) {
            if (myRank == 0)
                std::cout << k << '\t' << convergence_check << std::endl;
        }

        if (convergence_check <= eps) {
            break;
        }

        //! (7)	\f$ \beta = <r_{new}, r_{new}> / <r_{old}, r_{old}> \f$
        beta = delta[1] / delta[0];

        //! (8)	\f$ d_{new} = r_{new} + \beta d_{old} \f$
        d.Update(1., z, beta);

        /*
         * Check for convergence stalling
         */
        if (check_stalling && !(k % check_stalling_each)) {
            addToResQueue(convergence_check, convergence_check_old);
            convergence_check_old = convergence_check;

            stalled = checkResQueue();
            if (stalled) {
                if (ifprint) {
                    if (myRank == 0)
                        std::cout << "Convergence stalling detected..." << std::endl;
                }
                break;
            }
        }

        ++k;
    }
    if (ifprint && ((k - 1) % print_each)) {
        if (myRank == 0)
            std::cout << k - 1 << '\t' << convergence_check << std::endl;
    }
    iterations_num = k;
    residual_norm = convergence_check;
}
}

#endif /* CG_H_ */
