/*
 * BiCG.h
 *
 *  Created on: Apr 4, 2016
 *      Author: maxim
 */

/*!
 * \file BiCG.h
 *
 * \brief File contains class for BiConjugate Gradient method
 */

#ifndef KRYLOV_BICG_H_
#define KRYLOV_BICG_H_

#include "../../SSE/Wrappers.h"
#include "../SolversBase.h"

namespace slv {

/*!
 * \ingroup KrylovSolvers
 * \class BiCG
 * \brief BiConjugate Gradient method
 *
 * BiConjugate Gradient method works works with both symmetric and non-symmetric matrices.
 *
 * \note This method is unstable in some cases, especially without preconditioning
 * (see preconditioned variant).
 *
 * Class contains two overloaded methods for solving original and preconditioned systems for both
 * dense and sparse incoming matrices
 *
 * The typical usage example is as follows:
 *
 * \code
 * int n = 100;
 * wrp::VectorD x(n);
 * wrp::VectorD b(n);
 * SparseMatrix<double, RowMajor> A(n, n);
 *
 * // Fill matrix and right hand side
 * ...
 *
 * BiCG bicg;
 *
 * // Build preconditioner if necessary...
 *
 * // Set type of stopping criteria
 * bicg.SetStopCriteria(RBNORM);
 *
 * // Set tolerance
 * bicg.ChangeAcc(1e-8);
 *
 * // Set maximum number of iterations
 * bicg.ChangeMaxIter(100);
 *
 * // Solve the system
 * bicg.solve(A, x, b, x); // Or, in case of preconditioning: bicg.solve(Preco, A, x, b, x);
 *
 * // Print out number of iterations and residual
 * std::cout << "Iterations: " << bicg.Iterations() << std::endl;
 * std::cout << "Residual: " << bicg.Residual() << std::endl;
 * \endcode
 */
class BiCG: public Base {

public:

    BiCG() :
        Base() {
    }
    ;

    ~BiCG() {
    }
    ;

    /*!
     * \brief BiConjugate Gradient method
     *
     * \par For developers
     * Method contains:
     *  - 3 Matrix-Vector Multiplication \n
     *      (2 in loop, 1 out of loop)
     *  - 4 Vector-Vector Multiplication \n
     *      (4 in loop)
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
     * \brief Preconditioned BiConjugate Gradient method
     *
     * Preconditioned BiConjugate Gradients method uses right-preconditioned form, i.e.
     *                  \f{eqnarray*}{ A M^{-1} y =  b, \\ x = M^{-1} y \f}
     * where \f$ M^{-1}\f$ is a preconditioning matrix. Matrix \f$ M \f$ can be obtained via different ways, e.g.
     * Incomplete LU factorization, Incomplete Cholesky factorization etc.
     *
     * \note Method asks for two preconditioning matrices original and transposed. Thus if common preconditioners
     * are used amount of work is doubled
     *
     * \note In case of AMG as a preconditioner method uses one set of coarse grids for both non-transposed and
     * transposed matrices \f$ M \f$
     *
     * \par For developers
     * Method contains:
     *  - 3 Matrix-Vector Multiplication \n
     *      (2 in loop, 1 out of loop)
     *  - 2 Vector-Vector Multiplication \n
     *      (2 in loop)
     *  - 1 Matrix transpose
     *      (1 out of loop)
     *
     * \par
     * If not AMG is used as a preconditioner, then CG_CSR is used as a solver for preconditional
     * matrix.
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
void BiCG::solve(MatrixType &Matrix,							// Incoming CSR matrix
    VectorType &x,									// Vector of unknowns
    VectorType &b,									// Vector of right hand side
    VectorType &x0)									// Vector of initial guess
    {
    int k = 0;                              // iteration number
    double alpha = 0.;                 // part of method
    long double beta = 0.;                  // part of method
    double temp = 0.;                  // helper
    long double delta[2] = {0.};            // part of the method
    double convergence_check = 0.;          // keeps new residual
    double convergence_check_old = 0.;      // keeps old residual and used only is stalling checker is switched on
    double normalizer = 1.;                 // To normalize residual norm

    /*
     * MPI communicators
     */
    const int myRank = x.Comm().MyPID ();
    const Epetra_BlockMap _Map = x.Map();
    int size = _Map.NumMyElements();        // system size

    VectorType r(_Map);
    VectorType r_hat(_Map);
    VectorType p(_Map);
    VectorType p_hat(_Map);

    VectorType tmp(_Map);

    //! (1)	\f$ p_0 = r_0 = b - A x_0 \f$
    Matrix.Multiply(false, x0, tmp);
    r = b;
    r.Update(-1., tmp, 1.);

    //! (1')	\f$ \hat{p}_0 = \hat{r}_0 = b - A^T x_0 \f$
    Matrix.Multiply(true, x0, tmp);
    r_hat = b;
    r_hat.Update(-1., tmp, 1.);

    p = r; p_hat = r_hat;

    //! Set \f$ \delta_{new} = \alpha = ||r||_2^2 \f$
    r_hat.Dot(r, &convergence_check);
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
            normalizer = sqrt(delta[1]);
            break;
        case RBNORM:
            b.Norm2(&normalizer);
            break;
        case RWNORM:
            normalizer = weight;
            break;
        default:
            normalizer = 1.;
            break;
    }

    convergence_check = sqrt(alpha) / normalizer;

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

        //! (2)	\f$ \alpha = <r, \hat{r}> / <\hat{p}, A p> \f$
        Matrix.Multiply(false, p, tmp);
        p_hat.Dot(tmp, &temp);                      // Note, this value can be very-very-very!!! small, e.g. 1e-33 O_o

        if (fabs(temp) <= LDBL_EPSILON) {
            if (myRank == 0)
                std::cout << "BiCG has been interrupted..." << std::endl;
            break;
        }
        alpha /= temp;

        //! (3)	\f$ x_{new} = x_{old} + \alpha p \f$
        x.Update(alpha, p, 1.);

        //! (4)	\f$ r_{new} = r_{old} - \alpha A p \f$
        r.Update(-alpha, tmp, 1.);
//        x.Update2(p, 1., static_cast<double>(alpha), r, tmp, 1., static_cast<double>(-alpha));

        //! (5)	\f$ \hat{r}_{new} = \hat{r}_{old} - \alpha A^T \hat{p} \f$
        Matrix.Multiply(true, p_hat, tmp);
        r_hat.Update(-alpha, tmp, 1.);
//        r_hat.Update(tmp, 1., static_cast<double>(-alpha));

        //! (6)	\f$ \beta = <r_{new}, \hat{r}_{new}> / <r_{old}, \hat{r}_{old}> \f$
//        alpha = r.Dot(r_hat);
        r_hat.Dot(r, &alpha);
        delta[0] = delta[1];
        delta[1] = alpha;

        if (fabs(delta[0]) <= LDBL_EPSILON) {
            if (myRank == 0)
                std::cout << "BiCG has been interrupted..." << std::endl;
            break;
        }

        beta = static_cast<double>(delta[1] / delta[0]);

        //! (7)	\f$ p_{new} = r_{new} + \beta p_{old} \f$
//        p.Update(r, beta, 1.);
        p.Update(1., r, beta);

        //! (8)	\f$ \hat{p}_{new} = \hat{r}_{new} + \beta \hat{p}_{old} \f$
//        p_hat.Update(r_hat, beta, 1.);
        p_hat.Update(1., r_hat, beta);

//        convergence_check = r.Norm2() / normalizer;
        r.Norm2(&convergence_check);
        convergence_check /= normalizer;

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
    // ================================================== //
}

template<class Preco, class MatrixType, class VectorType>
void BiCG::solve(Preco &precond,							// Preconditioner class
    MatrixType &Matrix,							// Incoming CSR matrix
    VectorType &x,									// Vector of unknowns
    VectorType &b,									// Vector of right hand side
    VectorType &x0)								// Vector of initial guess
    {

    int k = 0;                              // iteration number
    long double alpha = 0.0L;               // part of the method
    double beta = 0.;                      // part of the method
    long double delta[2] = {0.0L};			// part of the method
    long double delta_0 = 0.0L;				// part of the method
    long double temp = 0.0L;               // helper
    double convergence_check = 0.;			// keeps new residual
    double convergence_check_old = 0.;// keeps old residual and used only is stalling checker is switched on
    double normalizer = 1.;					// To normalize residual norm

    /*
     * MPI communicators
     */
    const int myRank = x.Comm().MyPID ();
    const Epetra_BlockMap _Map = x.Map();
    int size = _Map.NumMyElements();        // system size

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
        solve(Matrix, x, b, x0);
        return;
    }

    VectorType r(_Map);
    VectorType r_hat(_Map);
    VectorType p(_Map);
    VectorType p_hat(_Map);
    VectorType z(_Map);
    VectorType z_hat(_Map);

    VectorType tmp(_Map);

    //! (1)	\f$ r_0 = b - A x_0 \f$
    Matrix.Multiply(false, x0, tmp);
    r = b;
    r.Update(-1., tmp, 1.);

    //! (2)	\f$ \hat{r}_0 = b - A^T x_0 \f$
    Matrix.Multiply(true, x0, tmp);
    r_hat = b;
    r_hat.Update(-1., tmp, 1.);

    r_hat.Dot(r, &convergence_check);

    /*!
     * Prepare stop criteria
     */
    switch (stop_criteria) {
        case RNORM:
            normalizer = 1.;
            break;
        case RRNORM:
        case INTERN:
            normalizer = convergence_check;
            break;
        case RBNORM:
            b.Norm2(&normalizer);
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
            std::cout << k << '\t' << convergence_check << std::endl;
    }
    ++k;

    //! Start iterative loop
    while (1) {
        if (k > MaxIter) {
            break;
        }

        /*!
         *  (3) Apply preconditioners \f$ M z = r \f$, \f$ M^T \hat{z} = \hat{r} \f$
         */
        precond.solve(Matrix, z, r, true);
        precond.solve(Matrix, z_hat, r_hat, true);

        z.Dot(r_hat, &alpha);

        if (alpha == 0) {
            if (myRank == 0) {
                std::cout << "PBiCG Method failed..." << std::endl;
            }
            break;
        }

        //! If iteration is not the first one do (4) - (6)
        if (k > 1) {
            delta[0] = delta[1];
            delta[1] = alpha;

            //! (4)	\f$ \beta = <r_{new}, r_{new}> / <r_{old}, r_{old}> \f$
            beta = static_cast<double>(delta[1] / delta[0]);

            //! (5)	\f$ p_{new} = r_{new} + \beta p_{old} \f$
//            p.Update(z, beta, 1.);
            p.Update(1., z, beta);

            //! (6)	\f$ \hat{p}_{new} = \hat{r}_{new} + \beta \hat{p}_{old} \f$
//            p_hat.Update(z_hat, beta, 1.);
            p_hat.Update(1., z_hat, beta);
        }
        //! Else set \f$ p = z \f$, \f$ \hat{p} = \hat{z} \f$, \f$ \delta_{new} = \delta_0 = <z, \hat{r}> \f$
        else {
            delta[1] = alpha;
            delta_0 = delta[1];
            p = z;
            p_hat = z_hat;
        }

        //! (7)	\f$ \alpha = <z, \hat{r}> / <\hat{p}, A p> \f$
        Matrix.Multiply(false, p, tmp);
//        temp = p_hat.Dot(tmp);
        p_hat.Dot(tmp, &temp);

        if (fabs(temp) <= LDBL_EPSILON) {
            if (myRank == 0) {
                std::cout << "PBiCG has been interrupted..." << std::endl;
            }
            break;
        }

        alpha /= temp;

        //! (8)	\f$ x_{new} = x_{old} + \alpha p \f$
        x.Update(alpha, p, 1.);

        //! (9)	\f$ r_{new} = r_{old} - \alpha A p \f$
        r.Update(-alpha, tmp, 1.);
//        x.Update2(p, 1., static_cast<double>(alpha), r, tmp, 1., static_cast<double>(-alpha));

        //! (10)	\f$ \hat{r} = \hat{r}_{old} - \alpha A^T \hat{p} \f$
        Matrix.Multiply(true, p_hat, tmp);
//        r_hat.Update(tmp, 1., static_cast<double>(-alpha));
        r_hat.Update(-alpha, tmp, 1.);

        if (stop_criteria != INTERN)
            convergence_check = r.Norm2() / normalizer;
        else
            convergence_check = fabs(static_cast<double>(delta[1] / delta_0));
        ;

        /*!
         * Check convergence
         */
        if (ifprint && !(k % print_each)) {
            if (myRank == 0) {
                std::cout << k << '\t' << convergence_check << std::endl;
            }
        }

        if (convergence_check <= eps) {
            break;
        }

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

#endif /* KRYLOV_BICG_H_ */
