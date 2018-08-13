/*
 * BiCGSTAB2.h
 *
 *  Created on: Apr 4, 2016
 *      Author: maxim
 */

/*!
 * \file BiCGSTAB2.h
 *
 * \brief File contains class for BiConjugate Gradient Stabilized (2) method
 */

#ifndef KRYLOV_BICGSTAB2_H_
#define KRYLOV_BICGSTAB2_H_

#include "../../SSE/Wrappers.h"
#include "../SolversBase.h"

namespace slv_mpi {

/*!
 * \ingroup KrylovSolvers
 * \class BiCGSTAB2
 * \brief BiConjugate Gradient Stabilized (2) method
 *
 * BiConjugate Gradients Stabilized (2) method works with both symmetric and non-symmetric matrices.
 * This method is stable and is an improved version of BiCGSTAB.
 *
 * Was proposed by Sleijpen G, Fokkema D., "BICGStab(L) for linear equations involving unsymmetric matrices with complex spectrum",
 * 1993, ETNA, Vol. 1, pp. 11-32.
 *
 * The algorithm has been taken from H. van der Vorst, "Iterative Krylov Methods for Large Linear Systems",
 * 2003, Cambridge Monographs on Applied and Computational Mathematics, vol. 13.
 *
 * \note In general it is just an unrolled algorithm from Sleijpen and Fokkema
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
 * BiCGSTAB2 bicgstab;
 *
 * // Build preconditioner if necessary...
 *
 * // Set type of stopping criteria
 * bicgstab.SetStopCriteria(RBNORM);
 *
 * // Set tolerance
 * bicgstab.ChangeAcc(1e-8);
 *
 * // Set maximum number of iterations
 * bicgstab.ChangeMaxIter(100);
 *
 * // Solve the system
 * bicgstab.solve(A, x, b, x); // Or, in case of preconditioning: bicgstab.solve(Preco, A, x, b, x);
 *
 * // Print out number of iterations and residual
 * std::cout << "Iterations: " << bicgstab.Iterations() << std::endl;
 * std::cout << "Residual: " << bicgstab.Residual() << std::endl;
 * \endcode
 */
class BiCGSTAB2: public Base {

public:

    BiCGSTAB2(MPI_Comm _comm) :
        Base(_comm) {
    }
    ;

    ~BiCGSTAB2() {
    }
    ;

    /*!
     * \brief Conjugate Gradient method
     *
     * \par For developers
     * Method contains:
     *  - 14 vector updates
     *    - 4 Matrix-Vector Multiplication \n
     *      (3 in loop, 1 out of loop)
     *    - 9 Vector-Vector Multiplication \n
     *      (5 in loop, 1 out of loop)
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
     * \brief Preconditioned Conjugate Gradient method
     *
     * Preconditioned BiConjugate Gradient Stabilized (2) method uses right-preconditioned form, i.e.
     *                  \f{eqnarray*}{ A M^{-1} y =  b, \\ x = M^{-1} y \f}
     *
     * \par For developers
     * Method contains:
     *  - 14 vector updates
     *    - 4 Matrix-Vector Multiplication \n
     *      (3 in loop, 1 out of loop)
     *    - 9 Vector-Vector Multiplication \n
     *      (5 in loop, 1 out of loop)
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
void BiCGSTAB2::solve(
                    MatrixType &Matrix,
                    VectorType &x,
                    VectorType &b,
                    VectorType &x0) {

    int k = 0;                          // iteration number
    double alpha = 0.;                  // part of the method
    double rho[2] = {0.};               // part of the method
    double gamma = 0.;                  // part of the method
    double beta = 0.;                   // part of the method
    long double omega_1 = 0.0L;         // part of the method, stored as a long to prevent overflow
    long double omega_2 = 0.0L;         // part of the method, stored as a long to prevent overflow
    long double mu = 0.0L;              // part of the method, stored as a long to prevent overflow
    long double nu = 0.0L;              // part of the method, stored as a long to prevent overflow
    long double tau = 0.0L;             // part of the method, stored as a long to prevent overflow

    double convergence_check = 0.;      // keeps new residual
    double convergence_check_old = 0.;  // keeps old residual and used only is stalling checker is switched on
    double normalizer = 1.;

    /*
     * MPI communicators
     */
    const int myRank = x.Comm().MyPID ();
    const Epetra_BlockMap _Map = x.Map();
    int size = _Map.NumMyElements();    // local system size

    VectorType r(_Map);
    VectorType r_hat_0(_Map);
    VectorType u(_Map);
    VectorType v(_Map);
    VectorType s(_Map);
    VectorType w(_Map);
    VectorType t(_Map);

    //! (0) \f$ r = \hat{r}_0 = b - A * x_0\f$
    Matrix.Multiply(false, x0, v);
    r = b;
    r.Update(-1., v, 1.);
    r_hat_0 = r;                            // Actually r_hat_0 is an arbitrary vector

    //! (1) \f$ u = 0 \f$, \f$ \alpha = \rho[0] = \omega_2 = 1\f$
    wrp_mpi::Zero(u.Values(), size);
    alpha   = 0.;
    rho[0]  = 1.;
    omega_2 = 1.;

    /*!
     * Prepare stop criteria
     */
    switch (stop_criteria) {
        case RNORM:
            normalizer = 1.;
            break;
        case RRNORM: case INTERN:
            normalizer = wrp_mpi::Norm2(r.Values(), size, communicator);
            break;
        case RBNORM:
            normalizer = wrp_mpi::Norm2(b.Values(), size, communicator);
            break;
        case RWNORM:
            normalizer =  weight;
            break;
        default:
            normalizer = 1.;
            break;
    }

    /*
     * Check residual. Stop if initial guess satisfies convergence criteria.
     */
    convergence_check = wrp_mpi::Norm2(r.Values(), size, communicator) / normalizer;
    if (convergence_check < eps) {
        if ( ifprint && !(k % print_each) ) {
            if (myRank == 0)
                std::cout << k << '\t' << convergence_check << std::endl;
        }
        iterations_num = k;
        residual_norm = convergence_check;
        return;
    }

    if ( ifprint )
        std::cout << k << '\t' << convergence_check/normalizer << std::endl;
    ++k;

    //! Start iterative loop
    while(1) {

        if(k > MaxIter) {
            break;
        }

        //! (2) \f$ \rho[0] = - \omega_2 \rho[0] \ \f$
        rho[0] = - omega_2 * rho[0];
        if (rho[0] == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (rho[0] == 0.0)" << std::endl;
            break;
        }
        /*!
         * Even Bi-CG step
         */
        //! (3) \f$ \rho[1] = <\hat{r}_0, r>\f$, \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
//          rho[1] = r_hat_0.dot(r);
        rho[1] = wrp_mpi::Dot(r_hat_0.Values(), r.Values(), size, communicator);
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];
        if (rho[0] == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (rho[0] == 0.0)" << std::endl;
            break;
        }

        //! (4) \f$ u = r - \beta u \f$
//          u = r - beta * u;
        wrp_mpi::Update(u.Values(), r.Values(), -beta, 1., size);

        //! (5) \f$ v = A u \f$
//          v.noalias() = Matrix * u;
        Matrix.Multiply(false, u, v);

        //! (6) \f$ \gamma = <v, \hat{r}_0> \f$, \f$ \alpha = \rho[0] / \gamma \f$
//          gamma = v.dot(r_hat_0);
        gamma = wrp_mpi::Dot(v.Values(), r_hat_0.Values(), size, communicator);

        // Check for breakdown (probably may occur)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (7) \f$ r = r - \alpha v \f$
//          r -= alpha * v;
        wrp_mpi::Update(r.Values(), v.Values(), 1., -alpha, size);

        //! (8) \f$ s = A r \f$
//          s.noalias()  = Matrix * r;
        Matrix.Multiply(false, r, s);

        //! (9) \f$ x = x + \alpha u \f$
//          x += alpha * u;
        wrp_mpi::Update(x.Values(), u.Values(), 1., alpha, size);
        /*!
         * Odd Bi-CG step
         */
        //! (10) \f$ \rho[1] = <\hat{r}_0, s>\f$, \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
//          rho[1] = r_hat_0.dot(s);
        rho[1] = wrp_mpi::Dot(r_hat_0.Values(), s.Values(), size, communicator);
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];

        //! (11) \f$ v = s - \beta v \f$
//          v = s - beta * v;
        wrp_mpi::Update(v.Values(), s.Values(), -beta, 1., size);
//          wrp_mpi::Update2(x, u, 1., alpha, v, s, -beta, 1., size);

        //! (12) \f$ w = A v \f$
//          w.noalias() = Matrix * v;
        Matrix.Multiply(false, v, w);

        //! (13) \f$ \gamma = <w, \hat{r}_0> \f$, \f$ \alpha = \rho[0] / \gamma \f$
//          gamma = w.dot(r_hat_0);
        gamma = wrp_mpi::Dot(w.Values(), r_hat_0.Values(), size, communicator);

        // Check for breakdown (may occur if matrix is diagonal)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (14) \f$ u = r - \beta u \f$
//          u = r - beta * u;
        wrp_mpi::Update(u.Values(), r.Values(), -beta, 1., size);

        //! (15) \f$ r = r - \alpha v \f$
//          r -= alpha * v;
        wrp_mpi::Update(r.Values(), v.Values(), 1., -alpha, size);

        //! (16) \f$ s = s - \alpha w\f$
//          s -= alpha * w;
        wrp_mpi::Update(s.Values(), w.Values(), 1., -alpha, size);
//          wrp_mpi::Update2(r, v, 1., -alpha, s, w, 1., -alpha, size);

        //! (17) \f$ t = A s\f$
//          t.noalias() = Matrix * s;
        Matrix.Multiply(false, s, t);

        /*!
         * GCR(2)-part
         */
        //! (18) \f$ \omega_1 = <r, s> \f$, \f$ \mu = <s, s> \f$, \f$ \nu = <s, t> \f$, \f$ \tau = <t, t> \f$
//          omega_1 = r.dot(s);
//          mu = s.squaredNorm();
//          nu = s.dot(t);
//          tau = t.squaredNorm();
        omega_1 = wrp_mpi::Dot(r.Values(), s.Values(), size, communicator);
        mu = wrp_mpi::Dot(s.Values(), s.Values(), size, communicator);
        nu = wrp_mpi::Dot(s.Values(), t.Values(), size, communicator);
        tau = wrp_mpi::Dot(t.Values(), t.Values(), size, communicator);

        if (mu == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (mu == 0.0)" << std::endl;
            break;
        }

        //! (19) \f$ \omega_2 = <r, t> \f$
//          omega_2 = r.dot(t);
        omega_2 = wrp_mpi::Dot(r.Values(), t.Values(), size, communicator);

        //! (20) \f$ \tau = \tau - \nu^2 / \mu \f$
        tau -= nu * nu / mu;

        if (tau == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (tau == 0.0)" << std::endl;
            break;
        }

        //! (21) \f$ \omega_2 = (\omega_2 - \nu \omega_1 / \mu) / \tau \f$
        omega_2 = (omega_2 - (nu * omega_1) / mu) / tau;

        //! (22) \f$ \omega_1 = (\omega_1 - \nu \omega_2) / \mu \f$
        omega_1 = (omega_1 - nu * omega_2) / mu;

        //! (23) \f$ x = x + \omega_1 r + \omega_2 s + \alpha u \f$
//          x = x + omega_1 * r + omega_2 * s + alpha * u;
        wrp_mpi::Update(x.Values(), r.Values(), s.Values(), u.Values(), 1.,
                static_cast<double>(omega_1), static_cast<double>(omega_2), alpha, size);

        //! (24) \f$ r = r - \omega_1 s - \omega_2 t \f$
//          r = r - omega_1 * s - omega_2 * t;
        wrp_mpi::Update(r.Values(), s.Values(), t.Values(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        /*!
         * Check convergence
         */
//          convergence_check = r.norm() / normalizer;
        convergence_check = wrp_mpi::Norm2(r.Values(), size, communicator) / normalizer;

        if ( ifprint && !(k % print_each) ) {
            if (myRank == 0)
                std::cout << k << '\t' << convergence_check << std::endl;
        }

        if( convergence_check <= eps && k > 1) {
            break;
        }

        /*
         * Check for convergence stalling
         */
        if (check_stalling && !(k % check_stalling_each) ) {
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

        //! (25) \f$ u = u - \omega_1 v - \omega_2 w \f$
//          u = u - omega_1 * v - omega_2 * w;
        wrp_mpi::Update(u.Values(), v.Values(), w.Values(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        ++k;
    }
    if ( ifprint && ((k-1) % print_each) ) {
        if (myRank == 0)
            std::cout << k-1 << '\t' << convergence_check << std::endl;
    }
    iterations_num = k;
    residual_norm = convergence_check;
//    MPI_Barrier(communicator);
}

template<class Preco, class MatrixType, class VectorType>
void BiCGSTAB2::solve(
                    Preco &precond,
                    MatrixType &Matrix,
                    VectorType &x,
                    VectorType &b,
                    VectorType &x0) {

    int k = 0;                          // iteration number
    double alpha = 0.;                  // part of the method
    double rho[2] = {0.};               // part of the method
    double gamma = 0.;                  // part of the method
    double beta = 0.;                   // part of the method
    long double omega_1 = 0.0L;         // part of the method, stored as a long to prevent overflow
    long double omega_2 = 0.0L;         // part of the method, stored as a long to prevent overflow
    long double mu = 0.0L;              // part of the method, stored as a long to prevent overflow
    long double nu = 0.0L;              // part of the method, stored as a long to prevent overflow
    long double tau = 0.0L;             // part of the method, stored as a long to prevent overflow

    double r_norm_0 = 0.;               // Preconditioned norm
    double convergence_check = 0.;      // keeps new residual
    double convergence_check_old = 0.;  // keeps old residual and used only is stalling checker is switched on
    double normalizer = 1.;             // normalizer for the residual

    /*
     * MPI communicators
     */
    const int myRank = x.Comm().MyPID ();
    const Epetra_BlockMap _Map = x.Map();
    int size = _Map.NumMyElements();    // local system size

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
    VectorType r_hat_0(_Map);
    VectorType u(_Map);
    VectorType v(_Map);
    VectorType s(_Map);
    VectorType w(_Map);
    VectorType t(_Map);
    VectorType u_hat(_Map);
    VectorType r_hat(_Map);
    VectorType v_hat(_Map);
    VectorType s_hat(_Map);
    VectorType tmp(_Map);

    // Right preconditioner
    precond.solve(Matrix, tmp, x0, false);
    x0 = tmp;

    //! (0) \f$ r = \hat{r}_0 = b - A * x_0\f$
//    r = (b - Matrix * x0);
    Matrix.Multiply(false, x0, v);
    r = b;
    r.Update(-1., v, 1.);
    wrp_mpi::Copy(r_hat_0.Values(), r.Values(), size);            // Actually r_hat_0 is an arbitrary vector

    //! (1) \f$ u = 0 \f$, \f$ w = 0 \f$, \f$ v = 0 \f$, \f$ \alpha = \rho[0] = \omega_1 = \omega_2 = 1\f$
    wrp_mpi::Assign(u.Values(), 0.0, size);
    wrp_mpi::Assign(w.Values(), 0.0, size);
    wrp_mpi::Assign(v.Values(), 0.0, size);
    alpha = rho[0] = omega_1 = omega_2 = 1.;

    //! (2) Solve \f$ M y = r \f$, set \f$ r = y \f$
    // Case of left preconditioner
//      precond.solve(Matrix, tmp, r, false);
//      r = tmp;

    r_norm_0 = wrp_mpi::Norm2(r.Values(), size, communicator);

    /*!
     * Prepare stop criteria
     */
    switch (stop_criteria) {
        case RNORM:
            normalizer = 1.;
            break;
        case RRNORM: case INTERN:
            normalizer = r_norm_0;
            break;
        case RBNORM:
            normalizer = wrp_mpi::Norm2(b.Values(), size, communicator);
            break;
        case RWNORM:
            normalizer =  weight;
            break;
        default:
            normalizer = 1.;
            break;
    }

    /*
     * Check residual. Stop if initial guess satisfies convergence criteria.
     */
    convergence_check = r_norm_0 / normalizer;
    if (convergence_check < eps) {
        if ( ifprint && !(k % print_each) ) {
            if (myRank == 0)
                std::cout << k << '\t' << convergence_check << std::endl;
        }
        iterations_num = k;
        residual_norm = convergence_check;
        return;
    }

    if ( ifprint ) {
        if (myRank == 0)
            std::cout << k << '\t' << convergence_check << std::endl;
    }
    ++k;

    //! Start iterative loop
    while(1) {

        if(k > MaxIter) {
            break;
        }

        //! (3) \f$ \rho[0] = - \omega_2 \rho[0] \ \f$
        rho[0] = - omega_2 * rho[0];
        if (rho[0] == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (rho[0] == 0.0)" << std::endl;
            break;
        }

        /*!
         * Even Bi-CG step
         */
        //! (4) \f$ \rho[1] = <\hat{r}_0, r>\f$, \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
        rho[1] = wrp_mpi::Dot(r_hat_0.Values(), r.Values(), size, communicator);
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];
        if (rho[0] == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (rho[0] == 0.0)" << std::endl;
            break;
        }

        //! (5) \f$ u = r - \beta u \f$
        wrp_mpi::Update(u.Values(), r.Values(), -beta, 1., size);

        //! (6) \f$ v = A M^{-1} u \f$
        precond.solve(Matrix, tmp, u, false);
        Matrix.Multiply(false, tmp, v);

        // Case of left preconditioner
//          //! (6) \f$ v = M^{-1} A u \f$
//          tmp = Matrix * u;
//          precond.solve(Matrix, v, tmp, false);

        //! (7) \f$ \gamma = <v, \hat{r}_0> \f$, \f$ \alpha = \rho[0] / \gamma \f$
        gamma = wrp_mpi::Dot(v.Values(), r_hat_0.Values(), size, communicator);

        // Check for breakdown (probably may occur)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (gamma == 0.0)" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (8) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r.Values(), v.Values(), 1., -alpha, size);

        //! (9) \f$ s = A M^{-1} r \f$
        precond.solve(Matrix, tmp, r, false);
        Matrix.Multiply(false, tmp, s);

        // Case of left preconditioner
//          //! (9) \f$ s = M^{-1} A r \f$
//          tmp  = Matrix * r;
//          precond.solve(Matrix, s, tmp, false);

        //! (10) \f$ x = x + \alpha u \f$
        wrp_mpi::Update(x.Values(), u.Values(), 1., alpha, size);

        /*!
         * Odd Bi-CG step
         */
        //! (11) \f$ \rho[1] = <\hat{r}_0, s>\f$, \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
        rho[1] = wrp_mpi::Dot(r_hat_0.Values(), s.Values(), size, communicator);
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];

        //! (12) \f$ v = s - \beta v \f$
//          v = s - beta * v;
        wrp_mpi::Update(v.Values(), s.Values(), -beta, 1., size);

        //! (13) \f$ w = A M^{-1} v \f$
        precond.solve(Matrix, tmp, v, false);
        Matrix.Multiply(false, tmp, w);

        // Case of left preconditioner
//          //! (13) \f$ w = M^{-1} A v \f$
//          tmp  = Matrix * v;
//          precond.solve(Matrix, w, tmp, false);

        //! (14) \f$ \gamma = <w, \hat{r}_0> \f$, \f$ \alpha = \rho[0] / \gamma \f$
        gamma = wrp_mpi::Dot(w.Values(), r_hat_0.Values(), size, communicator);

        // Check for breakdown (probably may occur)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (gamma == 0.0)" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (15) \f$ u = r - \beta u \f$
        wrp_mpi::Update(u.Values(), r.Values(), -beta, 1., size);

        //! (16) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r.Values(), v.Values(), 1., -alpha, size);

        //! (17) \f$ s = s - \alpha w\f$
        wrp_mpi::Update(s.Values(), w.Values(), 1., -alpha, size);

        //! (18) \f$ t = A M^{-1} s\f$
        precond.solve(Matrix, tmp, s, false);
        Matrix.Multiply(false, tmp, t);

        // Case of left preconditioner
//          //! (18) \f$ t = M^{-1} A s\f$
//          tmp  = Matrix * s;
//          precond.solve(Matrix, t, tmp, false);

        /*
         * GCR(2)-part
         */
        //! (19) \f$ \omega_1 = <r, s> \f$, \f$ \mu = <s, s> \f$, \f$ \nu = <s, t> \f$, \f$ \tau = <t, t> \f$
          omega_1 = wrp_mpi::Dot(r.Values(), s.Values(), size, communicator);
          mu = wrp_mpi::Dot(s.Values(), s.Values(), size, communicator);
          nu = wrp_mpi::Dot(s.Values(), t.Values(), size, communicator);
          tau = wrp_mpi::Dot(t.Values(), t.Values(), size, communicator);

          //! (20) \f$ \omega_2 = <r, t> \f$
          omega_2 = wrp_mpi::Dot(r.Values(), t.Values(), size, communicator);
//        omega_1 = mu = nu = tau = omega_2 = 0.0;
//#ifdef BUMBLEBEE_USE_OPENMP
//#pragma omp parallel for reduction(+:omega_1, mu, nu, tau, omega_2) schedule(dynamic, 10000)
//#endif
//        for(int i = 0; i < size; ++i) {
//            omega_1 += r.data()[i] * s.data()[i];
//            mu += s.data()[i] * s.data()[i];
//            nu += s.data()[i] * t.data()[i];
//            tau += t.data()[i] * t.data()[i];
//            omega_2 += r.data()[i] * t.data()[i];
//        }

        if (mu == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (mu == 0.0)" << std::endl;
            break;
        }

        //! (21) \f$ \tau = \tau - \nu^2 / \mu \f$
        tau -= nu * nu / mu;

        if (tau == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (tau == 0.0)" << std::endl;
            break;
        }

        //! (22) \f$ \omega_2 = (\omega_2 - \nu \omega_1 / \mu) / \tau \f$
        omega_2 = (omega_2 - (nu * omega_1) / mu) / tau;

        //! (23) \f$ \omega_1 = (\omega_1 - \nu \omega_2) / \mu \f$
        omega_1 = (omega_1 - nu * omega_2) / mu;

        //! (24) \f$ x = x + \omega_1 r + \omega_2 s + \alpha u \f$
        wrp_mpi::Update(x.Values(), r.Values(), s.Values(), u.Values(), 1.,
                static_cast<double>(omega_1), static_cast<double>(omega_2), alpha, size);

        //! (25) \f$ r = r - \omega_1 s - \omega_2 t \f$
        wrp_mpi::Update(r.Values(), s.Values(), t.Values(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        /*!
         * Check convergence
         */
        // Case of left preconditioner
//          tmp = r;
//          precond.solve(Matrix, tmp, r, false);
//          convergence_check = tmp.norm() / normalizer;

        convergence_check = wrp_mpi::Norm2(r.Values(), size, communicator) / normalizer;
        if ( ifprint && !(k % print_each) ) {
            if (myRank == 0)
                std::cout << k << '\t' << convergence_check << std::endl;
        }

        if( convergence_check <= eps && k > 1) {
            break;
        }

        /*
         * Check for convergence stalling
         */
        if (check_stalling && !(k % check_stalling_each) ) {
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

        //! (25) \f$ u = u - \omega_1 v - \omega_2 w \f$
        wrp_mpi::Update(u.Values(), v.Values(), w.Values(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        ++k;
    }
    precond.solve(Matrix, tmp, x, false);
    wrp_mpi::Copy(x.Values(), tmp.Values(), size);

    if ( ifprint && ((k-1) % print_each) ) {
        if (myRank == 0)
            std::cout << k-1 << '\t' << convergence_check << std::endl;
    }
    iterations_num = k;
    residual_norm = convergence_check;
//    MPI_Barrier(communicator);
}
}

#endif /* KRYLOV_BICGSTAB2_H_ */
