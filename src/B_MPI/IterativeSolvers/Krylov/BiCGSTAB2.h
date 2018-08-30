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
#include <Epetra_Time.h>

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
    long double reduced_g[5];           // used for global communication
    long double reduced_l[5];           // used for global communication

    double convergence_check = 0.;      // keeps new residual
    double convergence_check_old = 0.;  // keeps old residual and used only is stalling checker is switched on
    double normalizer = 1.;

    /*
     * MPI communicators
     */
    const int myRank = x.getMap()->getComm()->getRank();
//    const Epetra_BlockMap _Map = x.getMap();
    int size = x.getMap()->getNodeNumElements();    // local system size

    VectorType r(x.getMap());
    VectorType r_hat_0(x.getMap());
    VectorType u(x.getMap());
    VectorType v(x.getMap());
    VectorType s(x.getMap());
    VectorType w(x.getMap());
    VectorType t(x.getMap());

    //! (0) \f$ r = \hat{r}_0 = b - A * x_0\f$
    Matrix.apply(x0, v);
    r = b;
    r.update(-1., v, 1.);
    r_hat_0 = r;                            // Actually r_hat_0 is an arbitrary vector

    //! (1) \f$ u = 0 \f$, \f$ \alpha = \rho[0] = \omega_2 = 1\f$
    wrp_mpi::Zero(u.getDataNonConst().get(), size);
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
            normalizer = wrp_mpi::Norm2(r.getDataNonConst().get(), size, communicator);
            break;
        case RBNORM:
            normalizer = wrp_mpi::Norm2(b.getDataNonConst().get(), size, communicator);
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
    convergence_check = wrp_mpi::Norm2(r.getDataNonConst().get(), size, communicator) / normalizer;
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
        std::cout << k << '\t' << convergence_check/normalizer << std::endl;
    }
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
        rho[1] = wrp_mpi::Dot(r_hat_0.getDataNonConst().get(), r.getDataNonConst().get(), size, communicator);
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];
        if (rho[0] == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (rho[0] == 0.0)" << std::endl;
            break;
        }

        //! (4) \f$ u = r - \beta u \f$
        wrp_mpi::Update(u.getDataNonConst().get(), r.getDataNonConst().get(), -beta, 1., size);

        //! (5) \f$ v = A u \f$
        Matrix.apply(u, v);

        //! (6) \f$ \gamma = <v, \hat{r}_0> \f$, \f$ \alpha = \rho[0] / \gamma \f$
        gamma = wrp_mpi::Dot(v.getDataNonConst().get(), r_hat_0.getDataNonConst().get(), size, communicator);

        // Check for breakdown (probably may occur)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (7) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r.getDataNonConst().get(), v.getDataNonConst().get(), 1., -alpha, size);

        //! (8) \f$ s = A r \f$
        Matrix.apply(r, s);

        //! (9) \f$ x = x + \alpha u \f$
        wrp_mpi::Update(x.getDataNonConst().get(), u.getDataNonConst().get(), 1., alpha, size);

        /*!
         * Odd Bi-CG step
         */
        //! (10) \f$ \rho[1] = <\hat{r}_0, s>\f$, \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
        rho[1] = wrp_mpi::Dot(r_hat_0.getDataNonConst().get(), s.getDataNonConst().get(), size, communicator);
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];

        //! (11) \f$ v = s - \beta v \f$
        wrp_mpi::Update(v.getDataNonConst().get(), s.getDataNonConst().get(), -beta, 1., size);
//          wrp_mpi::Update2(x, u, 1., alpha, v, s, -beta, 1., size);

        //! (12) \f$ w = A v \f$
        Matrix.apply(v, w);

        //! (13) \f$ \gamma = <w, \hat{r}_0> \f$, \f$ \alpha = \rho[0] / \gamma \f$
        gamma = wrp_mpi::Dot(w.getDataNonConst().get(), r_hat_0.getDataNonConst().get(), size, communicator);

        // Check for breakdown (may occur if matrix is diagonal)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (14) \f$ u = r - \beta u \f$
        wrp_mpi::Update(u.getDataNonConst().get(), r.getDataNonConst().get(), -beta, 1., size);

        //! (15) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r.getDataNonConst().get(), v.getDataNonConst().get(), 1., -alpha, size);

        //! (16) \f$ s = s - \alpha w\f$
        wrp_mpi::Update(s.getDataNonConst().get(), w.getDataNonConst().get(), 1., -alpha, size);

        //! (17) \f$ t = A s\f$
        Matrix.apply(s, t);

        /*!
         * GCR(2)-part
         */
        //! (18) \f$ \omega_1 = <r, s> \f$, \f$ \mu = <s, s> \f$, \f$ \nu = <s, t> \f$, \f$ \tau = <t, t> \f$
        //! (19) \f$ \omega_2 = <r, t> \f$
        reduced_g[0] = wrp_mpi::DotLocal(r.getDataNonConst().get(), s.getDataNonConst().get(), size, communicator);
        reduced_g[1] = wrp_mpi::DotLocal(s.getDataNonConst().get(), s.getDataNonConst().get(), size, communicator);
        reduced_g[2] = wrp_mpi::DotLocal(s.getDataNonConst().get(), t.getDataNonConst().get(), size, communicator);
        reduced_g[3] = wrp_mpi::DotLocal(t.getDataNonConst().get(), t.getDataNonConst().get(), size, communicator);
        reduced_g[4] = wrp_mpi::DotLocal(r.getDataNonConst().get(), t.getDataNonConst().get(), size, communicator);
        MPI_Allreduce(&reduced_g, &reduced_l, 5, MPI_LONG_DOUBLE, MPI_SUM, communicator);

        omega_1 = reduced_l[0];
        mu = reduced_l[1];
        nu = reduced_l[2];
        tau = reduced_l[3];
        omega_2 = reduced_l[4];

        if (mu == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (mu == 0.0)" << std::endl;
            break;
        }

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
        wrp_mpi::Update(x.getDataNonConst().get(), r.getDataNonConst().get(), s.getDataNonConst().get(), u.getDataNonConst().get(), 1.,
                static_cast<double>(omega_1), static_cast<double>(omega_2), alpha, size);

        //! (24) \f$ r = r - \omega_1 s - \omega_2 t \f$
        wrp_mpi::Update(r.getDataNonConst().get(), s.getDataNonConst().get(), t.getDataNonConst().get(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        /*!
         * Check convergence
         */
        convergence_check = wrp_mpi::Norm2(r.getDataNonConst().get(), size, communicator) / normalizer;

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
        wrp_mpi::Update(u.getDataNonConst().get(), v.getDataNonConst().get(), w.getDataNonConst().get(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        ++k;
    }
    if ( ifprint && ((k-1) % print_each) ) {
        if (myRank == 0)
            std::cout << k-1 << '\t' << convergence_check << std::endl;
    }
    iterations_num = k;
    residual_norm = convergence_check;
}

template<class Preco, class MatrixType, class VectorType>
void BiCGSTAB2::solve(
                    Preco &precond,
                    MatrixType &Matrix,
                    VectorType &x,
                    VectorType &b,
                    VectorType &x0) {

    double time1, time2, min_time, max_time, full_time;

    time1 = MPI_Wtime();
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
    long double reduced_l[5];           // used for global communication
    long double reduced_g[5];           // used for global communication

    double r_norm_0 = 0.;               // Preconditioned norm
    double convergence_check = 0.;      // keeps new residual
    double convergence_check_old = 0.;  // keeps old residual and used only is stalling checker is switched on
    double normalizer = 1.;             // normalizer for the residual

    /*
     * MPI communicators
     */
    const int myRank = x.getMap()->getComm()->getRank();
    int size = x.getMap()->getNodeNumElements();    // local system size

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

    VectorType r(x.getMap());
    VectorType r_hat_0(x.getMap());
    VectorType u(x.getMap());
    VectorType v(x.getMap());
    VectorType s(x.getMap());
    VectorType w(x.getMap());
    VectorType t(x.getMap());
    VectorType u_hat(x.getMap());
    VectorType r_hat(x.getMap());
    VectorType v_hat(x.getMap());
    VectorType s_hat(x.getMap());
    VectorType tmp(x.getMap());

    // Right preconditioner
    precond.solve(Matrix, tmp, x0, false);
    x0 = tmp;

    //! (0) \f$ r = \hat{r}_0 = b - A * x_0\f$
    Matrix.apply(x0, v);
    r = b;
    r.update(-1., v, 1.);
    wrp_mpi::Copy(r_hat_0.getDataNonConst().get(), r.getDataNonConst().get(), size);            // Actually r_hat_0 is an arbitrary vector

    //! (1) \f$ u = 0 \f$, \f$ w = 0 \f$, \f$ v = 0 \f$, \f$ \alpha = \rho[0] = \omega_1 = \omega_2 = 1\f$
    wrp_mpi::Assign(u.getDataNonConst().get(), 0.0, size);
    wrp_mpi::Assign(w.getDataNonConst().get(), 0.0, size);
    wrp_mpi::Assign(v.getDataNonConst().get(), 0.0, size);
    alpha = rho[0] = omega_1 = omega_2 = 1.;

    //! (2) Solve \f$ M y = r \f$, set \f$ r = y \f$
    // Case of left preconditioner
//      precond.solve(Matrix, tmp, r, false);
//      r = tmp;

    r_norm_0 = wrp_mpi::Norm2(r.getDataNonConst().get(), size, communicator);

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
            normalizer = wrp_mpi::Norm2(b.getDataNonConst().get(), size, communicator);
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

    time2 = MPI_Wtime();
    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, communicator);
    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);
    if (myRank == 0) {
        full_time = max_time - min_time;
        std::cout << "Setup time: " << full_time << std::endl;
    }

    time1 = MPI_Wtime();
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
        rho[1] = wrp_mpi::Dot(r_hat_0.getDataNonConst().get(), r.getDataNonConst().get(), size, communicator);
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];
        if (rho[0] == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (rho[0] == 0.0)" << std::endl;
            break;
        }

        //! (5) \f$ u = r - \beta u \f$
        wrp_mpi::Update(u.getDataNonConst().get(), r.getDataNonConst().get(), -beta, 1., size);

        //! (6) \f$ v = A M^{-1} u \f$
        precond.solve(Matrix, tmp, u, false);
        Matrix.apply(tmp, v);

        // Case of left preconditioner
//          //! (6) \f$ v = M^{-1} A u \f$
//          tmp = Matrix * u;
//          precond.solve(Matrix, v, tmp, false);

        //! (7) \f$ \gamma = <v, \hat{r}_0> \f$, \f$ \alpha = \rho[0] / \gamma \f$
        gamma = wrp_mpi::Dot(v.getDataNonConst().get(), r_hat_0.getDataNonConst().get(), size, communicator);

        // Check for breakdown (probably may occur)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (gamma == 0.0)" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (8) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r.getDataNonConst().get(), v.getDataNonConst().get(), 1., -alpha, size);

        //! (9) \f$ s = A M^{-1} r \f$
        precond.solve(Matrix, tmp, r, false);
        Matrix.apply(tmp, s);

        // Case of left preconditioner
//          //! (9) \f$ s = M^{-1} A r \f$
//          tmp  = Matrix * r;
//          precond.solve(Matrix, s, tmp, false);

        //! (10) \f$ x = x + \alpha u \f$
        wrp_mpi::Update(x.getDataNonConst().get(), u.getDataNonConst().get(), 1., alpha, size);

        /*!
         * Odd Bi-CG step
         */
        //! (11) \f$ \rho[1] = <\hat{r}_0, s>\f$, \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
        rho[1] = wrp_mpi::Dot(r_hat_0.getDataNonConst().get(), s.getDataNonConst().get(), size, communicator);
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];

        //! (12) \f$ v = s - \beta v \f$
        wrp_mpi::Update(v.getDataNonConst().get(), s.getDataNonConst().get(), -beta, 1., size);

        //! (13) \f$ w = A M^{-1} v \f$
        precond.solve(Matrix, tmp, v, false);
        Matrix.apply(tmp, w);

        // Case of left preconditioner
//          //! (13) \f$ w = M^{-1} A v \f$
//          tmp  = Matrix * v;
//          precond.solve(Matrix, w, tmp, false);

        //! (14) \f$ \gamma = <w, \hat{r}_0> \f$, \f$ \alpha = \rho[0] / \gamma \f$
        gamma = wrp_mpi::Dot(w.getDataNonConst().get(), r_hat_0.getDataNonConst().get(), size, communicator);

        // Check for breakdown (probably may occur)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (gamma == 0.0)" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (15) \f$ u = r - \beta u \f$
        wrp_mpi::Update(u.getDataNonConst().get(), r.getDataNonConst().get(), -beta, 1., size);

        //! (16) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r.getDataNonConst().get(), v.getDataNonConst().get(), 1., -alpha, size);

        //! (17) \f$ s = s - \alpha w\f$
        wrp_mpi::Update(s.getDataNonConst().get(), w.getDataNonConst().get(), 1., -alpha, size);

        //! (18) \f$ t = A M^{-1} s\f$
        precond.solve(Matrix, tmp, s, false);
        Matrix.apply(tmp, t);

        // Case of left preconditioner
//          //! (18) \f$ t = M^{-1} A s\f$
//          tmp  = Matrix * s;
//          precond.solve(Matrix, t, tmp, false);

        /*
         * GCR(2)-part
         */
        //! (19) \f$ \omega_1 = <r, s> \f$, \f$ \mu = <s, s> \f$, \f$ \nu = <s, t> \f$, \f$ \tau = <t, t> \f$
        //! (20) \f$ \omega_2 = <r, t> \f$
        reduced_g[0] = wrp_mpi::DotLocal(r.getDataNonConst().get(), s.getDataNonConst().get(), size, communicator);
        reduced_g[1] = wrp_mpi::DotLocal(s.getDataNonConst().get(), s.getDataNonConst().get(), size, communicator);
        reduced_g[2] = wrp_mpi::DotLocal(s.getDataNonConst().get(), t.getDataNonConst().get(), size, communicator);
        reduced_g[3] = wrp_mpi::DotLocal(t.getDataNonConst().get(), t.getDataNonConst().get(), size, communicator);
        reduced_g[4] = wrp_mpi::DotLocal(r.getDataNonConst().get(), t.getDataNonConst().get(), size, communicator);
        MPI_Allreduce(&reduced_g, &reduced_l, 5, MPI_LONG_DOUBLE, MPI_SUM, communicator);

        omega_1 = reduced_l[0];
        mu = reduced_l[1];
        nu = reduced_l[2];
        tau = reduced_l[3];
        omega_2 = reduced_l[4];

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
        wrp_mpi::Update(x.getDataNonConst().get(), r.getDataNonConst().get(), s.getDataNonConst().get(), u.getDataNonConst().get(), 1.,
                static_cast<double>(omega_1), static_cast<double>(omega_2), alpha, size);

        //! (25) \f$ r = r - \omega_1 s - \omega_2 t \f$
        wrp_mpi::Update(r.getDataNonConst().get(), s.getDataNonConst().get(), t.getDataNonConst().get(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        /*!
         * Check convergence
         */
        // Case of left preconditioner
//          tmp = r;
//          precond.solve(Matrix, tmp, r, false);
//          convergence_check = tmp.norm() / normalizer;

        convergence_check = wrp_mpi::Norm2(r.getDataNonConst().get(), size, communicator) / normalizer;
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
        wrp_mpi::Update(u.getDataNonConst().get(), v.getDataNonConst().get(), w.getDataNonConst().get(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        ++k;
    }
    time2 = MPI_Wtime();
    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, communicator);
    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);
    if (myRank == 0) {
        full_time = max_time - min_time;
        std::cout << "Solve time: " << full_time << std::endl;
    }

    time1 = MPI_Wtime();
    precond.solve(Matrix, tmp, x, false);
    wrp_mpi::Copy(x.getDataNonConst().get(), tmp.getDataNonConst().get(), size);

    if ( ifprint && ((k-1) % print_each) ) {
        if (myRank == 0)
            std::cout << k-1 << '\t' << convergence_check << std::endl;
    }
    iterations_num = k;
    residual_norm = convergence_check;

    time2 = MPI_Wtime();
    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, communicator);
    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);
    if (myRank == 0) {
        full_time = max_time - min_time;
        std::cout << "Extra time: " << full_time << std::endl;
    }
}
}

#endif /* KRYLOV_BICGSTAB2_H_ */
