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

#ifndef KRYLOV_IBICGSTAB2_H_
#define KRYLOV_IBICGSTAB2_H_

#include "../../SSE/Wrappers.h"
#include "../SolversBase.h"
#include <Epetra_Time.h>

namespace slv_mpi {

/*!
 * \ingroup KrylovSolvers
 * \class IBiCGSTAB2
 * \brief Improved BiConjugate Gradient Stabilized (2) method
 *
 * Improved BiConjugate Gradients Stabilized (2) method works with both symmetric and non-symmetric matrices.
 * This method is stable and is an improved version of BiCGSTAB(2).
 *
 * Was proposed by Tong-Xiang Gu, Xian-Yu Zuo, Xing-Ping Liu and Pei-Lu Li "An improved parallel hybrid bi-conjugate gradient method suitable
 * for distributed parallel computing",
 * 2009, JournalofComputationalandApplied Mathematics, Vol. 226, pp. 55-65.
 *
 * \warning Class works only for preconditioned systems!
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
 * IBiCGSTAB2 bicgstab;
 *
 * // Build preconditioner...
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
 * bicgstab.solve(preco, A, x, b, x);
 *
 * // Print out number of iterations and residual
 * std::cout << "Iterations: " << bicgstab.Iterations() << std::endl;
 * std::cout << "Residual: " << bicgstab.Residual() << std::endl;
 * \endcode
 */
class IBiCGSTAB2: public Base {

public:

    IBiCGSTAB2(MPI_Comm _comm) :
        Base(_comm) {
    };

    ~IBiCGSTAB2() {
    };

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

template<class Preco, class MatrixType, class VectorType>
void IBiCGSTAB2::solve(
                    Preco &precond,
                    MatrixType &Matrix,
                    VectorType &x,
                    VectorType &b,
                    VectorType &x0) {

    double time1, time2, min_time, max_time, full_time;

    time1 = MPI_Wtime();
    int iter = 0;                          // iteration number
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

    long double a, _b, c, d, e, f, g, h, k;

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
//        solve(Matrix, x, b, x0);
        return;
    }

    VectorType r(_Map);
    VectorType r_hat_0(_Map);
    VectorType p(_Map);
    VectorType v(_Map);
    VectorType s(_Map);
    VectorType w(_Map);
    VectorType t(_Map);
    VectorType u(_Map);
    VectorType t_hat(_Map);
    VectorType w_hat(_Map);
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

    Matrix.Multiply(false, r, s);

    wrp_mpi::Copy(t.Values(), s.Values(), size);
    Matrix.Multiply(false, t, u);

    //! (1) \f$ u = 0 \f$, \f$ w = 0 \f$, \f$ v = 0 \f$, \f$ \alpha = \rho[0] = \omega_1 = \omega_2 = 1\f$
    wrp_mpi::Assign(p.Values(), 0.0, size);
    wrp_mpi::Assign(v.Values(), 0.0, size);
    wrp_mpi::Assign(w.Values(), 0.0, size);
    alpha = rho[0] = rho[1] = omega_1 = omega_2 = 1.;
    h = k = 0.;
    f = 1.;

    c = wrp_mpi::Dot(r_hat_0.Values(), r.Values(), size, communicator);
    d = wrp_mpi::Dot(r_hat_0.Values(), s.Values(), size, communicator);
    e = wrp_mpi::Dot(r_hat_0.Values(), t.Values(), size, communicator);

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
        if ( ifprint && !(iter % print_each) ) {
            if (myRank == 0)
                std::cout << iter << '\t' << convergence_check << std::endl;
        }
        iterations_num = iter;
        residual_norm = convergence_check;
        return;
    }

    if ( ifprint ) {
        if (myRank == 0)
            std::cout << iter << '\t' << convergence_check << std::endl;
    }
    ++iter;

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

        if (iter > MaxIter) {
            break;
        }

        //! (3) \f$ \rho[0] = - \omega_2 \rho[0] \ \f$
        rho[0] = - omega_2 * rho[1];
        if (rho[0] == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (rho[0] == 0.0)" << std::endl;
            break;
        }

        /*!
         * Even Bi-CG step
         */
        //! (4) \f$ \rho[1] = <\hat{r}_0, r>\f$, \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
        rho[1] = c - omega_1 * d - omega_2 * e;
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];
        if (rho[0] == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (rho[0] == 0.0)" << std::endl;
            break;
        }

        //! (5) \f$ u = r - \beta u \f$
        /* p = r - beta * (p - omega_1 * v - omega_2 * w) */
        /* v = s - omega_1 * t - omega_2 * u - beta * (v - omega_1 * w - omega_2 * x) */
        for(int i = 0; i < size; ++i) {
            p.Values()[i] = r.Values()[i] - beta * (p.Values()[i] - omega_1 * v.Values()[i] - omega_2 * w.Values()[i]);
            v.Values()[i] = s.Values()[i] - omega_1 * t.Values()[i] - omega_2 * u.Values()[i]
                           - beta * (v.Values()[i] - omega_1 * w.Values()[i] - omega_2 * x.Values()[i]);
        }

        //! (6) \f$ v = A M^{-1} u \f$
        precond.solve(Matrix, tmp, v, false);
        Matrix.Multiply(false, tmp, w_hat);

        // Case of left preconditioner
//          //! (6) \f$ v = M^{-1} A u \f$
//          tmp = Matrix * u;
//          precond.solve(Matrix, v, tmp, false);

        //! (7) \f$ \gamma = <v, \hat{r}_0> \f$, \f$ \alpha = \rho[0] / \gamma \f$
        gamma = d - omega_1 * e - omega_2 * f - beta * (gamma - omega_1 * h - omega_2 * k);

        // Check for breakdown (probably may occur)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (gamma == 0.0)" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (8) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r.Values(), v.Values(), 1., -alpha, size);
        /* s = s - omega_1 * t - omega_2 * u - alpha * w_hat */
        for(int i = 0; i < size; ++i) {
            s.Values()[i] = s.Values()[i] - omega_1 * t.Values()[i] - omega_2 * u.Values()[i] - alpha * w_hat.Values()[i];
        }

        //! (9) \f$ s = A M^{-1} r \f$
        precond.solve(Matrix, tmp, s, false);
        Matrix.Multiply(false, tmp, t_hat);

        // Case of left preconditioner
//          //! (9) \f$ s = M^{-1} A r \f$
//          tmp  = Matrix * r;
//          precond.solve(Matrix, s, tmp, false);

        //! (10) \f$ x = x + \alpha u \f$
        wrp_mpi::Update(x.Values(), p.Values(), 1., alpha, size);

        a = wrp_mpi::Dot(r_hat_0.Values(), t_hat.Values(), size, communicator);
        _b = wrp_mpi::Dot(r_hat_0.Values(), w_hat.Values(), size, communicator);

        /*!
         * Odd Bi-CG step
         */
        //! (11) \f$ \rho[1] = <\hat{r}_0, s>\f$, \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
        rho[1] = d - omega_1 * e - omega_2 * f - alpha * _b;
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];

        //! (12) \f$ p = r - \beta p \f$
        wrp_mpi::Update(p.Values(), r.Values(), -beta, 1., size);
        //! (12) \f$ v = s - \beta v \f$
        wrp_mpi::Update(v.Values(), s.Values(), -beta, 1., size);
        //! (12) \f$ w = t_hat - \beta w_hat \f$
        wrp_mpi::Update(w.Values(), t_hat.Values(), w_hat.Values(), 1., -beta, size);

        //! (13) \f$ w = A M^{-1} v \f$
        precond.solve(Matrix, tmp, w, false);
        Matrix.Multiply(false, tmp, x);

        // Case of left preconditioner
//          //! (13) \f$ w = M^{-1} A v \f$
//          tmp  = Matrix * v;
//          precond.solve(Matrix, w, tmp, false);

        //! (14) \f$ \gamma = <w, \hat{r}_0> \f$, \f$ \alpha = \rho[0] / \gamma \f$
        gamma = a - beta * _b;

        // Check for breakdown (probably may occur)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (gamma == 0.0)" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (16) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r.Values(), v.Values(), 1., -alpha, size);

        //! (17) \f$ s = s - \alpha w\f$
        wrp_mpi::Update(s.Values(), w.Values(), 1., -alpha, size);

        //! (15) \f$ t = t_hat - \alpha x \f$
        wrp_mpi::Update(t.Values(), t_hat.Values(), -alpha, 1., size);

        //! (18) \f$ t = A M^{-1} s\f$
        precond.solve(Matrix, tmp, t, false);
        Matrix.Multiply(false, tmp, u);

        // Case of left preconditioner
//          //! (18) \f$ t = M^{-1} A s\f$
//          tmp  = Matrix * s;
//          precond.solve(Matrix, t, tmp, false);

        /*
         * GCR(2)-part
         */
        //! (19) \f$ \omega_1 = <r, s> \f$, \f$ \mu = <s, s> \f$, \f$ \nu = <s, t> \f$, \f$ \tau = <t, t> \f$
        c = wrp_mpi::Dot(r.Values(), r_hat_0.Values(), size, communicator);
        d = wrp_mpi::Dot(s.Values(), r_hat_0.Values(), size, communicator);
        e = wrp_mpi::Dot(t.Values(), r_hat_0.Values(), size, communicator);
        f = wrp_mpi::Dot(u.Values(), r_hat_0.Values(), size, communicator);
        g = wrp_mpi::Dot(v.Values(), r_hat_0.Values(), size, communicator);
        h = wrp_mpi::Dot(w.Values(), r_hat_0.Values(), size, communicator);
        k = wrp_mpi::Dot(x.Values(), r_hat_0.Values(), size, communicator);

        omega_1 = wrp_mpi::Dot(r.Values(), s.Values(), size, communicator);
        mu = wrp_mpi::Dot(s.Values(), s.Values(), size, communicator);
        nu = wrp_mpi::Dot(s.Values(), t.Values(), size, communicator);
        tau = wrp_mpi::Dot(t.Values(), t.Values(), size, communicator);
        omega_2 = wrp_mpi::Dot(r.Values(), t.Values(), size, communicator);

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
        wrp_mpi::Update(x.Values(), r.Values(), s.Values(), p.Values(), 1.,
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
        if ( ifprint && !(iter % print_each) ) {
            if (myRank == 0)
                std::cout << iter << '\t' << convergence_check << std::endl;
        }

        if( convergence_check <= eps && iter > 1) {
            break;
        }

        /*
         * Check for convergence stalling
         */
        if (check_stalling && !(iter % check_stalling_each) ) {
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

        ++iter;
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
    wrp_mpi::Copy(x.Values(), tmp.Values(), size);

    if ( ifprint && ((iter-1) % print_each) ) {
        if (myRank == 0)
            std::cout << iter-1 << '\t' << convergence_check << std::endl;
    }
    iterations_num = iter;
    residual_norm = convergence_check;

    time2 = MPI_Wtime();
    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, communicator);
    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);
    if (myRank == 0) {
        full_time = max_time - min_time;
        std::cout << "Extra time: " << full_time << std::endl;
    }
//    MPI_Barrier(communicator);
}
}

#endif /* KRYLOV_BICGSTAB2_H_ */
