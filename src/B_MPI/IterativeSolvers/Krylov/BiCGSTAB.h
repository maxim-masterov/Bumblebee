/*
 * BiCGSTAB.h
 *
 *  Created on: Apr 4, 2016
 *      Author: maxim
 */

/*!
 * \file BiCGSTAB.h
 *
 * \brief File contains class for BiConjugate Gradient Stabilized method
 */

#ifndef KRYLOV_BICGSTAB_H_
#define KRYLOV_BICGSTAB_H_

#include "../../SSE/Wrappers.h"
#include "../SolversBase.h"

namespace slv_mpi {

/*!
 * \ingroup KrylovSolvers
 * \class BiCGSTAB
 * \brief BiConjugate Gradient Stabilized method
 *
 * BiConjugate Gradient Stabilized method works with both symmetric and non-symmetric matrices.
 * This method is stable and is an improved version of BiCG.
 *
 * \note Be aware, in some cases convergence of method has jumps with significant amplitude
 * (saltatory residual). Moreover, in some cases the amplitude of jumps can be very high, so be
 * sure to set proper tolerance.
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
 * BiCGSTAB bicgstab;
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
class BiCGSTAB: public Base {

public:

    BiCGSTAB(MPI_Comm _comm) :
        Base(_comm) {
    }
    ;

    ~BiCGSTAB() {
    }
    ;

    /*!
     * \brief BiConjugate Gradient Stabilized method
     *
     * \par For developers
     * Method contains:
     *  - 3 Matrix-Vector Multiplication \n
     *      (2 in loop, 1 out of loop)
     *  - 3 Vector-Vector Multiplication \n
     *      (3 in loop)
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
     * Preconditioned BiConjugate Gradient Stabilized method uses right-preconditioned form, i.e.
     *                  \f{eqnarray*}{ A M^{-1} y =  b, \\ x = M^{-1} y \f}
     * where \f$ M^{-1}\f$ is a preconditioning matrix. Matrix \f$ M \f$ can be obtained via different ways, e.g.
     * Incomplete LU factorization, Incomplete Cholesky factorization etc.
     *
     * \note Method is free-transpose, i.e. it doesn't ask for a transposition of original and preconditioning
     * matrices and can be cheaper than BiCG method
     *
     * \par For developers
     * Method contains:
     *  - 3 Matrix-Vector Multiplication \n
     *      (2 in loop, 1 out of loop)
     *  - 3 Vector-Vector Multiplication \n
     *      (3 in loop)
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
void BiCGSTAB::solve(
                    MatrixType &Matrix,
                    VectorType &x,
                    VectorType &b,
                    VectorType &x0) {

    int k = 0;                              // iteration number
     double alpha = 0.0L;               // part of the method
     double rho = 0.0L;                 // part of the method
     double rho_old = 0.0L;             // part of the method
     double omega = 0.0L;               // part of the method
     double temp = 0.0L;                // dummy variable
    double beta = 0.;                       // part of the method
    double convergence_check = 0.;          // keeps new residual
    double convergence_check_old = 0.;      // keeps old residual and used only if stalling checker is switched on
    double normalizer = 1.;                 // residual normalizer
    double r_0_norm = 0.;                   // used to calculate norm of vector r_0
    double threshold;                       // Helps to restart solver in a case of orthogonality of r and r_hat_0 vectors

    /*
     * MPI communicators
     */
    const int myRank = x.getMap()->getComm()->getRank();
    int size = x.getMap()->getNodeNumElements();    // local system size

    VectorType r(x.getMap());
    VectorType r_hat_0(x.getMap());
    VectorType p(x.getMap());
    VectorType s(x.getMap());
    VectorType v(x.getMap());

    VectorType tmp(x.getMap());

    //! (1) \f$ p_0 = r_0 = \hat{r}_0 = b - A * x_0 \f$
    Matrix.apply(x0, v);
    r = b;
    r.update(-1., v, 1.);
    r_hat_0 = r;                            // Actually r_hat_0 is an arbitrary vector

    //! Set \f$ \alpha = \rho = 1 \f$
    alpha = rho = omega = 1.;
    r_0_norm = wrp_mpi::Norm2(r_hat_0.getDataNonConst().get(), size, communicator);
    threshold = eps * eps * r_0_norm;

    /*!
     * Prepare stop criteria
     */
    switch (stop_criteria) {
        case RNORM:
            normalizer = 1.;
            break;
        case RRNORM: case INTERN:
            normalizer = r_0_norm;
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
    convergence_check = r_0_norm / normalizer;
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

        rho_old = rho;

        //! (2) \f$ \beta = <\hat{r}_0, r_{new}> / <\hat{r}_0, r_{old}> \f$
        rho = wrp_mpi::Dot(r_hat_0.getDataNonConst().get(), r.getDataNonConst().get(), size, communicator);
        if (fabs(rho) < threshold)  {   // If the residual vector r became too orthogonal to the
                                                // arbitrarily chosen direction r_hat_0
          r_hat_0 = r;                          // Restart with a new r0
          rho = r_0_norm = wrp_mpi::Norm2(r.getDataNonConst().get(), size, communicator);
          threshold = eps*eps*r_0_norm;
        }

        if (k > 1) {
            //! (3) \f$ d_{new} = r_{new} + \beta d_{old} \f$
            beta = (rho / rho_old) * (alpha / omega);
            /*
             * Warning! There is a difference between two statements:
             * p = r + beta * (p - omega * v);
             * and
             * p = r + beta*p - beta*omega*v;
             * Probably due to the roundoff error. In first statement
             * temporary vector in parenthesis is calculated first,
             * then it is scaled by beta. In the second statement
             * scaling by beta applied separately to two vectors.
             */
            wrp_mpi::Update(p.getDataNonConst().get(), r.getDataNonConst().get(), v.getDataNonConst().get(), beta, 1., -omega*beta, size);
        }
        else {
            wrp_mpi::Copy(p.getDataNonConst().get(), r.getDataNonConst().get(), size);
        }

        //! (4) \f$ \alpha = <\hat{r}_0, r> / <\hat{r}_0, A p> \f$
        Matrix.apply(p, v);
        temp = wrp_mpi::Dot(r_hat_0.getDataNonConst().get(), v.getDataNonConst().get(), size, communicator);
        alpha = rho / temp;

        //! (5) \f$  s = r_{old} - \alpha A p \f$
        wrp_mpi::Update(s.getDataNonConst().get(), r.getDataNonConst().get(), v.getDataNonConst().get(), 1., -alpha, size);

        //! (6) \f$ \omega = <t, s> / <t, t>  \f$, where \f$ t = A s \f$
        // (below t is replaced with r)
        Matrix.apply(s, r);
        temp = wrp_mpi::Dot(r.getDataNonConst().get(), r.getDataNonConst().get(), size, communicator);

        /*
         * TODO: checking for the breakdown below can be done right after 5th step (s=...) as follows
         *      if (||s|| <= tol ∗ ||b||)
         *          x += alpha * p;
         *          r = s;
         *          break;
         * Thus, matrix-vector multiplication and dot product may be avoided
         */

        // Check for breakdown
        if (temp != 0.0) {
            if (!use_add_stab) {                                        // If additional stabilization was not specified
                omega = wrp_mpi::Dot(r.getDataNonConst().get(), s.getDataNonConst().get(), size, communicator);
                omega /= temp;
            }
            else {                                                      // Otherwise use limiter (for the reference see doxygen of
                double s_n = wrp_mpi::Norm2(s.getDataNonConst().get(), size, communicator);                       // Base::SetAdditionalStabilizaation())
                double r_n = sqrt(temp);
                double c = wrp_mpi::Dot(r.getDataNonConst().get(), s.getDataNonConst().get(), size, communicator);

                if ( !std::signbit(c) )
                    omega = std::max(fabs(c), stab_criteria) * s_n / r_n;
                else
                    omega = - std::max(fabs(c), stab_criteria) * s_n / r_n;
            }
        }
        else {
            wrp_mpi::Update(x.getDataNonConst().get(), p.getDataNonConst().get(), 1., alpha, size);
            if (myRank == 0)
                std::cout << "BiCGSTAB has been interrupted..." << std::endl;
            break;
        }

        //! (7) \f$ x_{new} = x_{old} + \alpha p + \omega s  \f$
        wrp_mpi::Update(x.getDataNonConst().get(), p.getDataNonConst().get(), s.getDataNonConst().get(), 1., alpha, omega, size);

        //! (8) \f$ r_{new} = s - \omega A t  \f$
        wrp_mpi::Update(r.getDataNonConst().get(), s.getDataNonConst().get(), -omega, 1., size);

        convergence_check = wrp_mpi::Norm2(r.getDataNonConst().get(), size, communicator) / normalizer;

        /*!
         * Check convergence
         */
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
        if (check_stalling && (k % check_stalling_each) ) {
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
    if ( ifprint && ((k-1) % print_each) ) {
        if (myRank == 0)
            std::cout << k-1 << '\t' << convergence_check << std::endl;
    }
    iterations_num = k;
    residual_norm = convergence_check;
}

template<class Preco, class MatrixType, class VectorType>
void BiCGSTAB::solve(
                    Preco &precond,
                    MatrixType &Matrix,
                    VectorType &x,
                    VectorType &b,
                    VectorType &x0) {

    int k = 0;                              // iteration number
     double alpha = 0.;                 // part of the method
     double rho  = 0.;                  // part of the method
     double omega = 0.;                 // part of the method
     double temp = 0.;                  // dummy variable
     double rho_old = 0.;               // part of the method
    double beta = 0.;                       // part of the method
    double convergence_check = 0.;          // keeps new residual
    double convergence_check_old = 0.;      // keeps old residual and used only if stalling checker is switched on
    double normalizer = 1.;                 // residual normalizer
     double r_0_norm = 0.;              // used to calculate norm of vector r_0

     const int myRank = x.getMap()->getComm()->getRank();
     int size = x.getMap()->getNodeNumElements();    // local system size

    /*
     * First check if preconditioner has been built. If not - through a warning
     * and call for the unpreconditioned method
     */
    if (!precond.IsBuilt()) {
        if (myRank == 0) {
            std::cerr << "Warning! Preconditioner has not been built. Unpreconditioned method will be called instead..." << std::endl;
        }
        solve(Matrix, x, b, x0);
        return;
    }

    VectorType r(x.getMap());
    VectorType r_hat_0(x.getMap());
    VectorType p(x.getMap());
    VectorType s(x.getMap());
    VectorType v(x.getMap());

    VectorType s_hat(x.getMap());
    VectorType p_hat(x.getMap());

    //! (1) \f$ p_0 = r_0 = b - A x_0 \f$
    Matrix.apply(x0, v);
    r = b;
    r.update(-1., v, 1.);
    r_hat_0 = r;                                // Actually r_hat_0 is an arbitrary vector

    //! Set \f$ \alpha = \rho = 1 \f$
    rho = omega = 1.;
    alpha = 0.;
    r_0_norm = wrp_mpi::Norm2(r_hat_0.getDataNonConst().get(), size, communicator);

    /*!
     * Prepare stop criteria
     */
    switch (stop_criteria) {
        case RNORM:
            normalizer = 1.;
            break;
        case RRNORM: case INTERN:
            normalizer = r_0_norm;
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
    convergence_check = r_0_norm / normalizer;
    if (convergence_check < eps) {
        if ( ifprint && !(k % print_each) ) {
            if (myRank == 0)
                std::cout << k << '\t' << convergence_check << std::endl;
        }
        iterations_num = k;
        residual_norm = convergence_check;
        return;
    }

    if ( ifprint && myRank == 0)
        std::cout << k << '\t' << convergence_check/normalizer << std::endl;
    ++k;

    //! Start iterative loop
    while(1) {

        if(k > MaxIter) {
            break;
        }

        rho_old = rho;

        //! (2) \f$ \beta = <\hat{r}_0, r_{new}> / <\hat{r}_0, r_{old}> \f$
        rho = wrp_mpi::Dot(r_hat_0.getDataNonConst().get(), r.getDataNonConst().get(), size, communicator);
        if (fabs(rho) < (eps*eps*r_0_norm)) {       // If the residual vector r became too orthogonal to the
                                                    // arbitrarily chosen direction r_hat_0
            wrp_mpi::Copy(r_hat_0.getDataNonConst().get(), r.getDataNonConst().get(), size);            // Restart with a new r0
//            r_hat_0 = r;
            rho = r_0_norm = wrp_mpi::Norm2(r.getDataNonConst().get(), size, communicator);
        }

        if (k == 1) {
            wrp_mpi::Copy(p.getDataNonConst().get(), r.getDataNonConst().get(), size);
        }
        else {
            //! (3) \f$ d_{new} = r_{new} + \beta * d_{old} \f$
            beta = static_cast<double>( (rho/rho_old) * (alpha / omega) );
            wrp_mpi::Update(p.getDataNonConst().get(), r.getDataNonConst().get(), v.getDataNonConst().get(), beta, 1.,
                    static_cast<double>(-omega) * beta, size);
        }

        //! (4) Apply preconditioner. Solve \f$ M \hat{p} = p \f$
        precond.solve(Matrix, p_hat, p, false);

        //! (5) \f$ \alpha = <\hat{r}_0, r> / <\hat{r}_0, A \hat{p}> \f$
        Matrix.apply(p_hat, v);

        temp = wrp_mpi::Dot(r_hat_0.getDataNonConst().get(), v.getDataNonConst().get(), size, communicator);
        alpha = rho / temp;

        //! (6) \f$ s = r_{old} - \alpha A \hat{p} \f$
        // Actually after this step one can check L2-norm of s and stop if it is too small, since
        // anyway in such case omega will be NaN and algorithm will be terminated
        wrp_mpi::Update(s.getDataNonConst().get(), r.getDataNonConst().get(), v.getDataNonConst().get(), 1., static_cast<double>(-alpha), size);

        //! (7) Apply preconditioner. Solve \f$ M \hat{s} = s \f$
        precond.solve(Matrix, s_hat, s, false);

        //! (8) \f$ \omega = <t, s> / <t, t> \f$, where \f$ t = A \hat{s} \f$
        // (below t = r)
        Matrix.apply(s_hat, r);
        temp = wrp_mpi::Dot(r.getDataNonConst().get(), r.getDataNonConst().get(), size, communicator);

        // Check for breakdown
        if (temp != 0.0) {
            if (!use_add_stab) {                                        // If additional stabilization was not specified
                omega = wrp_mpi::Dot(r.getDataNonConst().get(), s.getDataNonConst().get(), size, communicator);
                omega /= temp;
            }
            else {                                                                                  // Otherwise use limiter (for the reference see doxygen of
                double s_n = wrp_mpi::Norm2(s.getDataNonConst().get(), size, communicator);                        // Base::SetAdditionalStabilizaation())
                double r_n = sqrt(temp);
                double c = static_cast<double>(wrp_mpi::Dot(r.getDataNonConst().get(), s.getDataNonConst().get(), size, communicator));
                c /= (s_n * r_n);

                if ( !std::signbit(c) )
                    omega = std::max(fabs(c), stab_criteria) * s_n / r_n;
                else
                    omega = -std::max(fabs(c), stab_criteria) * s_n / r_n;
            }
        }
        else {
            // (8*) Update solution and stop
            wrp_mpi::Update(x.getDataNonConst().get(), p_hat.getDataNonConst().get(), 1., static_cast<double>(alpha), size);
            if (myRank == 0)
                std::cout << "BiCGSTAB has been interrupted..." << std::endl;
            break;
        }

        //! (9) \f$ x_{new} = x_{old} + \alpha \hat{p} + \omega \hat{s} \f$
        wrp_mpi::Update(x.getDataNonConst().get(), p_hat.getDataNonConst().get(), s_hat.getDataNonConst().get(), 1., static_cast<double>(alpha),
                static_cast<double>(omega), size);

        //! (10)    \f$ r_{new} = s - \omega A t \f$
        wrp_mpi::Update(r.getDataNonConst().get(), s.getDataNonConst().get(), static_cast<double>(-omega), 1., size);

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
                if (ifprint && myRank == 0)
                    std::cout << "Convergence stalling detected..." << std::endl;
                break;
            }
        }

        ++k;
    }
    if ( ifprint && ((k-1) % print_each) ) {
        if (myRank == 0)
            std::cout << k-1 << '\t' << convergence_check << std::endl;
    }
    iterations_num = k;
    residual_norm = convergence_check;
}
}

#endif /* KRYLOV_BICGSTAB_H_ */
