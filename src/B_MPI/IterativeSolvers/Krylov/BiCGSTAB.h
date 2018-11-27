/* ****************************************************************************** *
 * MIT License                                                                    *
 *                                                                                *
 * Copyright (c) 2018 Maxim Masterov                                              *
 *                                                                                *
 * Permission is hereby granted, free of charge, to any person obtaining a copy   *
 * of this software and associated documentation files (the "Software"), to deal  *
 * in the Software without restriction, including without limitation the rights   *
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      *
 * copies of the Software, and to permit persons to whom the Software is          *
 * furnished to do so, subject to the following conditions:                       *
 *                                                                                *
 * The above copyright notice and this permission notice shall be included in all *
 * copies or substantial portions of the Software.                                *
 *                                                                                *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    *
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  *
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  *
 * SOFTWARE.                                                                      *
 * ****************************************************************************** */

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
template <class MatrixType, class VectorType>
class BiCGSTAB: public Base {

    VectorType *r;
    VectorType *r_hat_0;
    VectorType *p;
    VectorType *s;
    VectorType *v;
    VectorType *s_hat;
    VectorType *p_hat;
    VectorType *tmp;
    bool reallocate;
    bool allocated;

    void FreeAll() {
        if (r != nullptr) delete r;
        if (r_hat_0 != nullptr) delete r_hat_0;
        if (p != nullptr) delete p;
        if (s != nullptr) delete s;
        if (v != nullptr) delete v;
        if (s_hat != nullptr) delete s_hat;
        if (tmp != nullptr) delete tmp;
    }

public:

    BiCGSTAB(MPI_Comm _comm, bool _reallocate = false) :
        Base(_comm) {

        r = nullptr;
        r_hat_0 = nullptr;
        p = nullptr;
        s = nullptr;
        v = nullptr;
        s_hat = nullptr;
        p_hat = nullptr;
        tmp = nullptr;

        reallocate = _reallocate;
        allocated = false;
    }

    ~BiCGSTAB() {

        if (!reallocate) {
            FreeAll();
            allocated = false;
        }
    }

    /*!
     * \brief BiConjugate Gradient Stabilized method
     *
     * @param Matrix Incoming matrix
     * @param x Vector of unknowns
     * @param b Vector of RHS
     * @param x0 Vector of initial guess
     */
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
     * matrices.
     *
     * The class of preconditioner should have two methods:
     * - .IsBuilt() - to return true if preconditioner is built and false otherwise
     * - .solve(Matrix, y, b, false) - to apply preconditioner, where \e Matrix is a original matrix,
     *   \e y is a vector of unknowns, \e b is a vector of rhs and the last boolean indicates if preconditioner
     *   is called from the BiCG method or not.
     *
     * @param precond Object of preconditioner
     * @param Matrix Incoming matrix
     * @param x Vector of unknowns
     * @param b Vector of RHS
     * @param x0 Vector of initial guess
     */
    template<class Preco>
    void solve(
                Preco &precond,
                MatrixType &Matrix,
                VectorType &x,
                VectorType &b,
                VectorType &x0);
};

template <class MatrixType, class VectorType>
void BiCGSTAB<MatrixType, VectorType>::solve(
                    MatrixType &Matrix,
                    VectorType &x,
                    VectorType &b,
                    VectorType &x0) {

    int k = 0;                              // iteration number
    long double alpha = 0.0L;               // part of the method
    long double rho = 0.0L;                 // part of the method
    long double rho_old = 0.0L;             // part of the method
    long double omega = 0.0L;               // part of the method
    long double temp = 0.0L;                // dummy variable
    double beta = 0.;                       // part of the method
    double convergence_check = 0.;          // keeps new residual
    double convergence_check_old = 0.;      // keeps old residual and used only if stalling checker is switched on
    double normalizer = 1.;                 // residual normalizer
    double r_0_norm = 0.;                   // used to calculate norm of vector r_0
    double threshold;                       // Helps to restart solver in a case of orthogonality of r and r_hat_0 vectors

    /*
     * MPI communicators
     */
    const int myRank = x.Comm().MyPID ();
    const Epetra_BlockMap _Map = x.Map();
    int size = _Map.NumMyElements();        // local system size

    if (!allocated || reallocate) {
        r = new VectorType(_Map);
        r_hat_0 = new VectorType(_Map);
        p = new VectorType(_Map);
        s = new VectorType(_Map);
        v = new VectorType(_Map);
        tmp = new VectorType(_Map);
        allocated = true;
    }

    //To enforce "first touch"
    wrp_mpi::Assign(r->Values(), 0., size);
    wrp_mpi::Assign(r_hat_0->Values(), 0., size);
    wrp_mpi::Assign(p->Values(), 0., size);
    wrp_mpi::Assign(s->Values(), 0., size);
    wrp_mpi::Assign(v->Values(), 0., size);
    wrp_mpi::Assign(tmp->Values(), 0., size);

    //! (1) \f$ p_0 = r_0 = \hat{r}_0 = b - A * x_0 \f$
    Matrix.Multiply(false, x0, *v);
    wrp_mpi::Copy(r->Values(), b.Values(), size);
    wrp_mpi::Update(r->Values(), v->Values(), 1., -1., size);
    wrp_mpi::Copy(r_hat_0->Values(), r->Values(), size);        // Actually r_hat_0 is an arbitrary vector

    //! Set \f$ \alpha = \rho = 1 \f$
    alpha = rho = omega = 1.;
    r_0_norm = wrp_mpi::Norm2(r_hat_0->Values(), size, communicator);
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
        rho = wrp_mpi::Dot(r_hat_0->Values(), r->Values(), size, communicator);
        if (fabs(rho) < threshold)  {                               // If the residual vector r became too orthogonal to the
                                                                    // arbitrarily chosen direction r_hat_0
          r_hat_0 = r;                                              // Restart with a new r0
          rho = r_0_norm = wrp_mpi::Norm2(r->Values(), size, communicator);
          rho = r_0_norm;
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
            wrp_mpi::Update(p->Values(), r->Values(), v->Values(), beta, 1., -omega *beta, size);
        }
        else {
            wrp_mpi::Copy(p->Values(), r->Values(), size);
        }

        //! (4) \f$ \alpha = <\hat{r}_0, r> / <\hat{r}_0, A p> \f$
        Matrix.Multiply(false, *p, *v);
        temp = wrp_mpi::Dot(v->Values(), r_hat_0->Values(), size, communicator);
        alpha = rho / temp;

        //! (5) \f$  s = r_{old} - \alpha A p \f$
        wrp_mpi::Update(s->Values(), r->Values(), v->Values(), 1., -alpha, size);

        //! (6) \f$ \omega = <t, s> / <t, t>  \f$, where \f$ t = A s \f$
        // (below t is replaced with r)
        Matrix.Multiply(false, *s, *r);
        temp = wrp_mpi::Dot(r->Values(), r->Values(), size, communicator);

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
            if (!use_add_stab) {                                                        // If additional stabilization was not specified
                omega = wrp_mpi::Dot(r->Values(), s->Values(), size, communicator);
                omega /= temp;
            }
            else {                                                                      // Otherwise use limiter (for the reference see doxygen of
                double s_n = wrp_mpi::Norm2(s->Values(), size, communicator);           // Base::SetAdditionalStabilizaation())
                double r_n = sqrt(temp);
                double c = wrp_mpi::Dot(r->Values(), s->Values(), size, communicator);
                c /= s_n * r_n;

                if ( !std::signbit(c) )
                    omega = std::max(fabs(c), stab_criteria) * s_n / r_n;
                else
                    omega = - std::max(fabs(c), stab_criteria) * s_n / r_n;
            }
        }
        else {
            wrp_mpi::Update(x.Values(), p->Values(), 1., alpha, size);
            if (myRank == 0)
                std::cout << "BiCGSTAB has been interrupted..." << std::endl;
            break;
        }

        //! (7) \f$ x_{new} = x_{old} + \alpha p + \omega s  \f$
        wrp_mpi::Update(x.Values(), p->Values(), s->Values(), 1., alpha, omega, size);

        //! (8) \f$ r_{new} = s - \omega A t  \f$
        wrp_mpi::Update(r->Values(), s->Values(), -omega, 1., size);

        convergence_check = wrp_mpi::Norm2(r->Values(), size, communicator) / normalizer;

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

    if (reallocate) {
        FreeAll();
    }
}

template <class MatrixType, class VectorType>
template<class Preco>
void BiCGSTAB<MatrixType, VectorType>::solve(
                    Preco &precond,
                    MatrixType &Matrix,
                    VectorType &x,
                    VectorType &b,
                    VectorType &x0) {

    int k = 0;                              // iteration number
     double alpha = 0.;                     // part of the method
     double rho  = 0.;                      // part of the method
     double omega = 0.;                     // part of the method
     double temp = 0.;                      // dummy variable
     double rho_old = 0.;                   // part of the method
    double beta = 0.;                       // part of the method
    double convergence_check = 0.;          // keeps new residual
    double convergence_check_old = 0.;      // keeps old residual and used only if stalling checker is switched on
    double normalizer = 1.;                 // residual normalizer
     double r_0_norm = 0.;                  // used to calculate norm of vector r_0

    const int myRank = x.Comm().MyPID ();
    const Epetra_BlockMap _Map = x.Map();
    int size = _Map.NumMyElements();        // local system size

    /*
     * First check if preconditioner has been built. If not - throw a warning
     * and call for the unpreconditioned method
     */
    if (!precond.IsBuilt()) {
        if (myRank == 0) {
            std::cerr << "Warning! Preconditioner has not been built. Unpreconditioned method will be called instead..." << std::endl;
        }
        solve(Matrix, x, b, x0);
        return;
    }

    if (!allocated || reallocate) {
        r = new VectorType(_Map);
        r_hat_0 = new VectorType(_Map);
        p = new VectorType(_Map);
        s = new VectorType(_Map);
        v = new VectorType(_Map);
        s_hat = new VectorType(_Map);
        p_hat = new VectorType(_Map);
        allocated = true;
    }

    //To enforce "first touch"
    wrp_mpi::Assign(r->Values(), 0., size);
    wrp_mpi::Assign(r_hat_0->Values(), 0., size);
    wrp_mpi::Assign(p->Values(), 0., size);
    wrp_mpi::Assign(s->Values(), 0., size);
    wrp_mpi::Assign(v->Values(), 0., size);
    wrp_mpi::Assign(s_hat->Values(), 0., size);
    wrp_mpi::Assign(p_hat->Values(), 0., size);

    //! (1) \f$ p_0 = r_0 = b - A x_0 \f$
    Matrix.Multiply(false, x0, *v);
    wrp_mpi::Copy(r->Values(), b.Values(), size);
    wrp_mpi::Update(r->Values(), v->Values(), 1., -1., size);
    wrp_mpi::Copy(r_hat_0->Values(), r->Values(), size);        // Actually r_hat_0 is an arbitrary vector

    //! Set \f$ \alpha = \rho = 1 \f$
    rho = omega = 1.;
    alpha = 0.;
    r_0_norm = wrp_mpi::Norm2(r_hat_0->Values(), size, communicator);

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
        rho = wrp_mpi::Dot(r->Values(), r_hat_0->Values(), size, communicator);
        if (fabs(rho) < (eps*eps*r_0_norm)) {                                   // If the residual vector r became too orthogonal to the
                                                                                // arbitrarily chosen direction r_hat_0
            wrp_mpi::Copy(r_hat_0->Values(), r->Values(), size);                // Restart with a new r0
            rho = r_0_norm = wrp_mpi::Norm2(r->Values(), size, communicator);
        }

        if (k == 1) {
            wrp_mpi::Copy(p->Values(), r->Values(), size);
        }
        else {
            //! (3) \f$ d_{new} = r_{new} + \beta * d_{old} \f$
            beta = static_cast<double>( (rho/rho_old) * (alpha / omega) );
            wrp_mpi::Update(p->Values(), r->Values(), v->Values(), beta, 1.,
                    static_cast<double>(-omega) * beta, size);
        }

        //! (4) Apply preconditioner. Solve \f$ M \hat{p} = p \f$
        precond.solve(Matrix, *p_hat, *p, false);

        //! (5) \f$ \alpha = <\hat{r}_0, r> / <\hat{r}_0, A \hat{p}> \f$
        Matrix.Multiply(false, *p_hat, *v);

        temp = wrp_mpi::Dot(v->Values(), r_hat_0->Values(), size, communicator);
        alpha = rho / temp;

        //! (6) \f$ s = r_{old} - \alpha A \hat{p} \f$
        // Actually after this step one can check L2-norm of s and stop if it is too small, since
        // anyway in such case omega will be NaN and algorithm will be terminated
        wrp_mpi::Update(s->Values(), r->Values(), v->Values(), 1., static_cast<double>(-alpha), size);

        //! (7) Apply preconditioner. Solve \f$ M \hat{s} = s \f$
        precond.solve(Matrix, *s_hat, *s, false);

        //! (8) \f$ \omega = <t, s> / <t, t> \f$, where \f$ t = A \hat{s} \f$
        // (below t = r)
        Matrix.Multiply(false, *s_hat, *r);
        temp = wrp_mpi::Dot(r->Values(), r->Values(), size, communicator);

        // Check for breakdown
        if (temp != 0.0) {
            if (!use_add_stab) {                                        // If additional stabilization was not specified
                omega = wrp_mpi::Dot(r->Values(), s->Values(), size, communicator);
                omega /= temp;
            }
            else {                                                                                  // Otherwise use limiter (for the reference see doxygen of
                double s_n = wrp_mpi::Norm2(s->Values(), size, communicator);                        // Base::SetAdditionalStabilizaation())
                double r_n = sqrt(temp);
                double c = static_cast<double>(wrp_mpi::Dot(r->Values(), s->Values(), size, communicator));
                c /= (s_n * r_n);

                if ( !std::signbit(c) )
                    omega = std::max(fabs(c), stab_criteria) * s_n / r_n;
                else
                    omega = -std::max(fabs(c), stab_criteria) * s_n / r_n;
            }
        }
        else {
            // (8*) Update solution and stop
            wrp_mpi::Update(x.Values(), p_hat->Values(), 1., static_cast<double>(alpha), size);
            if (myRank == 0)
                std::cout << "BiCGSTAB has been interrupted..." << std::endl;
            break;
        }

        //! (9) \f$ x_{new} = x_{old} + \alpha \hat{p} + \omega \hat{s} \f$
        wrp_mpi::Update(x.Values(), p_hat->Values(), s_hat->Values(), 1., static_cast<double>(alpha),
                static_cast<double>(omega), size);

        //! (10)    \f$ r_{new} = s - \omega A t \f$
        wrp_mpi::Update(r->Values(), s->Values(), static_cast<double>(-omega), 1., size);

        /*!
         * Check convergence
         */
        convergence_check = wrp_mpi::Norm2(r->Values(), size, communicator) / normalizer;

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

    if (reallocate) {
        FreeAll();
    }
}
}

#endif /* KRYLOV_BICGSTAB_H_ */
