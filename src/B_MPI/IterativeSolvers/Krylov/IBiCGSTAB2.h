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
 * Was proposed by Tong-Xiang Gp, Xian-Yu Zuo, Xing-Ping Liu and Pei-Lu Li "An improved parallel hybrid bi-conjugate gradient method suitable
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
template <class MatrixType, class VectorType>
class IBiCGSTAB2: public Base {

    VectorType *r;
    VectorType *r_hat_0;
    VectorType *u;
    VectorType *v;
    VectorType *s;
    VectorType *w;
    VectorType *t;
    VectorType *u_hat;
    VectorType *r_hat;
    VectorType *v_hat;
    VectorType *s_hat;
    VectorType *w_hat;
    VectorType *t_hat;
    VectorType *p;
    VectorType *xi;
    VectorType *tmp;

    bool reallocate;
    bool allocated;

public:

    IBiCGSTAB2(MPI_Comm _comm, bool _reallocate = false) :
        Base(_comm) {
        reallocate = _reallocate;
        allocated = false;

        r = nullptr;
        r_hat_0 = nullptr;
        u = nullptr;
        v = nullptr;
        s = nullptr;
        w = nullptr;
        t = nullptr;
        u_hat = nullptr;
        r_hat = nullptr;
        v_hat = nullptr;
        s_hat = nullptr;
        w_hat =nullptr;
        t_hat = nullptr;
        p = nullptr;
        xi = nullptr;
        tmp = nullptr;
    }

    ~IBiCGSTAB2() {
        if (!reallocate) {
            if (r != nullptr) delete r;
            if (r_hat_0 != nullptr) delete r_hat_0;
            if (u != nullptr) delete u;
            if (v != nullptr) delete v;
            if (s != nullptr) delete s;
            if (w != nullptr) delete w;
            if (t != nullptr) delete t;
            if (u_hat != nullptr) delete u_hat;
            if (r_hat != nullptr) delete r_hat;
            if (v_hat != nullptr) delete v_hat;
            if (s_hat != nullptr) delete s_hat;
            if (w_hat != nullptr) delete w_hat;
            if (t_hat != nullptr) delete t_hat;
            if (p) delete p;
            if (xi != nullptr) delete xi;
            if (tmp != nullptr) delete tmp;
            allocated = false;
        }
    }

    /*!
     * \brief Improved Hybrid BiConjugate Gradient Stabilized method
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
    void solve(
                MatrixType &Matrix,
                VectorType &x,
                VectorType &b,
                VectorType &x0);

    /*!
     * \brief Preconditioned Improved Hybrid BiConjugate Gradient Stabilized method
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
    template<class Preco>
    void solve(
                Preco &precond,
                MatrixType &Matrix,
                VectorType &x,
                VectorType &b,
                VectorType &x0);
};

template <class MatrixType, class VectorType>
void IBiCGSTAB2<MatrixType, VectorType>::solve(
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
    long double reduced_g[12];           // used for global communication
    long double reduced_l[12];           // used for global communication

    long double _c, _d, _e;
    long double _f, _a, _b;
    long double _h, _k;

    double convergence_check = 0.;      // keeps new residual
    double convergence_check_old = 0.;  // keeps old residual and used only is stalling checker is switched on
    double normalizer = 1.;

    /*
     * MPI communicators
     */
    const int myRank = x.Comm().MyPID ();
    const Epetra_BlockMap _Map = x.Map();
    int size = _Map.NumMyElements();    // local system size

    if (!allocated || reallocate) {
        r = new VectorType(_Map);
        r_hat_0 = new VectorType(_Map);
        u = new VectorType(_Map);
        v = new VectorType(_Map);
        s = new VectorType(_Map);
        w = new VectorType(_Map);
        t = new VectorType(_Map);
        w_hat = new VectorType(_Map);
        t_hat = new VectorType(_Map);
        p = new VectorType(_Map);
        xi = new VectorType(_Map);
        allocated = true;
    }

    //! (0) \f$ r = \hat{r}_0 = b - A * x_0\f$
    Matrix.Multiply(false, x0, *v);
    wrp_mpi::Copy(r->Values(), b.Values(), size);
    r->Update(-1., *v, 1.);
    wrp_mpi::Copy(r_hat_0->Values(), r->Values(), size);        // Actually r_hat_0 is an arbitrary vector

    //! (1) \f$ u = 0 \f$, \f$ p = 0 \f$, \f$ v = 0 \f$, \f$ w = 0 \f$, \f$ f = 0 \f$
    //!     \f$ \alpha = \omega_1[0] = 0 \f$, \f$ \rho[0] = \omega_2[0] = 1 \f$
    //!     \f$ s = A r \f$, \f$ t = s \f$
    wrp_mpi::Zero(u->Values(), size);
    wrp_mpi::Zero(p->Values(), size);
    wrp_mpi::Zero(v->Values(), size);
    wrp_mpi::Zero(w->Values(), size);

    Matrix.Multiply(false, *r, *s);
    wrp_mpi::Copy(t->Values(), s->Values(), size);

    reduced_g[0] = wrp_mpi::DotLocal(r->Values(), r_hat_0->Values(), size, communicator);
    reduced_g[1] = wrp_mpi::DotLocal(s->Values(), r_hat_0->Values(), size, communicator);
    MPI_Allreduce(&reduced_g, &reduced_l, 2, MPI_LONG_DOUBLE, MPI_SUM, communicator);
    _c = reduced_l[0];
    _d = reduced_l[1];
    _e = 0.;

    alpha   = 0.;
    rho[0]  = 1.;
    omega_1 = 0.;
    omega_2 = 1.;
    _f = 0.;

    /*!
     * Prepare stop criteria
     */
    switch (stop_criteria) {
        case RNORM:
            normalizer = 1.;
            break;
        case RRNORM: case INTERN:
            normalizer = wrp_mpi::Norm2(r->Values(), size, communicator);
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
    convergence_check = wrp_mpi::Norm2(r->Values(), size, communicator) / normalizer;
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
        //! (3) \f$ \rho[1] = -c - \omega_1 d - \omega_2 e \f$,
        //!     \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
        rho[1] = _c - omega_1 * _d - omega_2 * _e;
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];
        if (rho[0] == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (rho[0] == 0.0)" << std::endl;
            break;
        }

        //! (4) \f$ u = r - \beta u \f$
        wrp_mpi::Update(u->Values(), r->Values(), -beta, 1., size);

        //! (5) \f$ v = s - \omega_1 t - \omega_2 p - \beta v \f$
        wrp_mpi::Update(v->Values(), s->Values(), t->Values(), p->Values(), -beta,
                1., static_cast<double>(-omega_1), static_cast<double>(-omega_2), size);

        //! (6) \f$ \hat{w} = A v \f$
        Matrix.Multiply(false, *v, *w_hat);

        //! (7) \f$ \gamma = d - \omega_1 e - \omega_2 f - \beta (\gamma - \omega_1 h - omega_2 k ) \f$,
        //!     \f$ \alpha = \rho[0] / \gamma \f$
        gamma = _d - omega_1 * _e - omega_2 * _f - beta * (gamma - omega_1 * _h - omega_2 * _k);

        // Check for breakdown (probably may occur)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (8) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r->Values(), v->Values(), 1., -alpha, size);

        //! (9) \f$ s = s - \omega_1 t - omega_2 p - \alpha \hat{w} \f$
        wrp_mpi::Update(s->Values(), t->Values(), p->Values(), w_hat->Values(), 1.,
                static_cast<double>(-omega_1), static_cast<double>(-omega_2),
                static_cast<double>(-alpha),  size);

        //! (10) \f$ \hat{t} = A s \f$
        Matrix.Multiply(false, *s, *t_hat);

        //! (11) \f$ x = x + \alpha u \f$
        wrp_mpi::Update(x.Values(), u->Values(), 1., alpha, size);

        //! (12) \f$ a = <\hat{t}, \hat{r}_0> \f$, \f$ b = <\hat{w}, \hat{r}_0> \f$
        reduced_g[0] = wrp_mpi::DotLocal(t_hat->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[1] = wrp_mpi::DotLocal(w_hat->Values(), r_hat_0->Values(), size, communicator);
        MPI_Allreduce(&reduced_g, &reduced_l, 2, MPI_LONG_DOUBLE, MPI_SUM, communicator);
        _a = reduced_l[0];
        _b = reduced_l[1];

        /*!
         * Odd Bi-CG step
         */
        //! (13) \f$ \rho[1] = d - \omega_1 e - \omega_2 f - \alpha b \f$,
        //!      \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
        rho[1] = _d - omega_1 * _e - omega_2 * _f - alpha * _b;
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];

        //! (14) \f$ v = s - \beta v \f$
        wrp_mpi::Update(v->Values(), s->Values(), -beta, 1., size);

        //! (15) \f$ w = \hat{t} - \beta \hat{w} \f$
        wrp_mpi::Update(w->Values(), t_hat->Values(), w_hat->Values(), 1., -beta, size);

        //! (16) \f$ \xi = A w \f$
        Matrix.Multiply(false, *w, *xi);

        //! (17) \f$ \gamma = a - \beta b \f$, \f$ \alpha = \rho[0] / \gamma \f$
        gamma = _a - beta * _b;

        // Check for breakdown (may occur if matrix is diagonal)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (18) \f$ u = r - \beta u \f$
        wrp_mpi::Update(u->Values(), r->Values(), -beta, 1., size);

        //! (19) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r->Values(), v->Values(), 1., -alpha, size);

        //! (20) \f$ s = s - \alpha w\f$
        wrp_mpi::Update(s->Values(), w->Values(), 1., -alpha, size);

        //! (21) \f$ t = \hat{t} - \alpha \xi \f$
        wrp_mpi::Update(t->Values(), t_hat->Values(), xi->Values(), 1., -alpha, size);

        //! (22) \f$ p = A t \f$
        Matrix.Multiply(false, *t, *p);

        /*!
         * GCR(2)-part
         */
        //! (23) \f$ \omega_1 = <r, s> \f$, \f$ \mu = <s, s> \f$, \f$ \nu = <s, t> \f$, \f$ \tau = <t, t> \f$,
        //!      \f$ \omega_2 = <r, t> \f$
        reduced_g[0] = wrp_mpi::DotLocal(r->Values(), s->Values(), size, communicator);
        reduced_g[1] = wrp_mpi::DotLocal(s->Values(), s->Values(), size, communicator);
        reduced_g[2] = wrp_mpi::DotLocal(s->Values(), t->Values(), size, communicator);
        reduced_g[3] = wrp_mpi::DotLocal(t->Values(), t->Values(), size, communicator);
        reduced_g[4] = wrp_mpi::DotLocal(r->Values(), t->Values(), size, communicator);

        //! (24) \f$ c = <r, \hat{r}_0> \f$, \f$ d = <s, \hat{r}_0> \f$, \f$ e = <t, \hat{r}_0> \f$,
        //!      \f$ f = <p, \hat{r}_0> \f$, \f$ h = <w, \hat{r}_0> \f$, \f$ k = <xi, \hat{r}_0> \f$
        //!      \f$ \gamma = <v, \hat{r}_0> \f$
        reduced_g[5] = wrp_mpi::DotLocal(r->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[6] = wrp_mpi::DotLocal(s->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[7] = wrp_mpi::DotLocal(t->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[8] = wrp_mpi::DotLocal(p->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[9] = wrp_mpi::DotLocal(w->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[10] = wrp_mpi::DotLocal(xi->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[11] = wrp_mpi::DotLocal(v->Values(), r_hat_0->Values(), size, communicator);

        MPI_Allreduce(&reduced_g, &reduced_l, 12, MPI_LONG_DOUBLE, MPI_SUM, communicator);

        omega_1 = reduced_l[0];
        mu = reduced_l[1];
        nu = reduced_l[2];
        tau = reduced_l[3];
        omega_2 = reduced_l[4];

        _c = reduced_l[5];
        _d = reduced_l[6];
        _e = reduced_l[7];
        _f = reduced_l[8];
        _h = reduced_l[9];
        _k = reduced_l[10];
        gamma = reduced_l[11];

        if (mu == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (mu == 0.0)" << std::endl;
            break;
        }

        //! (25) \f$ \tau = \tau - \nu^2 / \mu \f$
        tau -= nu * nu / mu;

        if (tau == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (tau == 0.0)" << std::endl;
            break;
        }

        //! (26) \f$ \omega_2 = (\omega_2 - \nu \omega_1 / \mu) / \tau \f$
        omega_2 = (omega_2 - (nu * omega_1) / mu) / tau;

        //! (22) \f$ \omega_1 = (\omega_1 - \nu \omega_2) / \mu \f$
        omega_1 = (omega_1 - nu * omega_2) / mu;

        //! (27) \f$ x = x + \omega_1 r + \omega_2 s + \alpha u \f$
        wrp_mpi::Update(x.Values(), r->Values(), s->Values(), u->Values(), 1.,
                static_cast<double>(omega_1), static_cast<double>(omega_2), alpha, size);

        //! (28) \f$ r = r - \omega_1 s - \omega_2 t \f$
        wrp_mpi::Update(r->Values(), s->Values(), t->Values(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

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
                if (ifprint) {
                    if (myRank == 0)
                        std::cout << "Convergence stalling detected..." << std::endl;
                }
                break;
            }
        }

        //! (29) \f$ u = u - \omega_1 v - \omega_2 w \f$
        wrp_mpi::Update(u->Values(), v->Values(), w->Values(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        //! (30) \f$ v = v - \omega_1 w - \omega_2 xi \f$
        wrp_mpi::Update(v->Values(), w->Values(), xi->Values(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        ++k;
    }
    if ( ifprint && ((k-1) % print_each) ) {
        if (myRank == 0)
            std::cout << k-1 << '\t' << convergence_check << std::endl;
    }
    iterations_num = k;
    residual_norm = convergence_check;

    if (reallocate) {
        delete r;
        delete r_hat_0;
        delete u;
        delete v;
        delete s;
        delete w;
        delete t;
        delete w_hat;
        delete t_hat;
        delete p;
        delete xi;
    }
}

template <class MatrixType, class VectorType>
template<class Preco>
void IBiCGSTAB2<MatrixType, VectorType>::solve(
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
    long double reduced_g[12];           // used for global communication
    long double reduced_l[12];           // used for global communication

    long double _c, _d, _e;
    long double _f, _a, _b;
    long double _h, _k;

    double convergence_check = 0.;      // keeps new residual
    double convergence_check_old = 0.;  // keeps old residual and used only is stalling checker is switched on
    double normalizer = 1.;

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

    if (!allocated || reallocate) {
        r = new VectorType(_Map);
        r_hat_0 = new VectorType(_Map);
        u = new VectorType(_Map);
        v = new VectorType(_Map);
        s = new VectorType(_Map);
        w = new VectorType(_Map);
        t = new VectorType(_Map);
        w_hat = new VectorType(_Map);
        t_hat = new VectorType(_Map);
        p = new VectorType(_Map);
        xi = new VectorType(_Map);
        tmp = new VectorType(_Map);
        allocated = true;
    }

    // Right preconditioner
    precond.solve(Matrix, *tmp, x0, false);
    wrp_mpi::Copy(x0.Values(), tmp->Values(), size);

    //! (0) \f$ r = \hat{r}_0 = b - A * x_0\f$
    Matrix.Multiply(false, x0, *v);
    wrp_mpi::Copy(r->Values(), b.Values(), size);
    r->Update(-1., *v, 1.);
    wrp_mpi::Copy(r_hat_0->Values(), r->Values(), size);            // Actually r_hat_0 is an arbitrary vector

    //! (1) \f$ u = 0 \f$, \f$ p = 0 \f$, \f$ v = 0 \f$, \f$ w = 0 \f$, \f$ f = 0 \f$
    //!     \f$ \alpha = \omega_1[0] = 0 \f$, \f$ \rho[0] = \omega_2[0] = 1 \f$
    //!     \f$ s = A r \f$, \f$ t = s \f$
    wrp_mpi::Zero(u->Values(), size);
    wrp_mpi::Zero(p->Values(), size);
    wrp_mpi::Zero(v->Values(), size);
    wrp_mpi::Zero(w->Values(), size);

    Matrix.Multiply(false, *r, *s);
    wrp_mpi::Copy(t->Values(), s->Values(), size);

    precond.solve(Matrix, *tmp, *s, false);
    wrp_mpi::Copy(s->Values(), tmp->Values(), size);

    _c = wrp_mpi::Dot(r->Values(), r_hat_0->Values(), size, communicator);
    _d = wrp_mpi::Dot(s->Values(), r_hat_0->Values(), size, communicator);
    _e = _d;//wrp_mpi::Dot(t->Values(), r_hat_0->Values(), size, communicator);//0.;

    std::cout << _c << " " << _d << " " << _e << "\n";
    alpha   = 1.;
    rho[0]  = 1.;
    omega_1 = 0.;
    omega_2 = 1.;
    _f = 0.;
    _h = 0.;

    /*!
     * Prepare stop criteria
     */
    switch (stop_criteria) {
        case RNORM:
            normalizer = 1.;
            break;
        case RRNORM: case INTERN:
            normalizer = wrp_mpi::Norm2(r->Values(), size, communicator);
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
    convergence_check = wrp_mpi::Norm2(r->Values(), size, communicator) / normalizer;
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
        //! (3) \f$ \rho[1] = -c - \omega_1 d - \omega_2 e \f$,
        //!     \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
        rho[1] = _c - omega_1 * _d - omega_2 * _e;
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];
        if (rho[0] == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (rho[0] == 0.0)" << std::endl;
            break;
        }

        //! (4) \f$ u = r - \beta u \f$
        wrp_mpi::Update(u->Values(), r->Values(), -beta, 1., size);

        //! (5) \f$ v = s - \omega_1 t - \omega_2 p - \beta v \f$
        wrp_mpi::Update(v->Values(), s->Values(), t->Values(), p->Values(), -beta,
                1., static_cast<double>(-omega_1), static_cast<double>(-omega_2), size);

        //! (6) \f$ \hat{w} = A M^{-1} v \f$
//        Matrix.Multiply(false, *v, *w_hat);
        precond.solve(Matrix, *tmp, *v, false);
        Matrix.Multiply(false, *tmp, *w_hat);

        //! (7) \f$ \gamma = d - \omega_1 e - \omega_2 f - \beta (\gamma - \omega_1 h - omega_2 k ) \f$,
        //!     \f$ \alpha = \rho[0] / \gamma \f$
        gamma = _d - omega_1 * _e - omega_2 * _f - beta * (gamma - omega_1 * _h - omega_2 * _k);

        std::cout << "\n";
        std::cout << "omega_1: " << omega_1 << "\n";
        std::cout << "omega_2: " << omega_2 << "\n";
        std::cout << "rho[0]: " << rho[0] << "\n";
        std::cout << "alpha: " << alpha << "\n";
        std::cout << "beta: " << beta << "\n";
        std::cout << "gamma: " << gamma << "\n\n";

        // Check for breakdown (probably may occur)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (8) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r->Values(), v->Values(), 1., -alpha, size);

        //! (9) \f$ s = s - \omega_1 t - omega_2 p - \alpha \hat{w} \f$
        wrp_mpi::Update(s->Values(), t->Values(), p->Values(), w_hat->Values(), 1.,
                static_cast<double>(-omega_1), static_cast<double>(-omega_2),
                static_cast<double>(-alpha),  size);

        //! (10) \f$ \hat{t} = A M^{-1} s \f$
//        Matrix.Multiply(false, *s, *t_hat);
        precond.solve(Matrix, *tmp, *s, false);
        Matrix.Multiply(false, *tmp, *t_hat);

        //! (11) \f$ x = x + \alpha u \f$
        wrp_mpi::Update(x.Values(), u->Values(), 1., alpha, size);

        //! (12) \f$ a = <\hat{t}, \hat{r}_0> \f$, \f$ b = <\hat{w}, \hat{r}_0> \f$
        reduced_g[0] = wrp_mpi::DotLocal(t_hat->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[1] = wrp_mpi::DotLocal(w_hat->Values(), r_hat_0->Values(), size, communicator);
        MPI_Allreduce(&reduced_g, &reduced_l, 2, MPI_LONG_DOUBLE, MPI_SUM, communicator);
        _a = reduced_l[0];
        _b = reduced_l[1];

        /*!
         * Odd Bi-CG step
         */
        //! (13) \f$ \rho[1] = d - \omega_1 e - \omega_2 f - \alpha b \f$,
        //!      \f$ \beta = \alpha \rho[1] / \rho[0] \f$, \f$ \rho[0] = \rho[1] \f$
        rho[1] = _d - omega_1 * _e - omega_2 * _f - alpha * _b;
        beta = alpha * rho[1] / rho[0];
        rho[0] = rho[1];

        std::cout << "rho[0]: " << rho[0] << "\n";
        std::cout << "alpha: " << alpha << "\n";
        std::cout << "beta: " << beta << "\n";
        std::cout << "gamma: " << gamma << "\n\n";


        //! (14) \f$ v = s - \beta v \f$
        wrp_mpi::Update(v->Values(), s->Values(), -beta, 1., size);

        //! (15) \f$ w = \hat{t} - \beta \hat{w} \f$
        wrp_mpi::Update(w->Values(), t_hat->Values(), w_hat->Values(), 1., -beta, size);

        //! (16) \f$ \xi = A M^{-1} w \f$
//        Matrix.Multiply(false, *w, *xi);
        precond.solve(Matrix, *tmp, *w, false);
        Matrix.Multiply(false, *tmp, *xi);

        //! (17) \f$ \gamma = a - \beta b \f$, \f$ \alpha = \rho[0] / \gamma \f$
        gamma = _a - beta * _b;

        // Check for breakdown (may occur if matrix is diagonal)
        if (gamma == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted" << std::endl;
            break;
        }

        alpha = rho[0] / gamma;

        //! (18) \f$ u = r - \beta u \f$
        wrp_mpi::Update(u->Values(), r->Values(), -beta, 1., size);

        //! (19) \f$ r = r - \alpha v \f$
        wrp_mpi::Update(r->Values(), v->Values(), 1., -alpha, size);

        //! (20) \f$ s = s - \alpha w\f$
        wrp_mpi::Update(s->Values(), w->Values(), 1., -alpha, size);

        //! (21) \f$ t = \hat{t} - \alpha \xi \f$
        wrp_mpi::Update(t->Values(), t_hat->Values(), xi->Values(), 1., -alpha, size);

        //! (22) \f$ p = A M^{-1} t \f$
//        Matrix.Multiply(false, *t, *p);
        precond.solve(Matrix, *tmp, *t, false);
        Matrix.Multiply(false, *tmp, *p);

        /*!
         * GCR(2)-part
         */
        //! (23) \f$ \omega_1 = <r, s> \f$, \f$ \mu = <s, s> \f$, \f$ \nu = <s, t> \f$, \f$ \tau = <t, t> \f$,
        //!      \f$ \omega_2 = <r, t> \f$
        reduced_g[0] = wrp_mpi::DotLocal(r->Values(), s->Values(), size, communicator);
        reduced_g[1] = wrp_mpi::DotLocal(s->Values(), s->Values(), size, communicator);
        reduced_g[2] = wrp_mpi::DotLocal(s->Values(), t->Values(), size, communicator);
        reduced_g[3] = wrp_mpi::DotLocal(t->Values(), t->Values(), size, communicator);
        reduced_g[4] = wrp_mpi::DotLocal(r->Values(), t->Values(), size, communicator);

        //! (24) \f$ c = <r, \hat{r}_0> \f$, \f$ d = <s, \hat{r}_0> \f$, \f$ e = <t, \hat{r}_0> \f$,
        //!      \f$ f = <p, \hat{r}_0> \f$, \f$ h = <w, \hat{r}_0> \f$, \f$ k = <xi, \hat{r}_0> \f$
        //!      \f$ \gamma = <v, \hat{r}_0> \f$
        reduced_g[5] = wrp_mpi::DotLocal(r->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[6] = wrp_mpi::DotLocal(s->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[7] = wrp_mpi::DotLocal(t->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[8] = wrp_mpi::DotLocal(p->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[9] = wrp_mpi::DotLocal(w->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[10] = wrp_mpi::DotLocal(xi->Values(), r_hat_0->Values(), size, communicator);
        reduced_g[11] = wrp_mpi::DotLocal(v->Values(), r_hat_0->Values(), size, communicator);

        MPI_Allreduce(&reduced_g, &reduced_l, 12, MPI_LONG_DOUBLE, MPI_SUM, communicator);

        omega_1 = reduced_l[0];
        mu = reduced_l[1];
        nu = reduced_l[2];
        tau = reduced_l[3];
        omega_2 = reduced_l[4];

        _c = reduced_l[5];
        _d = reduced_l[6];
        _e = reduced_l[7];
        _f = reduced_l[8];
        _h = reduced_l[9];
        _k = reduced_l[10];
        gamma = reduced_l[11];

        if (mu == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (mu == 0.0)" << std::endl;
            break;
        }

        //! (25) \f$ \tau = \tau - \nu^2 / \mu \f$
        tau -= nu * nu / mu;

        if (tau == 0.0) {
            if (myRank == 0)
                std::cout << "BiCGSTAB(2) has been interrupted due to (tau == 0.0)" << std::endl;
            break;
        }

        //! (26) \f$ \omega_2 = (\omega_2 - \nu \omega_1 / \mu) / \tau \f$
        omega_2 = (omega_2 - (nu * omega_1) / mu) / tau;

        //! (22) \f$ \omega_1 = (\omega_1 - \nu \omega_2) / \mu \f$
        omega_1 = (omega_1 - nu * omega_2) / mu;

        //! (27) \f$ x = x + \omega_1 r + \omega_2 s + \alpha u \f$
        wrp_mpi::Update(x.Values(), r->Values(), s->Values(), u->Values(), 1.,
                static_cast<double>(omega_1), static_cast<double>(omega_2), alpha, size);

        //! (28) \f$ r = r - \omega_1 s - \omega_2 t \f$
        wrp_mpi::Update(r->Values(), s->Values(), t->Values(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

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
                if (ifprint) {
                    if (myRank == 0)
                        std::cout << "Convergence stalling detected..." << std::endl;
                }
                break;
            }
        }

        //! (29) \f$ u = u - \omega_1 v - \omega_2 w \f$
        wrp_mpi::Update(u->Values(), v->Values(), w->Values(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        //! (30) \f$ v = v - \omega_1 w - \omega_2 xi \f$
        wrp_mpi::Update(v->Values(), w->Values(), xi->Values(), 1., static_cast<double>(-omega_1),
                static_cast<double>(-omega_2), size);

        ++k;
    }

    precond.solve(Matrix, *tmp, x, false);
    wrp_mpi::Copy(x.Values(), tmp->Values(), size);

    if ( ifprint && ((k-1) % print_each) ) {
        if (myRank == 0)
            std::cout << k-1 << '\t' << convergence_check << std::endl;
    }
    iterations_num = k;
    residual_norm = convergence_check;

    if (reallocate) {
        delete r;
        delete r_hat_0;
        delete u;
        delete v;
        delete s;
        delete w;
        delete t;
        delete w_hat;
        delete t_hat;
        delete p;
        delete xi;
        delete tmp;
    }
}
}

#endif /* KRYLOV_BICGSTAB2_H_ */
