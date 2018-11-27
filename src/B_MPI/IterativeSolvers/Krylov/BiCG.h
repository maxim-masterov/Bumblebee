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
 * \file BiCG.h
 *
 * \brief File contains class for BiConjugate Gradient method
 */

#ifndef KRYLOV_BICG_H_
#define KRYLOV_BICG_H_

#include "../../SSE/Wrappers.h"
#include "../SolversBase.h"

namespace slv_mpi {

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
 * wrp_mpi::VectorD x(n);
 * wrp_mpi::VectorD b(n);
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
template <class MatrixType, class VectorType>
class BiCG: public Base {

    VectorType *r;
    VectorType *r_hat;
    VectorType *p;
    VectorType *z;
    VectorType *p_hat;
    VectorType *z_hat;
    VectorType *tmp;
    bool reallocate;
    bool allocated;

    void FreeAll() {
        if (r != nullptr) {delete r; r = nullptr;}
        if (r_hat != nullptr) {delete r_hat; r_hat = nullptr;}
        if (p != nullptr) {delete p; p = nullptr;}
        if (z != nullptr) {delete z; z = nullptr;}
        if (p_hat != nullptr) {delete p_hat; p_hat = nullptr;}
        if (z_hat != nullptr) {delete z_hat; z_hat = nullptr;}
        if (tmp != nullptr) {delete tmp; tmp = nullptr;}
    }

public:

    BiCG(MPI_Comm _comm, bool _reallocate = false) :
        Base(_comm) {

        r = nullptr;
        r_hat = nullptr;
        p = nullptr;
        z = nullptr;
        p_hat = nullptr;
        z_hat = nullptr;
        tmp = nullptr;

        reallocate = _reallocate;
        allocated = false;
    }

    ~BiCG() {
        if (!reallocate) {
            FreeAll();
            allocated = false;
        }
    }

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
    template<class Preco>
    void solve(
                Preco &precond,
                MatrixType &Matrix,
                VectorType &x,
                VectorType &b,
                VectorType &x0);
};

template<class MatrixType, class VectorType>
void BiCG<MatrixType, VectorType>::solve(MatrixType &Matrix,
                                         VectorType &x,
                                         VectorType &b,
                                         VectorType &x0) {

    int k = 0;                              // iteration number
    double alpha = 0.;                      // part of method
    long double beta = 0.;                  // part of method
    double temp = 0.;                       // helper
    long double delta[2] = {0.};            // part of the method
    double convergence_check = 0.;          // keeps new residual
    double convergence_check_old = 0.;      // keeps old residual and used only if stalling checker is switched on
    double normalizer = 1.;                 // To normalize residual norm

    /*
     * MPI communicators
     */
    const int myRank = x.Comm().MyPID ();
    const Epetra_BlockMap _Map = x.Map();
    int size = _Map.NumMyElements();        // system size

    if (!allocated || reallocate) {
        r = new VectorType(_Map);
        r_hat = new VectorType(_Map);
        p = new VectorType(_Map);
        p_hat = new VectorType(_Map);
        tmp = new VectorType(_Map);
        allocated = true;
    }

    //To enforce "first touch"
    wrp_mpi::Assign(r->Values(), 0., size);
    wrp_mpi::Assign(r_hat->Values(), 0., size);
    wrp_mpi::Assign(p->Values(), 0., size);
    wrp_mpi::Assign(p_hat->Values(), 0., size);
    wrp_mpi::Assign(tmp->Values(), 0., size);

    //! (1)	\f$ p_0 = r_0 = b - A x_0 \f$
    Matrix.Multiply(false, x0, *tmp);
    wrp_mpi::Copy(r->Values(), b.Values(), size);
    wrp_mpi::Update(r->Values(), tmp->Values(), 1., -1., size);

    //! (1')	\f$ \hat{p}_0 = \hat{r}_0 = b - A^T x_0 \f$
    Matrix.Multiply(false, x0, *tmp);
    wrp_mpi::Copy(r_hat->Values(), b.Values(), size);
    wrp_mpi::Update(r_hat->Values(), tmp->Values(), 1., -1., size);

    wrp_mpi::Copy(p->Values(), r->Values(), size);
    wrp_mpi::Copy(p_hat->Values(), r_hat->Values(), size);

    //! Set \f$ \delta_{new} = \alpha = ||r||_2^2 \f$
    convergence_check = wrp_mpi::Dot(r_hat->Values(), r->Values(), size, communicator);
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
            normalizer = wrp_mpi::Norm2(b.Values(), size, communicator);
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
        Matrix.Multiply(false, *p, *tmp);

        // Note, this value can be very-very-very!!! small, e.g. 1e-33 O_o
        temp = wrp_mpi::Dot(p_hat->Values(), tmp->Values(), size, communicator);

        if (fabs(temp) <= LDBL_EPSILON) {
            if (myRank == 0)
                std::cout << "BiCG has been interrupted..." << std::endl;
            break;
        }
        alpha /= temp;

        //! (3)	\f$ x_{new} = x_{old} + \alpha p \f$
        wrp_mpi::Update(x.Values(), p->Values(), 1., alpha, size);

        //! (4)	\f$ r_{new} = r_{old} - \alpha A p \f$
        wrp_mpi::Update(r->Values(), tmp->Values(), 1., -alpha, size);

        //! (5)	\f$ \hat{r}_{new} = \hat{r}_{old} - \alpha A^T \hat{p} \f$
        Matrix.Multiply(true, *p_hat, *tmp);
        wrp_mpi::Update(r_hat->Values(), tmp->Values(), 1., -alpha, size);

        //! (6)	\f$ \beta = <r_{new}, \hat{r}_{new}> / <r_{old}, \hat{r}_{old}> \f$
        alpha = wrp_mpi::Dot(r_hat->Values(), r->Values(), size, communicator);
        delta[0] = delta[1];
        delta[1] = alpha;

        if (fabs(delta[0]) <= LDBL_EPSILON) {
            if (myRank == 0)
                std::cout << "BiCG has been interrupted..." << std::endl;
            break;
        }

        beta = static_cast<double>(delta[1] / delta[0]);

        //! (7)	\f$ p_{new} = r_{new} + \beta p_{old} \f$
        wrp_mpi::Update(p->Values(), r->Values(), beta, 1., size);

        //! (8)	\f$ \hat{p}_{new} = \hat{r}_{new} + \beta \hat{p}_{old} \f$
        wrp_mpi::Update(p_hat->Values(), r_hat->Values(), beta, 1., size);

        convergence_check = wrp_mpi::Norm2(r->Values(), size, communicator);
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

    if (reallocate) {
        FreeAll();
    }
}

template<class MatrixType, class VectorType>
template<class Preco>
void BiCG<MatrixType, VectorType>::solve(Preco &precond,
                                         MatrixType &Matrix,
                                         VectorType &x,
                                         VectorType &b,
                                         VectorType &x0) {

    int k = 0;                              // iteration number
    long double alpha = 0.0L;               // part of the method
    double beta = 0.;                       // part of the method
    long double delta[2] = {0.0L};			// part of the method
    long double delta_0 = 0.0L;				// part of the method
    long double temp = 0.0L;                // helper
    double convergence_check = 0.;			// keeps new residual
    double convergence_check_old = 0.;      // keeps old residual and used only if stalling checker is switched on
    double normalizer = 1.;					// To normalize residual norm

    /*
     * MPI communicators
     */
    const int myRank = x.Comm().MyPID ();
    const Epetra_BlockMap _Map = x.Map();
    int size = _Map.NumMyElements();        // system size

    /*
     * First, check if preconditioner has been built. If not - throw a warning
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
        r_hat = new VectorType(_Map);
        p = new VectorType(_Map);
        p_hat = new VectorType(_Map);
        z = new VectorType(_Map);
        z_hat = new VectorType(_Map);
        tmp = new VectorType(_Map);
        allocated = true;
    }

    //To enforce "first touch"
    wrp_mpi::Assign(r->Values(), 0., size);
    wrp_mpi::Assign(r_hat->Values(), 0., size);
    wrp_mpi::Assign(p->Values(), 0., size);
    wrp_mpi::Assign(p_hat->Values(), 0., size);
    wrp_mpi::Assign(z->Values(), 0., size);
    wrp_mpi::Assign(z_hat->Values(), 0., size);
    wrp_mpi::Assign(tmp->Values(), 0., size);

    //! (1)	\f$ r_0 = b - A x_0 \f$
    Matrix.Multiply(false, x0, *tmp);
//    r = b;
    wrp_mpi::Copy(r->Values(), b.Values(), size);
    r->Update(-1., *tmp, 1.);

    //! (2)	\f$ \hat{r}_0 = b - A^T x_0 \f$
    Matrix.Multiply(true, x0, *tmp);
    wrp_mpi::Copy(r_hat->Values(), b.Values(), size);
    r_hat->Update(-1., *tmp, 1.);

//    r_hat->Dot(*r, &convergence_check);
    convergence_check = wrp_mpi::Dot(r_hat->Values(), r->Values(), size, communicator);

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
        precond.solve(Matrix, *z, *r, true);
        precond.solve(Matrix, *z_hat, *r_hat, true);

//        z->Dot(*r_hat, &alpha);
        alpha = wrp_mpi::Dot(z->Values(), r_hat->Values(), size, communicator);

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
            wrp_mpi::Update(p->Values(), z->Values(), beta, 1., size);

            //! (6)	\f$ \hat{p}_{new} = \hat{r}_{new} + \beta \hat{p}_{old} \f$
            wrp_mpi::Update(p_hat->Values(), z_hat->Values(), beta, 1., size);
        }
        //! Else set \f$ p = z \f$, \f$ \hat{p} = \hat{z} \f$, \f$ \delta_{new} = \delta_0 = <z, \hat{r}> \f$
        else {
            delta[1] = alpha;
            delta_0 = delta[1];
            wrp_mpi::Copy(p->Values(), z->Values(), size);
            wrp_mpi::Copy(p_hat->Values(), z_hat->Values(), size);
        }

        //! (7)	\f$ \alpha = <z, \hat{r}> / <\hat{p}, A p> \f$
        Matrix.Multiply(false, *p, *tmp);
        temp = wrp_mpi::Dot(p_hat->Values(), tmp->Values(), size, communicator);

        if (fabs(temp) <= LDBL_EPSILON) {
            if (myRank == 0) {
                std::cout << "PBiCG has been interrupted..." << std::endl;
            }
            break;
        }

        alpha /= temp;

        //! (8)	\f$ x_{new} = x_{old} + \alpha p \f$
        wrp_mpi::Update(x.Values(), p->Values(), 1., alpha, size);

        //! (9)	\f$ r_{new} = r_{old} - \alpha A p \f$
        wrp_mpi::Update(r->Values(), tmp->Values(), 1., -alpha, size);

        //! (10)	\f$ \hat{r} = \hat{r}_{old} - \alpha A^T \hat{p} \f$
        Matrix.Multiply(true, *p_hat, *tmp);
        wrp_mpi::Update(r_hat->Values(), tmp->Values(), 1., -alpha, size);

        if (stop_criteria != INTERN) {
            convergence_check = wrp_mpi::Norm2(r->Values(), size, communicator);
            convergence_check /= normalizer;
        }
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

    if (reallocate) {
        FreeAll();
    }
}
}

#endif /* KRYLOV_BICG_H_ */
