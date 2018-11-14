# Bumblebee
The `Bumblebee` is a set of Krylov subspace solvers including: CG, BiCG, BiCGStab and BiCGStab(2). 
All solvers have two versions - unpreconditioned and preconditioned.

The library relies on the `Epetra` and `ML` libs from the `Trilinos` project (see https://trilinos.org/). 
The `Epetra` library is used to represent distributed matrices and vectors, as well as to 
perform Mat-Vec product. The `ML` library is used as an AMG preconditioner.

# Classes of preconditioners
The `Bumblebee` supports external preconditioners if they can be provided as an object of
a class with two methods: `::solve()` and `::IsBuilt()`. The `::solve()` method should take
reference to the matrix object, references to the LHS and RHS vectors and a boolean indicating
that the method is calles from BiCG method and special treatment of preconditioning matrix
might be applied. The method `::IsBuit()` should simply return `true` if preconditioning matrix
is built and `false` otherwise.

As an example of preconditioner class implementation see folder `AMG`, where ML library from 
the Trilinos project is wrapped in order to provide required interface.

# SSE/AVX support
The `Bumblebee` utilizes low-level intrinsics if symbol `USE_MAGIC_POWDER` is added as a 
compilation flag.

# Hybrid parallelization
The `Bumblebee` is created to support MPI parallelization strategy at first, but can also work
with hybrid parallelization. Just include `BUMBLEBEE_USE_OPENMP` symbol into the makefile and
be sure that Trilinos packages are compiled with OpenMP support (add `-D Trilinos_ENABLE_OpenMP:BOOL=ON`
to the do-configure script)

# do-configure script
The script is a simple call for CMake in order to build Trilinos. Use it in conjunction with the 
`Trilinos` manuals and tutorials to build all required libraries.

# Performance
Parallel performance tests were done for the BiCGStab(2) solver accelerated with AMG. The solver scales
well up to 1536 tested cored (strong scaling). An example is shown in the figure below. Tested problem
consisted of a solution of a heat diffusion problem (with constant diffusion coefficient) discretized on 
a uniform Cartesian grid with 27M DoF. Tests were performed on 2x12 cores Intel Xeon E5-2690 v3 CPU with
basic clock speed 2.6 GHz. Code was compiled with Intel C++ compiler v16.0.3 and Intel MPI v5.1.3.
![alt text](https://github.com/maxim-masterov/Bumblebee/blob/master/pics/bicgstab2_ml_27M.pdf "Bumblebee parallel performance")
