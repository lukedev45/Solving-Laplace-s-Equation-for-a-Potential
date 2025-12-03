# Solving Laplace’s Equation for a Potential  
*A numerical solution to Laplace’s equation on a 2D grid using finite-difference methods and iterative relaxation.*

## 📌 Overview  
This project implements a numerical solver for **Laplace’s equation** in two dimensions:

$$
\nabla^2 \phi(x, y) = 0
$$

The solver uses a **finite-difference discretisation** on a rectangular grid and iteratively relaxes the potential until convergence is achieved.  
This type of simulation is widely used in:

- Electrostatics (electric potential in charge-free regions)  
- Heat flow at steady-state  
- Fluid flow (stream functions)  
- Gravitational potential problems  
- General potential theory and harmonic functions  

The project also includes visualisation of the resulting potential distribution.

---

## 🎯 Motivation  
Laplace’s equation is one of the most fundamental PDEs in physics and mathematics.  
Solving it numerically:

- builds intuition for harmonic functions, potential theory, and boundary-value problems,  
- reinforces understanding of numerical partial-difference schemes,  
- provides a foundation for more advanced PDE solvers (Poisson equation, diffusion equation, Navier–Stokes, etc.),  
- and directly connects to computational physics and engineering simulations.

This project was created as part of my applied mathematics work to deepen my understanding of **finite-difference numerical methods** and **steady-state PDE modelling**.

---

## 🧠 Methods & Theory  

### 🧩 Discretisation  
The 2D Laplace equation is approximated using the standard 5-point stencil:

$$
\phi_{i,j}^{\text{new}} =
\frac{1}{4}
\left(
\phi_{i+1,j} +
\phi_{i-1,j} +
\phi_{i,j+1} +
\phi_{i,j-1}
\right)
$$


Grid spacing is assumed uniform (Δx = Δy).  
This converts the continuous PDE into a system of linear equations solved iteratively.

### 🔁 Iterative Relaxation  
The solver uses iterative updates (Jacobi or Gauss–Seidel style depending on implementation):

1. Initialise the potential grid  
2. Apply boundary conditions  
3. Update interior points using the stencil  
4. Continue until the maximum grid change < tolerance

This mirrors classical relaxation methods used in scientific computing.

---

## 🚀 Getting Started  

### 🔧 Requirements  
- Python 3.x  
- `numpy`  
- `matplotlib`  

Install dependencies:

```bash
pip install numpy matplotlib

git clone https://github.com/lukedev45/Solving-Laplace-s-Equation-for-a-Potential
cd Solving-Laplace-s-Equation-for-a-Potential

python laplace_solver.py
```

## 🚧 Future Extensions

There are several natural directions for extending this solver. These improvements increase realism, numerical stability, and computational performance.

### 🔹 1. Solve Poisson’s Equation
Extend the solver to handle the more general equation:

$$
\nabla^2 \phi = \rho(x, y)
$$


Useful for modelling charge distributions, gravitational potentials, and fluid pressure fields.

### 🔹 2. Implement Non-Uniform or Adaptive Grids
Allow variable spacing in the x and y directions:

- Concentrate resolution in regions of interest
- Reduce total grid size while maintaining accuracy
- Explore basic adaptive mesh refinement (AMR)

### 🔹 3. Add SOR (Successive Over-Relaxation)
Enhance convergence speed by modifying the update rule:

$$
\phi^{\text{new}} =
(1 - \omega)\,\phi^{\text{old}}
\;+\;
\omega\,\phi_{\text{relaxed}}
$$


Choosing \( 1 < \omega < 2 \) significantly accelerates relaxation.

### 🔹 4. Support Mixed Boundary Conditions
Add support for:

- **Dirichlet** (fixed value)
- **Neumann** (fixed derivative)
- **Robin** (mixed)

This allows simulation of more realistic physical situations.

### 🔹 5. Compute and Visualise the Electric Field
Once the potential is solved, compute:

$$
\mathbf{E} = -\nabla \phi
$$


and plot:

- Field vectors (quiver plots)
- Field magnitude maps
- Streamlines

### 🔹 6. GPU Acceleration
Accelerate the computation by using:

- `numba` (CPU JIT acceleration)
- `cupy` (CUDA GPU arrays)
- `jax` (autodiff + JIT)

This enables very large grids or real-time updates.

### 🔹 7. Extend to 3D Laplace Solver
Generalise to:

$$
\nabla^2 \phi(x, y, z) = 0
$$


using a 7-point stencil and volumetric visualisation.

### 🔹 8. Add Time-Dependent Solvers
Although Laplace’s equation is steady-state, adding:

- Heat equation
- Diffusion equation
- Wave equation

allows step-by-step evolution toward equilibrium.

