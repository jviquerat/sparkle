# sparkle

<p align="center">
  <img align="right" width="350" alt="logo" src="sparkle/msc/logo.png">
</p>

![master badge](https://github.com/jviquerat/sparkle/workflows/sparkle/badge.svg?branch=master)

`sparkle` is a parametric, gradient-free optimization library. It is designed to provide a common interface to various algorithms, and to make numerical experimentation easy.

Implementation of the following algorithms is planned:

- Particle swarm optimization (PSO)
- Covariance matrix adaptation evolution strategy (CMAES)
- Efficient global optimization (EGO)
- Policy based optimization (PBO)

More informations about each method can be obtained from the documentation. Below are several optimization examples performed with the different methods.

| **`parabola (pso)`**                                                      | **`rosenbrock (cmaes)`**                                             | **`sinebump (pso)`**                                             |
| :-----------------------------------------------------------------------: | :------------------------------------------------------------------: | :--------------------------------------------------------------: |
| <img height="250" alt="gif" src="sparkle/save/parabola_pso.gif">          | <img height="250" alt="gif" src="sparkle/save/rosenbrock_cmaes.gif"> | <img height="250" alt="gif" src="sparkle/save/sinebump_pso.gif"> |
| **`packing (cmaes)`**                                                     | **`?`**                                                              | **`?`**                                                          |
| <img height="250" alt="gif" src="sparkle/save/packing_circles_cmaes.gif"> | <img height="250" alt="gif" src="sparkle/msc/logo.png">              | <img height="250" alt="gif" src="sparkle/msc/logo.png">          |
