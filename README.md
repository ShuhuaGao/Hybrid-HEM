# A Hybrid Approach for Home Energy Management

## Environment setup

The major part of our code is written in [Julia](https://julialang.org/) 1.7.0. The RL part is implemented with Python 3.9 and PyTorch.

After installing Julia 1.7, enter its REPL from this root directory and
type the following two commands to set up the proper Julia environment.

- `]activate .` (activate the current environment)
- `instantiate` (install all dependencies of this project)


## File organization and how to run

- The implementation of the core functions including MILP and IL is provided in the HEMS package in *./src/HEMS*. All functions are documented in details.
- Jupyter notebooks in *./notebook* are executed to run different experiments with the above HEMS package. The notebooks can be run in [VS code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) or in a browser with [IJulia](https://github.com/JuliaLang/IJulia.jl). Alternatively, you may view the notebooks directly on GitHub.
  - Run *notebook\IL\IL_data.ipynb* to generate the expert data.
  - Run *notebook\IL\BC_shiftable.ipynb* to train the DNN agents for shiftable loads via IL based on the above expert data.
  - Run *notebook\IL\inspection.ipynb* to inspect characteristics of the algorithms as shown in the paper.
- The home configuration and scenario data are stored in the *./data* directory.
- The other directories contain intermediate images, data, and models etc.
