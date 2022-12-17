# RL-Laboratory
Official repository for the 2022/2023 Reinforcement Learning Laboratory of the University of Verona.

## Assignements
Following the link to the code snippets for the lessons:

**First Semester**
- [x] Lesson 1: MDP and Gym Environments [Code!](lessons/lesson_1_code.py) [Results!](results/lesson_1_results.txt)
- [x] Lesson 2: Value/Policy Iteration [Code!](lessons/lesson_2_code.py) [Results!](results/lesson_2_results.txt)
- [x] Lesson 3: Monte Carlo Methods [Code!](lessons/lesson_3_code.py) [Results!](results/lesson_3_results.txt)
- [ ] Lesson 4: Q-Learning and Sarsa *Coming Soon...*
- [ ] Lesson 5:  *Coming Soon...*
- [ ] Lesson 6:  *Coming Soon...*

**Second Semester**
- [ ] Lesson 7:  *Coming Soon...*
- [ ] Lesson 8:  *Coming Soon...*
- [ ] Lesson 9:  *Coming Soon...*
- [ ] Lesson 10:  *Coming Soon...*
- [ ] Lesson 11:  *Coming Soon...*
- [ ] Lesson 12:  *Coming Soon...*

## Tutorials
This repo includes a set of introductory tutorials to help accomplish the exercises. In detail, we provide the following Jupyter notebook that contains the basic instructions for the lab:
- **Tutorial 1 - Gym Environment:** [Here!](tutorials/tutorial_environment.ipynb)
- **Tutorial 2 - Neural Network and Keras:** *Coming Soon...*

## First Set-Up (Conda)
1. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your System.

2.  Install Miniconda
	- On Linux/Mac 
		- Use *./Miniconda3-latest-Linux-{version}.sh* to install.
		- *sudo apt-get install git* (may be required).
	- On Windows
		- Double click the installer to launch.
		- *NB: Ensure to install "Anaconda Prompt" and use it for the other steps.*

3.  Set-Up conda environment:
	- *git clone https://github.com/d-corsi/RL-Lab*
	- *conda env create -f RL-Lab/tools/rl-lab-environment.yml*

## First Set-Up (Python Virtual Environments)
Python virtual environments users (venv) can avoid the Miniconda installation. The following package should be installed:
  - scipy, numpy, gym
  - jupyter, matplotlib, tqdm
  - tensorflow, keras

## Authors
*  **Davide Corsi** - davide.corsi@univr.it
*  **Alberto Castellini** - alberto.castellini@univr.it
