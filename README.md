# RL-Laboratory
Official repository for the 2022/2023 Reinforcement Learning Laboratory of the University of Verona.

## Assignements
Following the link to the code snippets for the lessons:

**First Semester**
- [x] Lesson 1: MDP and Gym Environments [Code!](lessons/lesson_1_code.py) [Results!](results/lesson_1_results.txt) [Slides!](slides/slides_lesson_1.pdf)
- [x] Lesson 2: Value/Policy Iteration [Code!](lessons/lesson_2_code.py) [Results!](results/lesson_2_results.txt) [Slides!](slides/slides_lesson_2.pdf)
- [x] Lesson 3: Monte Carlo Methods [Code!](lessons/lesson_3_code.py) [Results!](results/lesson_3_results.txt) [Slides!](slides/slides_lesson_3.pdf)
- [x] Lesson 4: Q-Learning and Sarsa [Code!](lessons/lesson_4_code.py) [Results!](results/lesson_4_results.txt) [Slides!](slides/slides_lesson_4.pdf)
- [x] Lesson 5: Dyna-Q [Code!](lessons/lesson_5_code.py) [Results!](results/lesson_5_results.txt) [Slides!](slides/slides_lesson_5.pdf)
- [x] Lesson 6: Multi-Armed Bandit [Code!](lessons/lesson_6_code.py) [Results!](results/lesson_6_results.txt) [Slides!](slides/slides_lesson_6.pdf)

**Second Semester**
- [x] Lesson 7:  TensorFlow and Neural Networks [Code!](lessons/lesson_7_code.py) [Results!](results/lesson_7_results.txt) [Slides!](slides/slides_lesson_7.pdf)
- [x] Lesson 8:  Deep Q-Network [Code!](lessons/lesson_8_code.py) [Results!](results/lesson_8_results.txt) [Slides!](slides/slides_lesson_8.pdf)
- [x] Lesson 9:  Naive Policy Gradient [Code!](lessons/lesson_9_code.py) [Results!](results/lesson_9_results.txt) [Slides!](slides/slides_lesson_9.pdf)
- [x] Lesson 10: Actor Critic Architecture (A2C) [Code!](lessons/lesson_10_code.py) [Results!](results/lesson_10_results.txt) [Slides!](slides/slides_lesson_10.pdf)
- [ ] Lesson 11: Practical Problem *Coming Soon...*
- [ ] Lesson 12: Final Activity

## Slides

## Tutorials
This repo includes a set of introductory tutorials to help accomplish the exercises. In detail, we provide the following Jupyter notebook that contains the basic instructions for the lab:
- **Tutorial 1 - Gym Environment:** [Here!](tutorials/tutorial_environment.ipynb)
- **Tutorial 2 - Neural Network and TensorFlow:** [Here!](tutorials/tutorial_tensorflow.ipynb)

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
