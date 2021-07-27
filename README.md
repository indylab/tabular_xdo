# Tabular XDO

# Installation

(Tested on Ubuntu 20.04)

### Overview

1. Clone the repo
2. Set up a Conda env
3. Install OpenSpiel

### Clone repo with git submodules

```shell script
git clone --recursive git@github.com:indylab/tabular_xdo.git
cd tabular_xdo
```

If you've already cloned this repo but not the [submodules](/dependencies), you can clone them with:

```shell script
git submodule update --init --recursive
```

### Set up Conda environment

After installing [Anaconda](https://docs.anaconda.com/anaconda/install/), enter the repo directory and create the new
environment:

```shell script
conda env create -f environment.yml
conda activate grl
```

### Install Python modules

#### 1. DeepMind OpenSpiel (included dependency)

DeepMind's [OpenSpiel](https://github.com/deepmind/open_spiel) is used for poker game logic as well as tabular game
utilities.

```shell script
# Starting from the repo root
cd dependencies/open_spiel
export BUILD_WITH_ACPC=ON # to compile with the optional universal poker game variant
./install.sh
pip install -e . # This will start a compilation process. Will take a few minutes.
cd ../..
```

Installation is now done!

### Advanced Installation Notes (Optional)

If you need to compile/recompile OpenSpiel without pip installing it. Perform the following steps with your conda env *
active*. (The conda env needs to be active so that OpenSpiel can find and compile against the python development headers
in the env. Python version related issues may occur otherwise):

```shell script
export BUILD_WITH_ACPC=ON # to compile with the optional universal poker game variant
mkdir build
cd build
CXX=clang++ cmake -DPython_TARGET_VERSION=3.6 -DCMAKE_CXX_COMPILER=${CXX} -DPython3_FIND_VIRTUALENV=FIRST -DPython3_FIND_STRATEGY=LOCATION ../open_spiel
make -j$(nproc)
cd ../../..
```

To import OpenSpiel without using pip, add OpenSpiel directories to your PYTHONPATH in your ~
/.bashrc ([more details here](https://github.com/deepmind/open_spiel/blob/244d1b55eb3f9de2ab4a0e06341ff2847afea466/docs/install.md)):

```shell script
# Add the following lines to your ~/.bashrc:
# For the python modules in open_spiel.
export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel_submodule>
# For the Python bindings of Pyspiel
export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel_submodule>/build/python
```
# Running Experiments
To replicate figure 3.a, run the following commands:

```shell script
python xdo_psro_fixed_population_comparison.py --num_strats 2
python xdo_psro_fixed_population_comparison.py --num_strats 10
python xdo_psro_fixed_population_comparison.py --num_strats 20
python xdo_psro_fixed_population_comparison.py --num_strats 50
python xdo_psro_fixed_population_comparison.py --num_strats 100
python xdo_psro_fixed_population_comparison.py --num_strats 300
python xdo_psro_fixed_population_comparison.py --num_strats 1000
```

To replicate figure 3.b, run the following commands:

```shell script
python main_experiments.py --algorithm xdo --game_name leduc_poker
python main_experiments.py --algorithm psro --game_name leduc_poker
```

To replicate figure 3.c and figure 4.a, run the following commands:

```shell script
python main_experiments.py --algorithm xdo --game_name leduc_poker_dummy
python main_experiments.py --algorithm psro --game_name leduc_poker_dummy
python main_experiments.py --algorithm cfr --game_name leduc_poker_dummy
python main_experiments.py --algorithm xfp --game_name leduc_poker_dummy
```





