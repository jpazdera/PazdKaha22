  ____   __  __   ____    ____  
 / ___| |  \/  | |  _ \  |___ \ 
| |     | |\/| | | |_) |   __) |
| |___  | |  | | |  _ <   / __/ 
 \____| |_|  |_| |_| \_\ |_____|
================================

FILES:

Setup files and the core CMR2 code can be found in the top-level directory.
These include the following:
1) CMR2_pack_cyth.pyx - Contains the Cython implementation of CMR2.
2) setup_cmr2.py - A script which can be used to build and install CMR2_pack_cyth.
3) setup_env.sh - A script which can be used to create a new Python 3 environment
	able to run CMR2.

Code for fitting CMR2 can be found in the fitting/ subdirectory:
4) fitting/pso_cmr2_ltpFR3.py - Contains a Python function for running particle
	swarm optimization on CMR2.
5) fitting/optimization_utils.py - Contains a variety of utility functions used by the
	optimization algorithms, including the function that evaluates each parameter
	set by calling CMR2 and comparing its simulated behavior to the target
	behavior.
6) fitting/noise_maker_pso.py - Contains a function used by the PSO algorithm to generate
	random numbers used in particle initialization and motion.

Semantic similarity matrices and their associated wordpools are stored as text
files in the wordpools/ subdirectory:
7) wordpools/PEERS_wordpool.txt - Contains a list of the 1638 words in the PEERS
	wordpool. Word ID numbers correspond to the ordering of words in this list,
	from 1 to 1638.
8) wordpools/PEERS_w2v.txt - Contains a matrix of semantic similarities between
	all pairs of words in the PEERS wordpool, determined as the cosine similarity
	between the words' representations in word2vec space. Word order matches that
	used in PEERS_wordpool.txt.

-------------------------------------------------------------------------------

SETUP:

1A) If you do not already have a Python 3 environment set up, use the provided
shell script to set up an environment, build and install CMR2 and its required
packages, and set up a Jupyter kernel for the environment:

bash setup_env.sh


1B) If you already have a Python 3 environment in Anaconda, simply activate it 
(replace ENV_NAME with the name of your Python 3 environment), make sure
CMR2's dependencies are installed, and then run its installation script:

source activate ENV_NAME
conda install numpy scipy matplotlib mkl cython
python setup_cmr2.py install


2) Regardless of which of the two methods you used to install CMR, you should
now be able to import it into Python from anywhere using the following line:

import CMR2_pack_cyth as cmr

Once you have imported it, you can use the functions from the package just
like you would use any other package in Python. Note that the supplementary
files containing optimization algorithms will not be installed as part of the
package. If you wish to use these scripts, you will need to copy these files,
modify them as needed to work with your research project, and then run the
files directly (see below).


3) Any time you change or update CMR2_pack_cyth, you will need to rerun the
following line in order to rebuild the code and update your installation:

python setup_cmr2.py install 


4) If you wish to use the model fitting algorithms, you will also need to set
up a couple scripts that will help you test multiple models in parallel. These
files are located in the pyCMR2/pgo_files/ directory, not in pyCMR2/CMR2_Optimized.
If you do not already have a bin folder in your home directory, create one:

mkdir ~/bin

Then copy the two scripts from pgo_files (pgo and runpyfile.sh) into ~/bin.
Finally, edit the following line in your copy of runpyfile.sh, replacing 
ENV_NAME with the name of your Python 3 environment. If your Anaconda folder
is named something other than anaconda3, you will also need to edit the file
path to match the name of your Anaconda folder:

PY_COMMAND="/home1/$USER/anaconda3/envs/base/bin/python"

You should now be able to call the pgo function in your terminal from anywhere.
See the instructions below on how to use pgo in conjunction with the model
fitting algorithms to optimize your model fits.

-------------------------------------------------------------------------------

RUNNING CMR2:

Once you have imported CMR2_pack_cyth, there are three ways you can run
simulations with CMR. The first is through the run_cmr2_single_sess() function,
which allows you to simulate a single session using a given parameter set. The
second is through the run_cmr2_multi_sess() function, which allows you to
simulate multiple sessions using a single parameter set. The third method is to
initialize a CMR2 object and then build your own simulation code around it,
rather than using one of the two provided functions.

A couple helpful tips before we begin:
- For the "params" input to the functions below, you can use the function
CMR2_pack_cyth.make_params() to create a dictionary
containing all of the parameters that must be included in the params input to
the functions below. Just take the dictionary template it creates and fill in
your desired settings.
- For the "pres_mat" input to the functions below, you can use the
CMR2_pack_cyth.load_pres() function to load the presented item matrix from a
variety of data files, including the behavioral matrix files from ltp studies
in .json and .mat format.
- For the "sem_mat" inputs to the functions below, if your simulation uses one
of the wordpools provided in the wordpools subdirectory, you should be able to
find a text file in there containing the semantic similarity matrix for that
wordpool (e.g. PEERS_w2v.txt). Simply load it with np.loadtxt() and input it as
your similarity matrix.

Here are the three ways to run simulations with CMR2:
1) run_cmr2_single_sess(params, pres_mat, sem_mat, source_mat=None, mode='IFR')

This function simulates a single session with CMR2 and returns two numpy arrays
containing the ID numbers of recalled items and the response times of each of
those recalls, respectively. See the function's docstring for details on its
inputs, and note that you can choose whether to include source features
in your model and whether to simulate immediate or delayed free recall.

2) run_cmr2_multi_sess(params, pres_mat, identifiers, sem_mat, source_mat=None, mode='IFR')

This function simulates multiple sessions with CMR2, using the same parameter
set for all sessions. Like its single-session counterpart it returns two numpy
arrays containing the ID numbers of recalled items and the response times of
each of those recalls. See the function's docstring for details on its
inputs, and again note that you can choose whether to include source features
in your model and whether to simulate immediate or delayed free recall.

3) CMR2(params, pres_mat, sem_mat, source_mat=None, mode='IFR')

The inputs when creating a CMR2 object are identical to those you would provide
when using run_cmr2_single_sess(). Indeed, that function simply creates a CMR2
object and calls methods of the class in order to simulate each trial and
organize the results into recall and response time matrices. If desired, you
can work with the CMR2 object directly rather than using one of the wrapper
functions provided. You can then directly call the following methods of the
CMR2 class:

- run_trial(): Simulates an entire standard trial, consisting of the following steps:
        1) A pre-trial context shift
        2) A sequence of item presentations
        3) A pre-recall distractor (only if the mode was set to 'DFR') 
        4) A recall period
- present_item(item_idx, source=None, update_context=True, update_weights=True):
	Presents a single item (or distractor) to the model. This includes options
	for setting the source features of the item, as well as for choosing whether
	the context vector and/or weight matrices should be updated after presenting
	the item.
- simulate_recall(time_limit=60000, max_recalls=np.inf): Simulates a recall period
	with a duration of time_limit miliseconds. If the model makes max_recalls 
	retrievals, the recall period will terminate early (this can be used to
	avoid wasting time running models that make hundreds of recalls per trial).

It is NOT RECOMMENDED to manually run present_item() and simulate_recall()
unless you have read the code for run_trial() and understand the fields in the
CMR2 object that need to be updated over the course of a trial. Manually
running run_trial() is perfectly fine, however.

-------------------------------------------------------------------------------

FITTING CMR2:

In order to fit CMR2 to a set of data, you will need to use some type of
optimization algorithm to search the paramaeter space and evaluate the goodness
of fit of different parameter sets. Although you can choose to use any
algorithm you like, included is a particle swarm algorithm that has been used in
previous work. It can be found in the fitting/ subdirectory. As the name of
the file implies, the provided code was specialized for fitting ltpFR3. In order
to make use of these functions, you will need to make copies of them and
customize them for your purposes. You can find additional code in
optimization_utils.py to help you design your actual goodness-of-fit test,
score your model's recall performance, and more.

Regardless of which algorithm you are using, you can run your optimization jobs
in parallel by running the following commands. First, make sure your Python 3
environment is active (source activate EV_NAME), then run:

pgo FILENAME N_JOBS

Where FILENAME is the optimization algorithm's file path and N_JOBS is the
number of parallel jobs you wish to run. Remember you can view your jobs at any
time with qstat and can manually kill jobs using qdel. Please cluster responsibly.

In the PSO script, parallel instances will automatically track one another's 
progress to make sure the next iteration starts once all jobs have finished 
evaluating the current iteration. You therefore only need to run pgo once, 
rather than once per iteration.

IMPORTANT: The particl swarm leaves behind many files tracking intermediate steps
of the algorithm. Once the algorithm has finished, remember to delete all
tempfiles and keep only the files with the goodness of fit scores and parameter
values from each iteration.

1) pso_cmr2_ltpFR3.py: A particle swarm optimization algorithm. Includes
implementations of many different particle swarm variants (see its docstring
for reference). Particle swarms are designed to test small numbers of parameter
sets for hundreds/thousands of iterations. Parameter sets within an iteration
can be tested in parallel. Each new iteration cannot start until all parameter
sets from the current iteration have finished.
