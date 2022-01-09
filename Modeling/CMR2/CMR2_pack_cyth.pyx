from __future__ import print_function
import mkl
mkl.set_num_threads(1)
import os
import sys
import math
import time
import json
import numpy as np
import scipy.io
from glob import glob
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport log, sqrt
cimport numpy as np
cimport cython


"""
Known differences from Lynn Lohnas's code:
1) Cycle counter starts at 0 in this code instead of 1 during leaky accumulator.
2) No empty feature vector is presented at the end of the recall period.
"""


# Credit to "senderle" for the cython random number generation functions used below. Original code can be found at:
# https://stackoverflow.com/questions/42767816/what-is-the-most-efficient-and-portable-way-to-generate-gaussian-random-numbers
@cython.cdivision(True)
cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX


@cython.cdivision(True)
cdef double random_gaussian():
    cdef double x1, x2, w
    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void assign_random_gaussian_pair(double[:] out, int assign_ix):
    cdef double x1, x2, w
    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = sqrt((-2.0 * log(w)) / w)
    out[assign_ix] = x1 * w
    out[assign_ix + 1] = x2 * 2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cython_randn(int n):
    cdef int i
    np_result = np.zeros(n, dtype='f8', order='C')
    cdef double[:] result = np_result
    for i in range(n // 2):  # Int division ensures trailing index if n is odd.
        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()

    return result


class CMR2(object):

    def __init__(self, params, pres_mat, sem_mat, source_mat=None, mode='IFR'):
        """
        Initializes a CMR2 object and prepares it to simulate the session defined
        by pres_mat.
        
        :param params: A dictionary of model parameters and settings to use for the
            simulation. Call the CMR2_pack_cyth.make_params() function to get a
            dictionary template you can fill in with your desired values.
        :param pres_mat: A 2D array specifying the ID numbers of the words that will be
            presented to the model on each trial. Row i, column j should hold the ID number
            of the jth word to be presented on the ith trial. ID numbers are assumed to
            range from 1 to N, where N is the number of words in the semantic similarity
            matrix (sem_mat). 0s are treated as padding and are ignored, allowing you to
            zero-pad pres_mat if you wish to simulate trials with varying list length.
        :param sem_mat: A 2D array containing the pairwise semantic similarities between all
            words in the word pool. The ordering of words in the similarity matrix must
            match the word ID numbers, such that the scores for word k must be located along
            row k-1 and column k-1.
        :param source_mat: If None, source coding will not be used. Otherwise, source_mat
            should be a 3D array containing source features for each presented word. The
            matrix should have one row for each trial and one column for each serial
            position, with the third dimension having length equal to the number of source
            features you wish to simulate. Cell (i, j, k) should contain the value of the
            kth source feature of the jth item presented on list i. (DEFAULT=None)
        :param mode: A string indicating the type of free recall to simulate. Set to 'IFR'
            for immediate free recall or 'DFR' for delayed recall. (DEFUALT='IFR')
        """
        ##########
        #
        # Set up model parameters and presentation data
        #
        ##########

        self.params = params  # Dictionary of model parameters
        self.pres_nos = np.array(pres_mat, dtype=np.uint16)  # Presented item ID numbers (trial x serial position)
        self.sem_mat = np.array(sem_mat, dtype=np.float32)  # Semantic similarity matrix (e.g. Word2vec, LSA, WAS)
        if source_mat is None:
            self.nsources = 0
        else:
            self.sources = np.atleast_3d(source_mat).astype(np.float32)
            self.nsources = self.sources.shape[2]
            if self.sources.shape[0:2] != self.pres_nos.shape[0:2]:
                raise ValueError('Source matrix must have the same number of rows and columns as the presented item matrix.')
        if mode not in ('IFR', 'DFR'):
            raise ValueError('Mode must be "IFR" or "DFR", not %s.' % mode)
        self.mode = mode  # Recall type -> IFR or DFR
        self.phase = None
        self.learn_while_retrieving = self.params['learn_while_retrieving'] if 'learn_while_retrieving' in self.params else False

        # Determine the number of lists and the maximum list length
        self.nlists = self.pres_nos.shape[0]
        self.max_list_length = self.pres_nos.shape[1]

        # Create arrays of sorted and unique presented (nonzero) items
        self.nonzero_mask = self.pres_nos > 0
        self.pres_nos_sorted = np.sort(self.pres_nos[self.nonzero_mask])
        self.pres_nos_unique = np.unique(self.pres_nos_sorted)

        # Convert presented item ID numbers to indexes within the feature vector
        self.pres_indexes = np.searchsorted(self.pres_nos_unique, self.pres_nos)

        # Cut down semantic matrix to contain only the items in the session
        self.sem_mat = self.sem_mat[self.pres_nos_unique - 1, :][:, self.pres_nos_unique - 1]
        # Make sure items' associations with themselves are set to 0
        np.fill_diagonal(self.sem_mat, 0)

        ##########
        #
        # Set up context and feature vectors
        #
        ##########

        # Determine number of cells in each region of the feature/context vectors
        self.nitems = len(self.pres_nos)
        self.nitems_unique = len(self.pres_nos_unique)
        self.ndistractors = self.nlists  # One distractor prior to each list
        if self.mode == 'DFR':
            self.ndistractors += self.nlists  # One extra distractor before each recall period if running DFR
        self.ntemporal = self.nitems_unique + self.ndistractors
        self.nelements = self.ntemporal + self.nsources

        # Create context and feature vectors
        self.f = np.zeros((self.nelements, 1), dtype=np.float32)
        self.c = np.zeros_like(self.f)
        self.c_old = np.zeros_like(self.f)
        self.c_in = np.zeros_like(self.f)

        ##########
        #
        # Set up weight matrices
        #
        ##########

        # Set up primacy scaling vector
        self.prim_vec = self.params['phi_s'] * np.exp(-1 * self.params['phi_d'] * np.arange(self.max_list_length)) + 1

        # Set up learning rate matrix for M_FC (dimensions are context x features)
        self.L_FC = np.empty((self.nelements, self.nelements), dtype=np.float32)
        if self.nsources == 0:
            self.L_FC.fill(self.params['gamma_fc'])
        else:
            # Temporal Context x Item Features (items reinstating their previous temporal contexts)
            self.L_FC[:self.ntemporal, :self.ntemporal] = self.params['L_FC_tftc']
            # Temporal Context x Source Features (sources reinstating previous temporal contexts)
            self.L_FC[:self.ntemporal, self.ntemporal:] = self.params['L_FC_sftc']
            # Source Context x Item Features (items reinstating previous source contexts)
            self.L_FC[self.ntemporal:, :self.ntemporal] = self.params['L_FC_tfsc']
            # Source Context x Source Features (sources reinstating previous source contexts)
            self.L_FC[self.ntemporal:, self.ntemporal:] = self.params['L_FC_sfsc']

        # Set up learning rate matrix for M_CF (dimensions are features x context)
        self.L_CF = np.empty((self.nelements, self.nelements), dtype=np.float32)
        if self.nsources == 0:
            self.L_CF.fill(self.params['gamma_cf'])
        else:
            # Item Features x Temporal Context (temporal context cueing retrieval of items)
            self.L_CF[:self.ntemporal, :self.ntemporal] = self.params['L_CF_tctf']
            # Item Features x Source Context (source context cueing retrieval of items)
            self.L_CF[:self.ntemporal, self.ntemporal:] = self.params['L_CF_sctf']
            # Source Features x Temporal Context (temporal context cueing retrieval of sources)
            self.L_CF[self.ntemporal:, :self.ntemporal] = self.params['L_CF_tcsf']
            # Source Features x Source Context (source context cueing retrieval of sources)
            self.L_CF[self.ntemporal:, self.ntemporal:] = self.params['L_CF_scsf']

        # Initialize weight matrices as identity matrices
        self.M_FC = np.identity(self.nelements, dtype=np.float32)
        self.M_CF = np.identity(self.nelements, dtype=np.float32)

        # Scale the semantic similarity matrix by s_fc (Healey et al., 2016) and s_cf (Lohnas et al., 2015)
        fc_sem_mat = self.params['s_fc'] * self.sem_mat
        cf_sem_mat = self.params['s_cf'] * self.sem_mat

        # Complete the pre-experimental associative matrices by layering on the
        # scaled semantic matrices
        self.M_FC[:self.nitems_unique, :self.nitems_unique] += fc_sem_mat
        self.M_CF[:self.nitems_unique, :self.nitems_unique] += cf_sem_mat

        # Scale pre-experimental associative matrices by 1 - gamma
        self.M_FC *= 1 - self.L_FC
        self.M_CF *= 1 - self.L_CF

        #####
        #
        # Initialize leaky accumulator and recall variables
        #
        #####

        self.ret_thresh = np.ones(self.nitems_unique, dtype=np.float32)  # Retrieval thresholds
        self.nitems_in_race = self.params['nitems_in_accumulator']  # Number of items in accumulator
        self.rec_items = []  # Recalled items from each trial
        self.rec_times = []  # Rectimes of recalled items from each trial
        
        # Calculate dt_tau and its square root based on dt
        self.params['dt_tau'] = self.params['dt'] / 1000.  
        self.params['sq_dt_tau'] = np.sqrt(self.params['dt_tau'])
 
        ##########
        #
        # Initialize variables for tracking simulation progress
        #
        ##########

        self.trial_idx = 0  # Current trial number (0-indexed)
        self.serial_position = 0  # Current serial position (0-indexed)
        self.distractor_idx = self.nitems_unique  # Current distractor index
        self.first_source_idx = self.ntemporal  # Index of the first source feature

    def run_trial(self):
        """
        Simulates an entire standard trial, consisting of the following steps:
        1) A pre-trial context shift
        2) A sequence of item presentations
        3) A pre-recall distractor (only if the mode was set to 'DFR') 
        4) A recall period
        """
        ##########
        #
        # Shift context before start of new list
        #
        ##########
        
        # On first trial, present orthogonal item that starts the system;
        # On subsequent trials, present an interlist distractor item
        # Assume source context changes at same rate as temporal between trials
        self.phase = 'pretrial'
        self.serial_position = 0
        self.beta = 1 if self.trial_idx == 0 else self.params['beta_rec_post']
        self.beta_source = 1 if self.trial_idx == 0 else self.params['beta_rec_post']
        # Treat initial source and intertrial source as an even mixture of all sources
        #source = np.zeros(self.nsources) if self.nsources > 0 else None
        source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
        self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
        self.distractor_idx += 1

        ##########
        #
        # Present items
        #
        ##########
        
        self.phase = 'encoding'
        for self.serial_position in range(self.pres_indexes.shape[1]):
            # Skip over any zero-padding in the presentation matrix in order to allow variable list length
            if not self.nonzero_mask[self.trial_idx, self.serial_position]:  
                continue
            pres_idx = self.pres_indexes[self.trial_idx, self.serial_position]
            source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
            self.beta = self.params['beta_enc']
            self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0
            self.present_item(pres_idx, source, update_context=True, update_weights=True)

        ##########
        #
        # Pre-recall distractor (if delayed free recall)
        #
        ##########

        if self.mode == 'DFR':
            self.phase = 'distractor'
            self.beta = self.params['beta_distract']
            # Assume source context changes at the same rate as temporal during distractors
            self.beta_source = self.params['beta_distract']
            # By default, treat distractor source as an even mixture of all sources
            # [If your distractors and sources are related, you should modify this so that you can specify distractor source.]
            #source = np.zeros(self.nsources) if self.nsources > 0 else None
            source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
            self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
            self.distractor_idx += 1
        
        ##########
        #
        # Recall period
        #
        ##########
        
        self.phase = 'recall'
        self.beta = self.params['beta_rec']
        # Follow Polyn et al. (2009) assumption that beta_source is the same at encoding and retrieval
        self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0
        if 'max_recalls' in self.params:  # Limit number of recalls per trial if user has specified a maximum
            self.simulate_recall(time_limit=self.params['rec_time_limit'], max_recalls=self.params['max_recalls'])
        else: 
            self.simulate_recall(time_limit=self.params['rec_time_limit'])
        self.trial_idx += 1

    def present_item(self, item_idx, source=None, update_context=True, update_weights=True):
        """
        Presents a single item (or distractor) to the model by updating the
        feature vector. Also provides options to update context and the model's
        associative matrices after presentation.
        
        :param item_idx: The index of the cell within the feature vector that
            should be activated by the presented item. If None, presents an
            empty feature vector.
        :param source: If None, no source features will be activated. If a 1D
            array, the source features in the feature vector will be set to
            match the numbers in the source array.
        :param update_context: If True, the context vector will update after
            the feature vector has been updated. If False, the feature vector
            will update but the context vector will not change.
        :param update_weights: If True, the model's weight matrices will update
            to strengthen the association between the presented item and the
            state of context at the time of presentation. If False, no learning
            will occur after item presentation.
        """
        ##########
        #
        # Activate item's features
        #
        ##########
        
        # Activate the presented item itself
        self.f.fill(0)
        if item_idx is not None:
            self.f[item_idx] = 1

        # Activate the source feature(s) of the presented item
        if self.nsources > 0 and source is not None:
            self.f[self.first_source_idx:, 0] = np.atleast_1d(source)

        ##########
        #
        # Update context
        #
        ##########

        if update_context:

            # Compute c_in
            self.c_in = np.dot(self.M_FC, self.f)
            
            # Normalize the temporal and source subregions of c_in separately
            norm_t = np.sqrt(np.sum(self.c_in[:self.ntemporal] ** 2))
            if norm_t != 0:
                self.c_in[:self.ntemporal] /= norm_t
            if self.nsources > 0:
                norm_s = np.sqrt(np.sum(self.c_in[self.ntemporal:] ** 2))
                if norm_s != 0:
                    self.c_in[self.ntemporal:] /= norm_s
            # Set beta separately for temporal and source subregions
            beta_vec = np.empty_like(self.c)
            beta_vec[:self.ntemporal] = self.beta
            beta_vec[self.ntemporal:] = self.beta_source
            
            # Calculate rho for the temporal and source subregions
            rho_vec = np.empty_like(self.c)
            c_dot_t = np.dot(self.c[:self.ntemporal].T, self.c_in[:self.ntemporal])
            rho_vec[:self.ntemporal] = math.sqrt(1 + self.beta ** 2 * (c_dot_t ** 2 - 1)) - self.beta * c_dot_t
            c_dot_s = np.dot(self.c[self.ntemporal:].T, self.c_in[self.ntemporal:])
            rho_vec[self.ntemporal:] = math.sqrt(1 + self.beta_source ** 2 * (c_dot_s ** 2 - 1)) - self.beta_source * c_dot_s

            # Update context
            self.c_old = self.c.copy()
            self.c = (rho_vec * self.c_old) + (beta_vec * self.c_in)

        ##########
        #
        # Update weight matrices
        #
        ##########

        if update_weights:
            self.M_FC += self.L_FC * np.dot(self.c_old, self.f.T)
            if self.phase == 'encoding':  # Only apply primacy scaling during encoding
                self.M_CF += self.L_CF * self.prim_vec[self.serial_position] * np.dot(self.f, self.c_old.T)
            else:
                self.M_CF += self.L_CF * np.dot(self.f, self.c_old.T)

    def simulate_recall(self, time_limit=60000, max_recalls=np.inf):
        """
        Simulate a recall period, starting from the current state of context.
        
        :param time_limit: The simulated duration of the recall period (in ms).
            Determines how many cycles of the leaky accumulator will run before
            the recall period ends. (DEFAULT=60000)
        :param max_recalls: The maximum number of retrievals (not overt recalls)
            that the model is permitted to make. If this limit is reached, the
            recall period will end early. Use this setting to prevent the model
            from eating up runtime if its parameter set causes it to make
            hundreds of recalls per trial. (DEFAULT=np.inf)
        """
        self.rec_items.append([])
        self.rec_times.append([])
        cycles_elapsed = 0
        nrecalls = 0
        max_cycles = time_limit // self.params['dt']

        while cycles_elapsed < max_cycles and nrecalls < max_recalls:
            # Use context to cue items
            f_in = np.dot(self.M_CF, self.c)[:self.nitems_unique].flatten()

            # Identify set of items with the highest activation
            top_items = np.argsort(f_in)[self.nitems_unique-self.nitems_in_race:]
            top_activation = f_in[top_items]
            top_activation[top_activation < 0] = 0

            # Run accumulator until an item is retrieved
            winner_idx, ncycles = self.leaky_accumulator(top_activation, self.ret_thresh[top_items], int(max_cycles - cycles_elapsed))
            # Update elapsed time
            cycles_elapsed += ncycles
            nrecalls += 1

            # Perform the following steps only if an item was retrieved
            if winner_idx != -1:
                
                # Identify the feature index of the retrieved item
                item = top_items[winner_idx]

                # Decay retrieval thresholds, then set the retrieved item's threshold to maximum
                self.ret_thresh = 1 + self.params['alpha'] * (self.ret_thresh - 1)
                self.ret_thresh[item] = 1 + self.params['omega']

                # Present retrieved item to the model, with no source information
                if self.learn_while_retrieving:
                    self.present_item(item, source=None, update_context=True, update_weights=True)
                else:
                    self.present_item(item, source=None, update_context=True, update_weights=False)

                # Filter intrusions using temporal context comparison, and log item if overtly recalled
                c_similarity = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal])
                if c_similarity >= self.params['c_thresh']:
                    rec_itemno = self.pres_nos_unique[item]
                    self.rec_items[-1].append(rec_itemno)
                    self.rec_times[-1].append(cycles_elapsed * self.params['dt'])

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    @cython.cdivision(True)  # Skip checks for division by zero
    def leaky_accumulator(self, float [:] in_act, float [:] x_thresholds, Py_ssize_t max_cycles):
        """
        Simulates the item retrieval process using a leaky accumulator. Loops
        until an item is retrieved or the recall period ends.
        
        :param in_act: A 1D array containing the incoming activation values
            for all items in the competition.
        :param x_thresholds: A 1D array containing the activation thresholds
            required to retrieve each item in the competition.
        :param max_cycles: The maximum number of cycles the accumulator can run
            before the recall period ends.
            
        :returns: The index of the retrieved item (or -1 if no item was
            retrieved) and the number of cycles that elapsed before retrieval.
        """
        # Set up indexes
        cdef Py_ssize_t i, j, cycle = 0
        cdef Py_ssize_t nitems_in_race = in_act.shape[0]
        
        # Set up time constants
        cdef float dt_tau = self.params['dt_tau']
        cdef float sq_dt_tau = self.params['sq_dt_tau']

        # Pre-scale decay rate (kappa) based on dt
        cdef float kappa = self.params['kappa']
        kappa *= dt_tau
        # Pre-scale inhibition (lambda) based on dt
        cdef float lamb = self.params['lamb']
        lamb *= dt_tau
        # Take sqrt(eta) and pre-scale it based on sqrt(dt_tau)
        # Note that we do this because (for cythonization purposes) we multiply the noise 
        # vector by sqrt(eta), rather than directly setting the SD to eta
        cdef float eta = self.params['eta'] ** .5
        eta *= sq_dt_tau
        # Pre-scale incoming activation based on dt
        np_in_act_scaled = np.empty(nitems_in_race, dtype=np.float32)
        cdef float [:] in_act_scaled = np_in_act_scaled
        for i in range(nitems_in_race):
            in_act_scaled[i] = in_act[i] * dt_tau

        # Set up activation variables
        np_x = np.zeros(nitems_in_race, dtype=np.float32)
        cdef float [:] x = np_x
        cdef float act
        cdef float sum_x
        cdef float delta_x
        cdef double [:] noise_vec

        # Set up winner variables
        cdef int has_retrieved_item = 0
        cdef int nwinners = 0
        np_retrieved = np.zeros(nitems_in_race, dtype=np.int32)
        cdef int [:] retrieved = np_retrieved
        cdef int [:] winner_vec
        cdef int winner
        cdef (int, int) winner_and_cycle
        
        # Loop accumulator until retrieving an item or running out of time
        while cycle < max_cycles and not has_retrieved_item:

            # Compute sum of activations for lateral inhibition
            sum_x = 0
            i = 0
            while i < nitems_in_race:
                sum_x += x[i]
                i += 1

            # Update activation and check whether any items were retrieved
            noise_vec = cython_randn(nitems_in_race)
            i = 0
            while i < nitems_in_race:
                # Note that kappa, lambda, eta, and in_act have all been pre-scaled above based on dt
                x[i] += in_act_scaled[i] + (eta * noise_vec[i]) - (kappa * x[i]) - (lamb * (sum_x - x[i]))
                x[i] = max(x[i], 0)
                if x[i] >= x_thresholds[i]:
                    has_retrieved_item = 1
                    nwinners += 1
                    retrieved[i] = 1
                    winner = i
                i += 1
            
            cycle += 1
        
        # If no items were retrieved, set winner to -1
        if nwinners == 0:
            winner = -1
        # If multiple items crossed the retrieval threshold on the same cycle, choose one randomly
        elif nwinners > 1:
            winner_vec = np.zeros(nwinners, dtype=np.int32)
            i = 0
            j = 0
            while i < nitems_in_race:
                if retrieved[i] == 1:
                    winner_vec[j] = i
                    j += 1
                i += 1
            srand(time.time())
            rand_idx = rand() % nwinners  # see http://www.delorie.com/djgpp/doc/libc/libc_637.html
            winner = winner_vec[rand_idx]
        # If only one item crossed the retrieval threshold, we already set it as the winner above

        # Return winning item's index within in_act, as well as the number of cycles elapsed
        winner_and_cycle = (winner, cycle)
        return winner_and_cycle


##########
#
# Code to load data and run model
#
##########

def make_params(source_coding=False):
    """
    Returns a dictionary containing all parameters that need to be defined in
    order for CMR2 to run. Can be used as a template for the "params" input
    required by CMR2, run_cmr2_single_sess(), and run_cmr2_multi_sess().
    For notes on each parameter, see in-line comments.
    
    :param source_coding: If True, parameter dictionary will contain the
        parameters required for the source coding version of the model. If
        False, the dictionary will only condain parameters required for the
        base version of the model.
    
    :returns: A dictionary containing all of the parameters you need to define
        to run CMR2.
    """
    param_dict = {
        # Beta parameters
        'beta_enc': None,  # Beta encoding
        'beta_rec': None,  # Beta recall
        'beta_rec_post': None,  # Beta post-recall
        'beta_distract': None,  # Beta for distractor task
        
        # Primacy and semantic scaling
        'phi_s': None,
        'phi_d': None,
        's_cf': None,  # Semantic scaling in context-to-feature associations
        's_fc': 0,  # Semantic scaling in feature-to-context associations (Defaults to 0)
        
        # Recall parameters
        'kappa': None,
        'eta': None,
        'omega': None,
        'alpha': None,
        'c_thresh': None,
        'lamb': None,
        
        # Timing & recall settings
        'rec_time_limit': 60000.,  # Duration of recall period (in ms) (Defaults to 60000)
        'dt': 10,  # Number of milliseconds to simulate in each loop of the accumulator (Defaults to 10)
        'nitems_in_accumulator': 50,  # Number of items in accumulator (Defaults to 50)
        'max_recalls': 50,  # Maximum recalls allowed per trial (Defaults to 50)
        'learn_while_retrieving': False  # Whether associations should be learned during recall (Defaults to False)
    }
    
    # If not using source coding, set up 2 associative scaling parameters (gamma)
    if not source_coding:
        param_dict['gamma_fc'] = None  # Gamma FC
        param_dict['gamma_cf'] = None  # Gamma CF
        
    # If using source coding, add an extra beta parameter and set up 8 associative scaling parameters
    else:
        param_dict['beta_source'] = None  # Beta source
        
        param_dict['L_FC_tftc'] = None  # Scale of items reinstating past temporal contexts (Recommend setting to gamma FC)
        param_dict['L_FC_sftc'] = 0  # Scale of sources reinstating past temporal contexts (Defaults to 0)
        param_dict['L_FC_tfsc'] = None  # Scale of items reinstating past source contexts (Recommend setting to gamma FC)
        param_dict['L_FC_sfsc'] = 0  # Scale of sources reinstating past source contexts (Defaults to 0)
        
        param_dict['L_CF_tctf'] = None  # Scale of temporal context cueing past items (Recommend setting to gamma CF)
        param_dict['L_CF_sctf'] = None  # Scale of source context cueing past items (Recommend setting to gamma CF or fitting as gamma source)
        param_dict['L_CF_tcsf'] = 0  # Scale of temporal context cueing past sources (Defaults to 0, since model does not recall sources)
        param_dict['L_CF_scsf'] = 0  # Scale of source context cueing past sources (Defaults to 0, since model does not recall sources)
    
    return param_dict


def load_pres(path):
    """
    Loads matrix of presented items from a .txt file, a .json behavioral data,
    file, or a .mat behavioral data file. Uses numpy's loadtxt function, json's
    load function, or scipy's loadmat function, respectively.

    :param path: The path to a .txt, .json, or .mat file containing a matrix
        where item (i, j) is the jth word presented on trial i.
    
    :returns: A 2D array of presented items.
    """
    if os.path.splitext(path) == '.txt':
        data = np.loadtxt(path)
    elif os.path.splitext(path) == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
            data = data['pres_nos'] if 'pres_nos' in data else data['pres_itemnos']
    elif os.path.splitext(path) == '.mat':
        data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)['data'].pres_itemnos
    else:
        raise ValueError('Can only load presented items from .txt, .json, and .mat formats.')
    return np.atleast_2d(data)


def split_data(pres_mat, identifiers, source_mat=None):
    """
    If data from multiple subjects or sessions are in one matrix, separate out
    the data into separate presentation and source matrices for each unique
    identifier.

    :param pres_mat: A 2D array of presented items from multiple consolidated
        subjects or sessions.
    :param identifiers: A 1D array with length equal to the number of rows in
        pres_mat, where entry i identifies the subject/session/etc. to which
        row i of the presentation matrix belongs.
    :param source_mat: (Optional) A trials x serial positions x nsources array of
        source information for each presented item in pres_mat.
    
    :returns: A list of presented item matrices (one matrix per unique
        identifier), an array of the unique identifiers, and a list of source
        information matrices (one matrix per subject, None if no source_mat provided).
    """
    # Make sure input matrices are numpy arrays
    pres_mat = np.array(pres_mat)
    if source_mat is not None:
        source_mat = np.atleast_3d(source_mat)

    # Get list of unique IDs
    unique_ids = np.unique(identifiers)

    # Split data up by each unique identifier
    data = []
    sources = None if source_mat is None else []
    for i in unique_ids:
        mask = identifiers == i
        data.append(pres_mat[mask, :])
        if source_mat is not None:
            sources.append(source_mat[mask, :, :])

    return data, unique_ids, sources


def run_cmr2_single_sess(params, pres_mat, sem_mat, source_mat=None, mode='IFR'):
    """
    Simulates a single session of free recall using the specified parameter set.
    
    :param params: A dictionary of model parameters and settings to use for the
        simulation. Call the CMR2_pack_cyth.make_params() function to get a
        dictionary template you can fill in with your desired values.
    :param pres_mat: A 2D array specifying the ID numbers of the words that will be
        presented to the model on each trial. Row i, column j should hold the ID number
        of the jth word to be presented on the ith trial. ID numbers are assumed to
        range from 1 to N, where N is the number of words in the semantic similarity
        matrix (sem_mat). 0s are treated as padding and are ignored, allowing you to
        zero-pad pres_mat if you wish to simulate trials with varying list length.
    :param sem_mat: A 2D array containing the pairwise semantic similarities between all
        words in the word pool. The ordering of words in the similarity matrix must
        match the word ID numbers, such that the scores for word k must be located along
        row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used. Otherwise, source_mat
        should be a 3D array containing source features for each presented word. The
        matrix should have one row for each trial and one column for each serial
        position, with the third dimension having length equal to the number of source
        features you wish to simulate. Cell (i, j, k) should contain the value of the
        kth source feature of the jth item presented on list i. (DEFAULT=None)
    :param mode: A string indicating the type of free recall to simulate. Set to 'IFR'
        for immediate free recall or 'DFR' for delayed recall. (DEFUALT='IFR')
    
    :returns: Two 2D arrays. The first contains the ID numbers of the items the model
        recalled on each trial. The second contains the response times of each of
        those items relative to the start of the recall period.
    """
    ntrials = pres_mat.shape[0]

    # Simulate all trials of the session using CMR2
    cmr = CMR2(params, pres_mat, sem_mat, source_mat=source_mat, mode=mode)
    for i in range(ntrials):
        cmr.run_trial()

    # Get the model's simulated recall data
    rec_items = cmr.rec_items
    rec_times = cmr.rec_times

    # Identify the max number of recalls made on any trial
    max_recalls = max([len(trial_data) for trial_data in rec_times])

    # Zero-pad response data into an ntrials x max_recalls matrix
    rec_mat = np.zeros((ntrials, max_recalls), dtype=int)
    time_mat = np.zeros((ntrials, max_recalls))
    for i, trial_data in enumerate(rec_items):
        trial_nrec = len(trial_data)
        if trial_nrec > 0:
            rec_mat[i, :trial_nrec] = rec_items[i]
            time_mat[i, :trial_nrec] = rec_times[i]
    
    return rec_mat, time_mat


def run_cmr2_multi_sess(params, pres_mat, identifiers, sem_mat, source_mat=None, mode='IFR'):
    """
    Simulates multiple sessions of free recall using a single set of parameters.
    
    :param params: A dictionary of model parameters and settings to use for the
        simulation. Call the CMR2_pack_cyth.make_params() function to get a
        dictionary template you can fill in with your desired values.
    :param pres_mat: A 2D array specifying the ID numbers of the words that will be
        presented to the model on each trial. Row i, column j should hold the ID number
        of the jth word to be presented on the ith trial. ID numbers are assumed to
        range from 1 to N, where N is the number of words in the semantic similarity
        matrix (sem_mat). 0s are treated as padding and are ignored, allowing you to
        zero-pad pres_mat if you wish to simulate trials with varying list length.
    :param identifiers: A 1D array of session numbers, subject IDs, or other values
        indicating how the rows/trials in pres_mat and source_mat should be divided up
        into sessions. For example, one could simulate two four-trial sessions by
        setting identifiers to np.array([0, 0, 0, 0, 1, 1, 1, 1]), specifying that the
        latter four trials come from a different session than the first four trials.
    :param sem_mat: A 2D array containing the pairwise semantic similarities between all
        words in the word pool. The ordering of words in the similarity matrix must
        match the word ID numbers, such that the scores for word k must be located along
        row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used. Otherwise, source_mat
        should be a 3D array containing source features for each presented word. The
        matrix should have one row for each trial and one column for each serial
        position, with the third dimension having length equal to the number of source
        features you wish to simulate. Cell (i, j, k) should contain the value of the
        kth source feature of the jth item presented on list i. (DEFAULT=None)
    :param mode: A string indicating the type of free recall to simulate. Set to 'IFR'
        for immediate free recall or 'DFR' for delayed recall. (DEFUALT='IFR')
    
    :returns: Two 2D arrays. The first contains the ID numbers of the items the model
        recalled on each trial. The second contains the response times of each of
        those items relative to the start of the recall period.
    """
    now_test = time.time()

    # Split data based on identifiers provided
    pres, unique_ids, sources = split_data(pres_mat, identifiers, source_mat=source_mat)

    # Run CMR2 for each subject/session
    rec_items = []
    rec_times = []
    for i, sess_pres in enumerate(pres):
        sess_sources = None if sources is None else sources[i]
        out_tuple = run_cmr2_single_sess(params, sess_pres, sem_mat, source_mat=sess_sources, mode=mode)
        rec_items.append(out_tuple[0])
        rec_times.append(out_tuple[1])
    # Identify the maximum number of recalls made in any session
    max_recalls = max([sess_data.shape[1] for sess_data in rec_items])

    # Zero-pad response data into an total_trials x max_recalls matrix where rows align with those in the original data_mat
    total_trials = len(identifiers)
    rec_mat = np.zeros((total_trials, max_recalls), dtype=int)
    time_mat = np.zeros((total_trials, max_recalls))
    for i, uid in enumerate(unique_ids):
        sess_max_recalls = rec_items[i].shape[1]
        if sess_max_recalls > 0:
            rec_mat[identifiers == uid, :sess_max_recalls] = rec_items[i]
            time_mat[identifiers == uid, :sess_max_recalls] = rec_times[i]
    
    print("CMR Time: " + str(time.time() - now_test))
    
    return rec_mat, time_mat
