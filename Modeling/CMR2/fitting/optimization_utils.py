import json
import numpy as np
import scipy.stats as ss
import CMR2_pack_cyth as CMR2
from pybeh.spc import spc
from pybeh.pfr import pfr
from pybeh.pli import pli
from pybeh.temp_fact import temp_fact
from pybeh.dist_fact import dist_fact
from pybeh.make_recalls_matrix import make_recalls_matrix
from pybeh.create_intrusions import intrusions


def filter_by_condi(a, mods, prs, lls, dds, mod=None, pr=None, ll=None, dd=None):

    if pr == 's':
        pr = 1600
    elif pr == 'f':
        pr = 800

    ll = int(ll) if ll is not None else None
    dd = int(dd) if dd is not None else None

    ind = [i for i in range(len(a)) if ((ll is None or lls[i] == ll) and (pr is None or prs[i] == pr) and (mod is None or mods[i] == mod) and (dd is None or dds[i] == dd))]
    if len(ind) == 0:
        return np.array([])
    return np.array(a)[ind]


def pad_into_array(l, min_length=0):
    """
    Turn an array of uneven lists into a numpy matrix by padding shorter lists with zeros. Modified version of a
    function by user Divakar on Stack Overflow, here:
    http://stackoverflow.com/questions/32037893/numpy-fix-array-with-rows-of-different-lengths-by-filling-the-empty-elements-wi

    :param l: A list of lists
    :return: A numpy array made from l, where all rows have been made the same length via padding
    """
    l = np.array(l)
    # Get lengths of each row of data
    lens = np.array([len(i) for i in l])

    # If l was empty, we can simply return the empty numpy array we just created
    if len(lens) == 0:
        return lens

    # If all rows are the same length, return the original input as an array
    if lens.max() == lens.min() and lens.max() >= min_length:
        return l

    # Mask of valid places in each row
    mask = np.arange(max(lens.max(), min_length)) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=l.dtype)
    out[mask] = np.concatenate(l)

    return out


def get_data(data_files, wordpool_file, fixed_length=False):

    data_pres = np.empty((len(data_files), 16, 24), dtype='U32')  # Session x trial x serial position
    sources = np.zeros((len(data_files), 16, 24, 2))  # Session x trial x serial position x modality
    for i, data_file in enumerate(data_files):
        with open(data_file, 'r') as f:
            x = json.load(f)

        # Get words presented in this session (drop practice lists)
        data_pres[i, :, :] = np.array(x['pres_words'])[2:, :]
        # Get modality info
        for j, mod in enumerate(x['pres_mod'][2:]):
            sources[i, j, :, int(mod == 'v')] = 1

    # If fixed list length is activated, truncate lists to length 12 (or a given integer)
    if isinstance(fixed_length, int):
        if 0 < fixed_length <= 12:
            data_pres = data_pres[:, :, :fixed_length]
            sources = sources[:, :, :fixed_length, :]
        else:
            return ValueError('Invalid fixed list length. Must be between 1 and 12.')
    elif fixed_length == True:
        data_pres = data_pres[:, :, :12]
        sources = sources[:, :, :12, :]
    
    # Replace zeros with empty strings
    data_pres[data_pres == '0'] = ''

    # Get PEERS word pool
    wp = [s.lower() for s in np.loadtxt(wordpool_file, dtype='U32')]

    # Convert presented words to word ID numbers
    data_pres = np.searchsorted(wp, data_pres, side='right')
    
    # Create session indices
    sessions = []
    for n, sess_pres in enumerate(data_pres):
        sessions.append([n for _ in sess_pres])
    sessions = np.array(sessions)
    sessions = sessions.flatten()
    
    # Collapse sessions and trials of presented items into one dimension
    data_pres = data_pres.reshape((data_pres.shape[0] * data_pres.shape[1], data_pres.shape[2]))
    sources = sources.reshape((sources.shape[0] * sources.shape[1], sources.shape[2], sources.shape[3]))
        
    return data_pres, sessions, sources


def calc_spc(recalls, sessions, return_sem=False, listLength=12):

    s = spc(recalls, subjects=sessions, listLength=listLength)
    s_start = spc(recalls, subjects=sessions, listLength=listLength, start_position=[1])
    s_l4 = spc(recalls, subjects=sessions, listLength=listLength, start_position=listLength-np.arange(0, 4))

    if return_sem:
        return np.nanmean(s, axis=0), ss.sem(s, axis=0, nan_policy='omit'), \
               np.nanmean(s_start, axis=0), ss.sem(s_start, axis=0, nan_policy='omit'), \
               np.nanmean(s_l4, axis=0), ss.sem(s_l4, axis=0, nan_policy='omit')
    else:
        return s.mean(axis=0), np.nanmean(s_start, axis=0), np.nanmean(s_l4, axis=0)


def calc_pfr(recalls, sessions, return_sem=False, listLength=12):

    s = np.array(pfr(recalls, subjects=sessions, listLength=listLength))

    if return_sem:
        return s.mean(axis=0), ss.sem(s, axis=0)
    else:
        return s.mean(axis=0)


def calc_pli(intrusions, sessions, return_sem=False):

    s = np.array(pli(intrusions, subjects=sessions, per_list=True))

    if return_sem:
        return np.mean(s), ss.sem(s)
    else:
        return np.mean(s)


def pli_recency(intrusions, sessions, nmax=5, nskip=2, return_sem=False):
    
    u_sess = np.unique(sessions)
    n_sess = len(u_sess)

    result = np.zeros((n_sess, nmax))

    for i, sess in enumerate(u_sess):
        sess_intru = intrusions[sessions == sess]
        n_trials = len(sess_intru)
        pli_counts = np.zeros(n_trials-1)
        possible_counts = np.arange(n_trials-1, 0, -1)
        for trial, trial_data in enumerate(sess_intru):
            if trial < nskip:
                continue
            for item in trial_data:
                if item > 0:
                    pli_counts[item-1] += 1
        normed_counts = pli_counts / possible_counts
        result[i, :] = normed_counts[:nmax] / np.nansum(normed_counts)

    if return_sem:
        return np.nanmean(result, axis=0), ss.sem(result, axis=0, nan_policy='omit')
    else:
        return np.nanmean(result, axis=0)

    
def calc_temp_fact(recalls, sessions, listLength=12, skip_first_n=0, return_sem=False):
    
    s = temp_fact(recalls, sessions, listLength=listLength, skip_first_n=skip_first_n)
    
    if return_sem:
        return np.nanmean(s, axis=0), ss.sem(s, axis=0, nan_policy='omit')
    else:
        return np.nanmean(s, axis=0)


def calc_sem_fact(rec_nos, pres_nos, sessions, dist_mat, skip_first_n=0, return_sem=False):
    
    s = dist_fact(rec_nos, pres_nos, sessions, dist_mat, is_similarity=True, skip_first_n=skip_first_n)
    
    if return_sem:
        return np.nanmean(s, axis=0), ss.sem(s, axis=0, nan_policy='omit')
    else:
        return np.nanmean(s, axis=0)
    

def param_vec_to_dict(param_vec):
    """
    Convert parameter vector to dictionary format expected by CMR2.
    """
    # Generate a base paramater dictionary
    param_dict = CMR2.make_params(source_coding=False)
    
    # Set parameter values
    param_dict['beta_enc'] = param_vec[0]
    param_dict['beta_rec'] = param_vec[1]
    param_dict['gamma_fc'] = param_vec[2]
    param_dict['gamma_cf'] = param_vec[3]
    param_dict['phi_s'] = param_vec[4]
    param_dict['phi_d'] = param_vec[5]
    param_dict['kappa'] = param_vec[6]
    param_dict['eta'] = param_vec[7]
    param_dict['s_cf'] = param_vec[8]
    param_dict['beta_rec_post'] = param_vec[9]
    param_dict['omega'] = param_vec[10]
    param_dict['alpha'] = param_vec[11]
    param_dict['c_thresh'] = param_vec[12]
    param_dict['lamb'] = param_vec[13]
    
    # If using source coding
    #param_dict['L_FC_tftc'] = param_vec[2]
    #param_dict['L_FC_tfsc'] = param_vec[2]
    #param_dict['L_CF_tctf'] = param_vec[3]
    #param_dict['beta_source'] = param_vec[14]
    #param_dict['L_CF_sctf'] = param_vec[15]
    
    # Define recall period settings
    param_dict['dt'] = 250.
    param_dict['rec_time_limit'] = 60000.
    param_dict['nitems_in_accumulator'] = 48
    param_dict['max_recalls'] = 30
    param_dict['learn_while_retrieving'] = False

    return param_dict


def obj_func(param_vec, target_stats, data_pres, sessions, w2v, source_mat=None, return_recalls=False, is_sim1c=False):
    
    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec)
    
    # Run model with the parameters given in param_vec
    rec_nos, rts = CMR2.run_cmr2_multi_sess(param_dict, data_pres, sessions, w2v, source_mat=source_mat, mode='IFR')
    
    # Create recalls and intrusion matrices
    rec_nos = pad_into_array(rec_nos, min_length=1)
    cmr_recalls = make_recalls_matrix(pres_itemnos=data_pres, rec_itemnos=rec_nos)
    cmr_intrusions = intrusions(pres_itemnos=data_pres, rec_itemnos=rec_nos, subjects=np.zeros(rec_nos.shape[0]),
                                sessions=sessions)
    
    # Get the performance stats of the model's predicted recalls
    cmr_stats = dict(
        spc={},
        spc_sem={},
        spc_fr1={},
        spc_fr1_sem={},
        spc_frl4={},
        spc_frl4_sem={},
        pfr={},
        pfr_sem={},
        temp_fact={},
        temp_fact_sem={},
        sem_fact={},
        sem_fact_sem={}
    )
    list_lengths = np.sum(data_pres > 0, axis=1)
    for ll in np.unique(list_lengths):
        ll_string = str(ll)
        length_mask = list_lengths == ll
        recalls = cmr_recalls[length_mask, :]
        recno = rec_nos[length_mask, :]
        presno = data_pres[length_mask, :]
        sess = sessions[length_mask]
        cmr_stats['spc'][ll_string], cmr_stats['spc_sem'][ll_string],\
        cmr_stats['spc_fr1'][ll_string], cmr_stats['spc_fr1_sem'][ll_string],\
        cmr_stats['spc_frl4'][ll_string], cmr_stats['spc_frl4_sem'][ll_string] = calc_spc(recalls, sess, return_sem=True, listLength=ll)
        cmr_stats['pfr'][ll_string], cmr_stats['pfr_sem'][ll_string] = calc_pfr(recalls, sess, return_sem=True, listLength=ll)
        cmr_stats['temp_fact'][ll_string], cmr_stats['temp_fact_sem'][ll_string] = calc_temp_fact(recalls, sess, listLength=ll, skip_first_n=2, return_sem=True)
        cmr_stats['sem_fact'][ll_string], cmr_stats['sem_fact_sem'][ll_string] = calc_sem_fact(recno, presno, sess, w2v, skip_first_n=2, return_sem=True)
    cmr_stats['plis'], cmr_stats['plis_sem'] = calc_pli(cmr_intrusions, sessions, return_sem=True)
    cmr_stats['pli_recency'], cmr_stats['pli_recency_sem'] = pli_recency(cmr_intrusions, sessions, nmax=5, nskip=2, return_sem=True)
    
    # If no PLIs were made, score PLI recency as all zeros instead of NaNs
    cmr_stats['pli_recency'][np.isnan(cmr_stats['pli_recency'])] = 0
    
    # If the model never made more than two valid transitions, treat its clustering scores as 0 instead of NaN
    for stat in ('temp_fact', 'sem_fact'):
        for ll in cmr_stats[stat]:
            if np.isnan(cmr_stats[stat][ll]):
                cmr_stats[stat][ll] = 0
    
    # If the model never initiated recall from serial position 1 or from one of the final 4 positions, treat
    # that SPC as all zeros rather than all NaNs
    for stat in ('spc_fr1', 'spc_frl4'):
        for ll in cmr_stats[stat]:
            if np.any(np.isnan(cmr_stats[stat][ll])):
                cmr_stats[stat][ll].fill(0)

    # Score the model's behavioral stats as compared with the true data
    if is_sim1c:
        err = sim1c_error(target_stats, cmr_stats)
    else:
        err = chi_squared_error(target_stats, cmr_stats)

    cmr_stats['err'] = err
    cmr_stats['params'] = param_vec
    
    if return_recalls:
        return err, cmr_stats, cmr_recalls
    else:
        return err, cmr_stats


def chi_squared_error(target_stats, cmr_stats):
    
    y = []
    y_sem = []
    y_hat = []
    
    # Fit SPC and PFR
    for stat in ('spc_fr1', 'spc_frl4', 'pfr'):
        for ll in cmr_stats[stat]:
            # Skip serial position 1 for SPC when initiating recall from position 1 
            # (to avoid dividing by 0 standard error, since prec is always 1 by definition)
            if stat == 'spc_fr1':
                y.append(np.atleast_1d(target_stats[stat][ll][1:]))
                y_sem.append(np.atleast_1d(target_stats[stat + '_sem'][ll][1:]))
                y_hat.append(np.atleast_1d(cmr_stats[stat][ll][1:]))
            else:
                y.append(np.atleast_1d(target_stats[stat][ll]))
                y_sem.append(np.atleast_1d(target_stats[stat + '_sem'][ll]))
                y_hat.append(np.atleast_1d(cmr_stats[stat][ll]))
    
    # Fit PLIs and PLI recency (not separated by list length)
    for stat in ('plis', 'pli_recency'):
        y.append(np.atleast_1d(target_stats[stat]))
        y_sem.append(np.atleast_1d(target_stats[stat + '_sem']))
        y_hat.append(np.atleast_1d(cmr_stats[stat]))
        
    y = np.concatenate(y)
    y_sem = np.concatenate(y_sem)
    y_hat = np.concatenate(y_hat)
    
    chi2_err = np.mean(((y - y_hat) / y_sem) ** 2)
    
    return chi2_err


def sim1c_error(target_stats, cmr_stats):
    """
    Sim 1c fits only the conditional SPCs and PFR, and uses mean squared error 
    instead of chi-squared error because standard errors are not available.
    """
    y = []
    y_hat = []
    
    # Fit SPC and PFR
    for stat in ('spc_fr1', 'spc_frl4', 'pfr'):
        for ll in cmr_stats[stat]:
            # Skip serial position 1 for SPC when initiating recall from position 1 
            if stat == 'spc_fr1':
                y.append(np.atleast_1d(target_stats[stat][ll][1:]))
                y_hat.append(np.atleast_1d(cmr_stats[stat][ll][1:]))
            else:
                y.append(np.atleast_1d(target_stats[stat][ll]))
                y_hat.append(np.atleast_1d(cmr_stats[stat][ll]))
        
    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    
    mse = np.mean((y - y_hat) ** 2)

    return mse