import os
import json
import numpy as np
from glob import glob
from pybeh.spc import spc
from pybeh.pnr import pnr
from pybeh.crp import crp
from pybeh.temp_fact import temp_fact
from scipy.stats import sem
from write_to_json import write_stats_to_json


def run_stats(data_dir, stat_dir, force=False):
    """
    Use the data extracted from a psiTurk database to generate behavioral stats for each participant. Creates a stat
    data structure with the following format for each participant, then saves it to a JSON file:

    {
    stat1: {condition1: value, condition2: value, ...},
    stat2: {condition1: value, condition2: value, ...},
    stat3: {condition1: value, condition2: value, ...},
    ...
    }
    
    :param data_dir: The path to the directory where behavioral matrix data files are stored.
    :param report_dir: The path to the directory where new stats files will be saved.
    :param force: If False, only calculate stats for participants who do not already have a stats file (plus the average
     stat file). If True, calculate stats for all participants. (Default == False)
    """
    exclude = [s.decode('UTF-8') for s in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/EXCLUDED.txt', dtype='S8')]
    bad_sess = [s.decode('UTF-8') for s in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/BAD_SESS.txt', dtype='S8')]
    rejected = [s.decode('UTF-8') for s in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/REJECTED.txt', dtype='S8')]
    skip = np.union1d(np.union1d(exclude, bad_sess), rejected)
    with open('/data/eeg/scalp/ltp/ltpFR3_MTurk/VERSION_STARTS.json') as f:
        version_starts = json.load(f)
        version_starts = {int(v): version_starts[v] for v in version_starts}

    stats_to_run = ['prec', 'spc', 'pfr', 'psr', 'ptr', 'crp', 'crp_early', 'crp_late', 'plis', 'elis', 'reps', 'pli_recency', 'ffr_spc', 'temp_fact', 'irt', 'irt_sp_excl']

    filters = {'all': {'ll': None, 'pr': None, 'mod': None, 'dd': None},
               'a12': {'ll': 12, 'mod': 'a'}, 'a24': {'ll': 24, 'mod': 'a'},
               'v12': {'ll': 12, 'mod': 'v'}, 'v24': {'ll': 24, 'mod': 'v'},
               'f12': {'ll': 12, 'pr': 800}, 'f24': {'ll': 24, 'pr': 800},
               's12': {'ll': 12, 'pr': 1600}, 's24': {'ll': 24, 'pr': 1600},
               'sa12': {'ll': 12, 'pr': 1600, 'mod': 'a'}, 'sa24': {'ll': 24, 'pr': 1600, 'mod': 'a'},
               'sv12': {'ll': 12, 'pr': 1600, 'mod': 'v'}, 'sv24': {'ll': 24, 'pr': 1600, 'mod': 'v'},
               'fa12': {'ll': 12, 'pr': 800, 'mod': 'a'}, 'fa24': {'ll': 24, 'pr': 800, 'mod': 'a'},
               'fv12': {'ll': 12, 'pr': 800, 'mod': 'v'}, 'fv24': {'ll': 24, 'pr': 800, 'mod': 'v'}}

    # Calculate stats for new participants
    for data_file in glob(os.path.join(data_dir, '*.json')):
        subj = os.path.splitext(os.path.basename(data_file))[0]  # Get subject ID from file name
        outfile = os.path.join(stat_dir, '%s.json' % subj)  # Define file path for stat file
        if subj in bad_sess or ((os.path.exists(outfile) or subj in exclude or subj in rejected) and not force):  # Skip participants who already had stats calculated
            continue

        with open(data_file, 'r') as f:
            d = json.load(f)

        # Extract behavioral matrices from JSON object and convert to numpy arrays
        sub = np.array(d['subject'])
        condi = [(d['list_len'][i], d['pres_rate'][i], str(d['pres_mod'][i]), d['dist_dur'][i]) for i in range(len(d['serialpos']))]
        recalls = np.array(d['serialpos'])
        wasrec = np.array(d['recalled'])
        ffr_wasrec = np.array(d['ffr_recalled'])
        rt = np.array(d['rt'])
        recw = np.array(d['rec_words'])
        presw = np.array(d['pres_words'])
        intru = np.array(d['intrusions'])
        math = np.array(d['math_correct'])

        # Run all stats for a single participant and add the resulting stats object to the stats dictionary
        stats = stats_for_subj(sub, condi, recalls, wasrec, ffr_wasrec, rt, recw, presw, intru, math, stats_to_run, filters)
        write_stats_to_json(stats, outfile)

    # Calculate average stats. First we need to load the stats from all participants.
    stats = {}
    for stat_file in glob(os.path.join(stat_dir, '*.json')):
        subj = os.path.splitext(os.path.basename(stat_file))[0]
        if not subj.startswith('all'):
            with open(stat_file, 'r') as f:
                stats[subj] = json.load(f)

    # Now we calculate the average stats and save to a JSON file
    for version in version_starts:
        avg_stats = {}
        v_start = version_starts[version]
        v_end = None if version + 1 not in version_starts else version_starts[version + 1]

        outfile = os.path.join(stat_dir, 'all_v%d.json' % version)
        avg_stats['mean'], avg_stats['sem'], avg_stats['N'] = calculate_avg_stats(stats, stats_to_run, filters.keys(),
                                                                version_start=v_start, version_end=v_end)
        write_stats_to_json(avg_stats, outfile, average_stats=True)

        outfile = os.path.join(stat_dir, 'all_v%d_excl_wn.json' % version)
        avg_stats['mean'], avg_stats['sem'], avg_stats['N'] = calculate_avg_stats(stats, stats_to_run, filters.keys(),
                                                                                  version_start=v_start,
                                                                                  version_end=v_end,
                                                                                  exclude_wrote_notes=True)
        write_stats_to_json(avg_stats, outfile, average_stats=True)


def stats_for_subj(sub, condi, recalls, wasrec, ffr_wasrec, rt, recw, presw, intru, math, stats_to_run, filters):
    """
    Create a stats dictionary for a single participant.
    
    :param sub: A subject array for the stats calculations. As we are only running stats on one participant at a time, 
    this array will simply be [subj_name] * number_of_trials.
    :param condi: An array of tuples indicating which conditions were used on each trial (ll, pr, mod, dd)
    :param recalls: A matrix where item (i, j) is the serial position of the ith recall in trial j
    :param wasrec: A matrix where item (i, j) is 0 if the ith presented word in trial j was not recalled, 1 if it was
    :param rt: A matrix where item (i, j) is the response time (in ms) of the ith recall in trial j
    :param recw: A matrix where item (i, j) is the ith word recalled on trial j
    :param presw: A matrix where item (i, j) is the ith word presented on trial j
    :param intru: A list x items intrusions matrix (see recalls_to_intrusions)
    :return: 
    """
    stats = {stat: {} for stat in stats_to_run}
    for f in filters:
        # Get presentation and recall info just from trials that match the filter's set of conditions
        fsub, frecalls, fwasrec, fffr_wasrec, frt, frecw, fpresw, fintru = [filter_by_condi(a, condi, **filters[f]) for a in [sub, recalls, wasrec, ffr_wasrec, rt, recw, presw, intru]]

        # If no trials match the current filter, skip to next filter
        if fsub is None:
            continue

        # Calculate stats on all lists within the current condition. Note that some stats require filtering by list len
        ll = filters[f]['ll']
        if ll is not None:
            stats['prec'][f] = prec(fwasrec[:, :ll], fsub)[0]
            stats['spc'][f] = spc(frecalls, fsub, ll)[0]
            stats['ffr_spc'][f] = ffr_spc(fffr_wasrec, fsub, ll)[0]
            stats['pfr'][f] = pnr(frecalls, fsub, ll, n=0)[0]
            stats['psr'][f] = pnr(frecalls, fsub, ll, n=1)[0]
            stats['ptr'][f] = pnr(frecalls, fsub, ll, n=2)[0]
            stats['crp'][f] = crp(frecalls, fsub, ll, lag_num=4)[0]
            stats['crp_early'][f] = crp(frecalls[:, :3], fsub, ll, lag_num=5)[0]
            stats['crp_late'][f] = crp(frecalls, fsub, ll, lag_num=5, skip_first_n=2)[0]
            stats['temp_fact'][f] = temp_fact(frecalls, fsub, ll, skip_first_n=2)[0]
            stats['irt'][f] = irt_subj(frt, frecalls, ll)
            # Special version of the IRT which excludes trials that had a recall within the last 10 seconds
            rt_exclusion_mask = np.max(frt, axis=1) <= 50000
            if rt_exclusion_mask.sum() > 0:
                stats['irt_sp_excl'][f] = irt_subj(frt[rt_exclusion_mask], frecalls[rt_exclusion_mask], ll)
            else:
                stats['irt_sp_excl'][f] = np.empty((ll+1, ll+1))
                stats['irt_sp_excl'][f].fill(np.nan)
        stats['plis'][f] = avg_pli(fintru, fsub, frecw)[0]
        stats['elis'][f] = avg_eli(fintru, fsub)[0]
        stats['reps'][f] = avg_reps(frecalls, fsub)[0]
        stats['pli_recency'][f] = pli_recency(fintru, fsub, 5)[0]

    stats['rec_per_trial'] = np.nanmean(wasrec, axis=1)
    stats['math_per_trial'] = np.sum(math, axis=1)

    return stats


def calculate_avg_stats(s, stats_to_run, filters, version_start=1, version_end=None, exclude_wrote_notes=False):
    # Exclusion notes can be found at: https://app.asana.com/0/291595828487527/468440625589939/f
    exclude = [b.decode('UTF-8') for b in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/EXCLUDED.txt', dtype='S8')]
    bad_sess = [b.decode('UTF-8') for b in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/BAD_SESS.txt', dtype='S8')]
    rejected = [b.decode('UTF-8') for b in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/REJECTED.txt', dtype='S8')]
    skip = np.union1d(np.union1d(exclude, bad_sess), rejected)

    if exclude_wrote_notes:
        wrote_notes = [b.decode('UTF-8') for b in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/WROTE_NOTES.txt', dtype='S8')]
        skip = np.union1d(skip, wrote_notes)

    avs = {}
    stderr = {}
    Ns = {}
    for stat in stats_to_run:
        avs[stat] = {}
        stderr[stat] = {}
        Ns[stat] = {}
        for f in filters:
            # For the "all" filter, only do intrusion stats
            if f == 'all' and stat not in ('plis', 'elis', 'reps', 'pli_recency'):
                continue
            scores = []
            # Only use a subject's scores if they are from the correct version of the experiment and are not excluded
            for subj in s:
                snum = int(subj[-4:])
                if (subj not in skip) and (version_start is None or snum >= version_start) and \
                        (version_end is None or snum < version_end) and f in s[subj][stat]:
                    scores.append(s[subj][stat][f])
            scores = np.array(scores)
            avs[stat][f] = np.nanmean(scores, axis=0)
            stderr[stat][f] = sem(scores, axis=0, nan_policy='omit')
            Ns[stat][f] = np.sum(np.logical_not(np.isnan(scores)), axis=0, dtype=np.float64)

    return avs, stderr, Ns


def filter_by_condi(a, condi, ll=None, pr=None, mod=None, dd=None):
    """
    Filter a matrix of trial data to only get the data from trials that match the specified condition(s)
    :param a: A numpy array with the data from one trial on each row
    :param condi: A list of tuples indicating the list length, presentation rate, modality, distractor duration of each trial
    :param ll: Return only trials with this list length condition (ignore if None)
    :param pr: Return only trials with this presentation rate condition (ignore if None)
    :param mod: Return only trials with this presentation modality condition (ignore if None)
    :param dd: Return only trials with this distractor duration condition (ignore if None
    :return: A numpy array containing only the data from trials that match the specified condition(s)
    """
    ind = [i for i in range(len(condi)) if ((ll is None or condi[i][0] == ll) and (pr is None or condi[i][1] == pr) and (mod is None or condi[i][2] == mod) and (dd is None or condi[i][3] == dd))]
    if len(ind) == 0:
        return None
    return a[ind]


def prec(recalled, subjects):
    """
    Calculate the overall probability of recall for each subject, given a lists x items matrix where 0s indicate
    words that were not subsequently recalled and 1s indicate words that were subsequently recalled. Item (i, j)
    should indicate whether the jth word presented in list i was recalled.
    
    :param recalled: A lists x items matrix, indicating whether each presented word was subsequently recalled
    :param subjects: A list of subject codes, indicating which subject produced each row of was_recalled
    :return: An array containing the overall probability of recall for each unique participant
    """
    if len(recalled) == 0:
        return np.array([]), np.array([])
    usub = np.unique(subjects)
    result = np.array([np.nanmean(recalled[subjects == subj]) for subj in usub])
    stderr = np.array([sem(recalled[subjects == subj], nan_policy='omit') for subj in usub])

    return result, stderr


def ffr_spc(ffr_recalled, subjects, ll):
    if len(ffr_recalled) == 0:
        return np.array([])
    usub = np.unique(subjects)
    result = np.array([np.nanmean(ffr_recalled[subjects == subj], axis=0) for subj in usub])
    return result[:, :ll]


def pli_recency(intrusions, subjects, nmax):
    """
    Calculate the ratio of PLIs that originated from 1 list back, 2 lists back, etc. up until nmax lists back. Note that
    this requires the function to skip the first nmax lists of each subject's data.
    
    :param intrusions: An intrusions matrix in the format generated by recalls_to_intrusions
    :param subjects: A list of subject codes, indicating which subject produced each row of the intrusions matrix
    :param nmax: The maximum number of lists back to consider
    :return: An array of length nmax, where item i is the ratio of PLIs that originated from i+1 lists back
    """
    if len(intrusions) == 0 or nmax < 1:
        return np.array([])

    usub = np.unique(subjects)
    result = np.zeros((len(usub), nmax+1), dtype=float)
    ll = len(intrusions[0])
    for i, s in enumerate(usub):
        subj_intru = intrusions[subjects == s, :]
        for j in range(len(subj_intru)):
            # Skip the first nmax lists from each participant
            if j < nmax:
                continue
            for k in range(ll):
                if 1 <= subj_intru[j][k] <= nmax:
                    result[i, subj_intru[j][k] - 1] += 1

    result /= result.sum(axis=1)
    return result


def avg_pli(intrusions, subjects, rec_itemnos):
    """
    A modification of the behavioral toolbox's pli function. Calculate's each partcipant's average number of PLIs per 
    list instead of their total number of PLIs.
    
    :param intrusions: An intrusions matrix in the format generated by recalls_to_intrusions
    :param subjects: A list of subject codes, indicating which subject produced each row of the intrusions matrix
    :param rec_itemnos: A matrix in which each row is the list of IDs for all words recalled by a single subject on a
                        single trial. Rows are expected to be padded with 0s to all be the same length.
    :return: An array where each entry is the average number of PLIs per list for a single participant.
    """
    usub = np.unique(subjects)
    result = np.zeros(len(usub))
    for subject_index in range(len(usub)):
        count = 0.
        lists = 0.
        for subj in range(len(subjects)):
            if subjects[subj] == usub[subject_index]:
                lists += 1
                encountered = []
                for serial_pos in range(len(intrusions[0])):
                    if intrusions[subj][serial_pos] > 0 and rec_itemnos[subj][serial_pos] not in encountered:
                        count += 1
                        encountered.append(rec_itemnos[subj][serial_pos])
        result[subject_index] = count / lists if lists > 0 else np.nan

    return result


def avg_eli(intrusions=None, subjects=None):
    """
    A modification of the behavioral toolbox's xli function. Calculate's each partcipant's average number of ELIs per 
    list instead of their total number of ELIs.

    :param intrusions: An intrusions matrix in the format generated by recalls_to_intrusions
    :param subjects: A list of subject codes, indicating which subject produced each row of the intrusions matrix
    :return: An array where each entry is the average number of PLIs per list for a single participant.
    """
    usub = np.unique(subjects)
    result = np.zeros(len(usub))
    for subject_index in range(len(usub)):
        count = 0.
        lists = 0.
        for subj in range(len(subjects)):
            if subjects[subj] == usub[subject_index]:
                lists += 1
                for serial_pos in range(len(intrusions[0])):
                    if intrusions[subj][serial_pos] < 0:
                        count += 1
        result[subject_index] = count / lists if lists > 0 else np.nan
    return result


def avg_reps(rec_itemnos, subjects):
    """
    Calculate's each partcipant's average number of repetitions per list.
    
    :param rec_itemnos: A matrix in which each row is the list of IDs for all words recalled by a single subject on a
                        single trial. Rows are expected to be padded with 0s to all be the same length.
    :param subjects: A list of subject codes, indicating which subject produced each row of the intrusions matrix
    :return: An array where each entry is the average number of repetitions per list for a single participant.
    """
    usub = np.unique(subjects)
    result = np.zeros(len(usub))
    for subject_index in range(len(usub)):
        count = 0.
        lists = 0.
        for subj in range(len(subjects)):
            if subjects[subj] == usub[subject_index]:
                lists += 1
                # times_recalled is an array with one entry for each unique correctly recalled word, indicating the
                # number of times that word was recalled during the current list
                times_recalled = np.array([len(np.where(rec_itemnos[subj, :] == rec)[0]) for rec in np.unique(rec_itemnos[subj, :]) if rec > 0])
                # Subtract 1 from the number of times each correct word was recalled in the list to give the number of
                # repetitions
                repetitions = times_recalled - 1
                # Sum the number of repetitions made in the current list
                count += repetitions.sum()
        result[subject_index] = count / lists if lists > 0 else np.nan
    return result


def irt_subj(rectimes, recalls, ll):
    irt = np.zeros((ll+1, ll+1))
    trial_count = np.zeros_like(irt)

    for trial, row in enumerate(rectimes):
        # Get the first output position at which each word was recalled, filtering out intrusions and repetitions
        spos, idx = np.unique(recalls[trial], return_index=True)
        idx = idx[spos > 0]

        # Get only the RTs for correct recalls
        masked_rt = row[np.sort(idx)]

        # Count the number of correct recalls
        num_recs = np.sum(masked_rt != 0)

        # Group trials' RTs based on the number of correct recalls made
        if num_recs > 1:
            irt[num_recs, :num_recs - 1] += np.diff(masked_rt)
            trial_count[num_recs, :] += 1

    # Set NaNs where no data is available
    irt[irt == 0] = np.nan
    irt = irt / trial_count

    return irt
