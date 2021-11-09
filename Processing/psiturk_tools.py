import os
import json
import numpy as np
import pandas as pd
from glob import glob
from write_to_json import write_data_to_json
from sqlalchemy import create_engine, MetaData, Table
from pyxdameraulevenshtein import damerau_levenshtein_distance_ndarray


def load_psiturk_data(db_url, table_name, event_dir, data_column_name='datastring', force=False):
    """
    Extracts the data from each participant in a psiTurk study, then writes as a JSON file for each participant.

    :param db_url: The location of the database as a string. If using mySQL, must include username and password, e.g.
    'mysql://user:password@127.0.0.1:3306/mturk_db'
    :param table_name: The name of the experiment's table within the database, e.g. 'ltpFR3'
    :param event_dir: The path to the directory where raw event data will be written into JSON files.
    :param data_column_name: The name of the column in which psiTurk has stored the actual experiment event data. By
    default, psiTurk labels this column as 'datastring'
    :param force: If False, only write JSON files for participants that don't already have a JSON file. If True, create
    JSON files for all participants. (Default == False)
    """

    """
    Status codes are as follows:
    NOT_ACCEPTED = 0
    ALLOCATED = 1
    STARTED = 2
    COMPLETED = 3
    SUBMITTED = 4
    CREDITED = 5
    QUITEARLY = 6
    BONUSED = 7
    """
    complete_statuses = [3, 4, 5, 7]  # Status codes of subjects who have completed the study

    # Use sqlalchemy to load rows from specified table in the specified database
    engine = create_engine(db_url)
    metadata = MetaData()
    metadata.bind = engine
    table = Table(table_name, metadata, autoload=True)
    s = table.select()
    rows = s.execute()

    for row in rows:
        # Get subject ID
        subj_id = row['workerid']
        datafile_path = os.path.join(event_dir, '%s.json' % subj_id)
        inc_datafile_path = os.path.join(event_dir, 'incomplete', '%s.json' % subj_id)

        # Extract participant's data string
        data = row[data_column_name]

        # Only attempt to write a JSON file if the participant has data and does not already have a JSON file
        if data != '' and not os.path.exists(datafile_path):
            # Parse data string as a JSON object
            try:
                data = json.loads(data)
            except:
                print('Failed to parse session log for %s as a JSON object! Skipping...' % subj_id)
                continue

            # Write JSON data to a file if the file does not already exist
            if force or not os.path.exists(datafile_path):
                with open(datafile_path, 'w') as f:
                    json.dump(data, f)

        # Move logs from incomplete sessions to their own folder
        status = row['status']
        if status not in complete_statuses and os.path.exists(datafile_path):
            os.rename(datafile_path, inc_datafile_path)
        # Remove logs previously marked as incomplete if they have now become complete
        elif status in complete_statuses and os.path.exists(inc_datafile_path):
            os.remove(inc_datafile_path)


def process_psiturk_data(event_dir, behmat_dir, dict_path, force=False):
    """
    Post-process the raw psiTurk data extracted by load_psiturk_data. This involves creating recalls and presentation
    matrices, extracting list conditions, etc. The matrices created are saved to a JSON file for each participant.

    :param event_dir: The path to the directory where raw event data JSON files are stored.
    :param behmat_dir: The path to the directory where new behavioral matrix data files will be stored.
    :param dict_path: The path to the text file containing the English dictionary to use for spell-checking.
    :param force: If False, only process data from participants who do not already have a behavioral matrix file. If
    True, process data from all participants. (Default == False)
    """
    # Load the English dictionary file, remove spaces, and make all words lowercase
    with open(dict_path, 'r') as df:
        dictionary = df.readlines()
    dictionary = [word.lower().strip() for word in dictionary if ' ' not in word]

    # Load list of excluded participants
    exclude = [s.decode('UTF-8') for s in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/EXCLUDED.txt', dtype='S8')]
    bad_sess = [s.decode('UTF-8') for s in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/BAD_SESS.txt', dtype='S8')]
    rejected = [s.decode('UTF-8') for s in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/REJECTED.txt', dtype='S8')]

    # Process each participant's raw data into a JSON file of behavioral matrices
    for json_file in glob(os.path.join(event_dir, '*.json')):

        s = os.path.splitext(os.path.basename(json_file))[0]  # Get subject ID from file name
        outfile = os.path.join(behmat_dir, '%s.json' % s)  # Define file path for behavioral matrix file
        # Skip bad sessions and participants who have already been processed, excluded, or rejected
        if s in bad_sess or ((os.path.exists(outfile) or s in exclude or s in rejected) and not force):
            continue

        # Get participant's data as data frame
        with open(json_file, 'r') as f:
            data = json.load(f)
        s = data['workerId']
        data = [record['trialdata'] for record in data['data']]
        data = pd.DataFrame(data)
        print(s)

        # Initialize data entries for subject
        d = {}
        d['serialpos'] = []
        d['ffr_serialpos'] = []
        d['ffr_pres_trial'] = []
        d['rec_words'] = []
        d['ffr_rec_words'] = []
        d['pres_words'] = []
        d['list_len'] = []
        d['pres_rate'] = []
        d['pres_mod'] = []
        d['dist_dur'] = []
        d['math_correct'] = []

        # Set filters for recall and presentation events
        recalls_filter = data.type == 'FREE_RECALL'
        study_filter_aud = data.type == 'PRES_AUD'
        study_filter_vis = data.type == 'PRES_VIS'
        distractor_filter = data.type == 'DISTRACTOR'
        ffr_filter = data.type == 'FFR'

        # Get all presentation and recall events from the current subject
        s_pres = data.loc[study_filter_aud | study_filter_vis, ['trial', 'word', 'conditions']].as_matrix()
        s_recalls = data.loc[recalls_filter, ['trial', 'recwords', 'conditions', 'rt']].as_matrix()
        s_ffr = data.loc[ffr_filter, ['recwords', 'rt']].as_matrix()
        pres_trials = np.array([x[0] for x in s_pres])
        pres_words = np.array([str(x[1]).strip() for x in s_pres])
        rec_trials = np.array([x[0] for x in s_recalls])
        rec_words = np.array([[str(y).strip() for y in x[1] if str(y).strip() != ''] for x in s_recalls])
        d['rt'] = pad_into_array([[t for i, t in enumerate(x[3]) if str(x[1][i]).strip() != ''] for x in s_recalls])

        # Get distractor problems and responses, then total the number of correct answers on each trial
        s_dist = data.loc[distractor_filter, ['num1', 'num2', 'num3', 'responses']].as_matrix()
        for trial_data in s_dist:
            valid = np.where([ans.strip().isnumeric() for ans in trial_data[3]])
            n1 = np.array(trial_data[0])[valid].astype(int)
            n2 = np.array(trial_data[1])[valid].astype(int)
            n3 = np.array(trial_data[2])[valid].astype(int)
            resp = np.array(trial_data[3])[valid].astype(int)
            correct = n1 + n2 + n3 == resp
            d['math_correct'].append(correct.tolist())

        # Add presented words to the data structure
        d['pres_words'] = [[word for i, word in enumerate(pres_words) if pres_trials[i] == trial] for trial in np.unique(pres_trials)]

        # Get conditions for each trial and add them to the data structure
        conditions = [x[2] for x in s_recalls]
        d['list_len'] = [x[0] for x in conditions]
        d['pres_rate'] = [x[1] for x in conditions]
        d['pres_mod'] = [x[2] for x in conditions]
        d['dist_dur'] = [x[3] for x in conditions]

        # Create empty recalled and FFR_recalled matrix to mark whether each word was subsequently recalled
        d['recalled'] = np.zeros((len(d['pres_words']), np.max(d['list_len'])))
        d['ffr_recalled'] = np.zeros((len(d['pres_words']), np.max(d['list_len'])))

        # For each trial in a subject's session
        for i, t in enumerate(np.unique(pres_trials)):
            # Get a list of all words presented so far this session (to be able to search for PLIs)
            presented_so_far = pres_words[np.where(pres_trials <= t)]
            # Get a list of the trials each word was presented on
            when_presented = pres_trials[np.where(pres_trials <= t)]
            # Get an array of the words recalled this trial
            recalled_this_list = rec_words[np.where(rec_trials == t)][0] if len(rec_words[np.where(rec_trials == t)]) > 0 else []

            # Mark each recall as a correct recall, ELI, PLI, or other and make a recall list for the current trial
            sp = []
            for j, recall in enumerate(recalled_this_list):
                list_num, position, recall = which_item(recall, presented_so_far, when_presented, dictionary)
                recalled_this_list[j] = recall  # Replace recalled words with their spell-checked versions
                if list_num is None:
                    # ELIs and invalid strings get error code of -999 listed in their recalls matrix
                    sp.append(position)
                else:
                    # Mark word as recalled
                    d['recalled'][list_num, position-1] = 1
                    # PLIs get serial position of -n, where n is the number of lists back the word was presented
                    if list_num != t:
                        sp.append(list_num - t)
                    # Correct recalls get their serial position listed as is
                    else:
                        sp.append(position)

            # Add the current trial's recalls as a row in the participant's recalls matrix
            d['serialpos'].append(sp)
            d['rec_words'].append(recalled_this_list)
            d['recalled'][i, d['list_len'][i]:] = np.nan
            d['ffr_recalled'][i, d['list_len'][i]:] = np.nan

        # Process FFR data
        d['ffr_rt'] = []
        if len(s_ffr) == 1 and len(s_ffr[0]) == 2:
            for i, recall in enumerate(s_ffr[0][0]):
                if str(recall).strip() == '':
                    continue
                list_num, position, recall = which_item(recall, pres_words, pres_trials, dictionary)
                d['ffr_rec_words'].append(recall)
                d['ffr_serialpos'].append(position)
                d['ffr_rt'].append(s_ffr[0][1][i])
                if list_num is None:
                    d['ffr_pres_trial'].append(-999)
                else:
                    d['ffr_pres_trial'].append(list_num)
                    d['ffr_recalled'][list_num, position-1] = 1

        if max([len(x) for x in d['pres_words']]) > 24:
            print('%s SUBJECT RESTART DETECTED!! EXLCUDING!' % s)
        else:
            # Zero-pad recall-related arrays, create intrusions matrix, and create subject array, then save to JSON
            d['subject'] = [s for row in d['serialpos']]
            d['serialpos'] = pad_into_array(d['serialpos']).astype(int)
            d['rec_words'] = pad_into_array(d['rec_words'])
            d['pres_words'] = pad_into_array(d['pres_words'])
            d['intrusions'] = recalls_to_intrusions(d['serialpos'])
            d['math_correct'] = pad_into_array(d['math_correct']).astype(bool)
            write_data_to_json(d, outfile)


def which_item(recall, presented, when_presented, dictionary):
    """
    Determine the serial position of a recalled word. Extra-list intrusions are identified by looking them up in a word
    list. Unrecognized words are spell-checked.

    :param recall: A string typed by the subject into the recall entry box
    :param presented: The list of words seen by this subject so far, across all trials <= the current trial number
    :param when_presented: A listing of which trial each word was presented in
    :param dictionary: A list of strings that should be considered as possible extra-list intrusions
    :return: If a correct recall or PLI, returns the trial number and serial position of the word's presentation, plus
    the spelling-corrected version of the recall. If an ELI or invalid entry, returns a trial number of None, a serial
    position of -999, and the spelling-corrected version of the recall.
    """

    # Check whether the recall exactly matches a previously presented word
    seen, seen_where = self_term_search(recall, presented)

    # If word has been presented
    if seen:
        # Determine the list number and serial position of the word
        list_num = when_presented[seen_where]
        first_item = np.min(np.where(when_presented == list_num))
        serial_pos = seen_where - first_item + 1
        return int(list_num), int(serial_pos), recall

    # If the recalled word was not presented, but exactly matches any word in the dictionary, mark as an ELI
    in_dict, where_in_dict = self_term_search(recall, dictionary)
    if in_dict:
        return None, -999, recall

    # If the recall contains non-letter characters
    if not recall.isalpha():
        return None, -999, recall

    # If word is not in the dictionary, find the closest match based on edit distance
    recall = correct_spelling(recall, presented, dictionary)
    return which_item(recall, presented, when_presented, dictionary)


def self_term_search(find_this, in_this):
    for index, word in enumerate(in_this):
        if word == find_this:
            return True, index
    return False, None


def correct_spelling(recall, presented, dictionary):

    # edit distance to each item in the pool and dictionary
    dist_to_pool = damerau_levenshtein_distance_ndarray(recall, np.array(presented))
    dist_to_dict = damerau_levenshtein_distance_ndarray(recall, np.array(dictionary))

    # position in distribution of dist_to_dict
    ptile = np.true_divide(sum(dist_to_dict <= np.amin(dist_to_pool)), dist_to_dict.size)

    # decide if it is a word in the pool or an ELI
    if ptile <= .1:
        corrected_recall = presented[np.argmin(dist_to_pool)]
    else:
        corrected_recall = dictionary[np.argmin(dist_to_dict)]
    recall = corrected_recall
    return recall


def pad_into_array(l):
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

    # If all rows were the same length, just return the original input as an array
    if lens.max() == lens.min():
        return l

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=l.dtype)
    out[mask] = np.concatenate(l)

    return out


def recalls_to_intrusions(rec):
    """
    Convert a recalls matrix to an intrusions matrix. In the recalls matrix, ELIs should be denoted by -999 and PLIs
    should be denoted by -n, where n is the number of lists back the word was originally presented. All positive numbers
    are assumed to be correct recalls. The resulting intrusions matrix denotes correct recalls by 0, ELIs by -1, and
    PLIs by n, where n is the number of lists back the word was originally presented.

    :param rec: A lists x items recalls matrix, which is assumed to be a numpy array
    :return: A lists x items intrusions matrix
    """
    intru = rec.copy()
    # Set correct recalls to 0
    intru[np.where(intru > 0)] = 0
    # Convert negative numbers for PLIs to positive numbers
    intru *= -1
    # Convert ELIs to -1
    intru[np.where(intru == 999)] = -1
    return intru
