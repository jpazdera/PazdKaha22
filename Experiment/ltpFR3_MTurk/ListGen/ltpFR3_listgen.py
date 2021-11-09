#!/usr/bin/env python2
import random
import itertools
import numpy
import sys
import json
import copy


def make_bins_ltpFR3(semArray):
    """
    Creates four equal-width bins of WAS scores, identical to those used in ltpFR2. Then combine the middle two to give
    three bins: low similarity, medium similarity, and high similarity.

    A coordinate in semRows[i][j] and semCols[i][j] is the index of the jth word pair in semArray that falls in the ith
    similarity bin.
    """
    semArray_nondiag = semArray[numpy.where(semArray != 1)]
    # Find lowest and highest similarity
    min_sim = semArray_nondiag.min()
    max_sim = semArray_nondiag.max()
    # Split up the semantic space into four equal segments
    semBins = list(numpy.linspace(min_sim, max_sim, 4))
    # Combine the two middle bins by removing the bin boundary between them
    # semBins = semBins[:2] + semBins[3:]
    # Create bounds for the bins
    semBins = zip(*[semBins[i:] + semBins[-1:i] for i in range(2)])
    # For word pairs within the bounds of each bin, append the indices to semRows and semCols
    semRows = []
    semCols = []
    for bin in semBins:
        (i, j) = ((semArray > bin[0]) & (semArray < bin[1])).nonzero()
        semRows.append(i)
        semCols.append(j)

    return semRows, semCols

def randomize_conditions_ltpFR3(config):
    """
    Randomize the conditions for all sessions.

    :param config: The imported configuration file, containing all parameters for the experiment
    :return: A list of lists, where sublist n contains the ordering of list conditions for the nth session. cond[x][y][0]
    defines the length of session x, list y; cond[x][y][1] defines the presentation rate of session x, list y;
    cond[x][y][2] defines whether session x, list y uses visual or auditory presentation; cond[x][y][3] defines the
    duration of the pre-list distractor task for session x, list y.
    """
    options = [c for c in itertools.product(config.listLength, config.presRate, config.modality, config.distDur)]
    cond = []

    for i in range(config.nSessions):
        sess = []
        for j in range(config.reps):
            random.shuffle(options)
            sess += options[:]
        cond.append(sess)

    return cond


def choose_pairs_ltpFR3(wp_tot, cond, config, semRows, semCols):
    """
    Selects word pairs to use in each list of each session.

    :param wp_tot: A list containing all the words of the word pool. The order of the words is expected to correspond to
    the indices used by semRows and semCols.
    :param cond: A list of lists, where sublist n contains the ordering of list conditions for the nth session.
    :param config: The imported configuration file, containing all parameters for the experiment.
    :param semRows: See make_bins_ltpFR3()
    :param semCols: See make_bins_ltpFR3()
    :return: pairs - pairs[x][y][z] is the zth word pair in session x, list y
    :return: pair_dicts - a list of dictionaries, where each dictionary contains all word pairs from a given session
    :return: practice_lists - A list containing two practice lists, each with 18 words
    """
    # pairs[x][y][z] will be the zth pair of words in the yth list on session x
    pairs = []
    # points to the other word in the pair for a given session
    pair_dicts = []
    # Deep copy the full word pool into full_wp_allowed, so it can be shuffled for each session without altering wp_tot
    full_wp = wp_tot[:]
    # Make word pairs for each session
    session_num = 0
    while session_num < config.nSessions:
        #print 'Making session', session_num, ':',
        #sys.stdout.flush()
        # Shuffle the order of the word pool; I believe this is technically only necessary for the first session, in
        # order to randomize which words are selected for the practice lists. All other lists have their items randomly
        # chosen anyway
        '''
        IMPORTANT NOTE!!!:
        Lists containing more than 2080 elements should not be randomized with shuffle, as explained here:
        http://stackoverflow.com/questions/3062741/maximal-length-of-list-to-shuffle-with-python-random-shuffle

        The full word pool contains 1638 words, so this is only a concern if the word pool is ever expanded.

        '''
        random.shuffle(full_wp)

        # The first session has two 18-word practice lists
        if session_num == 0:
            practice_lists = [full_wp[:18], full_wp[18:36]]
            sess_wp_allowed = full_wp[36:]
        else:
            sess_wp_allowed = full_wp[:]

        # sess_pairs[x][y] will be the yth pair in the xth list on the current session
        sess_pairs = []

        # Track number of attempts to create the lists for the current session
        sess_tries = 0

        # Track whether the session completed successfully
        goodSess = True

        # Make word pairs for each list in the current session
        list_num = 0
        while list_num < len(cond[session_num]):
            #print list_num,
            #sys.stdout.flush()

            # list_pairs[x] will be the xth pair in the current list on the current session
            list_pairs = []

            # Track number of attempts to create the current list
            list_tries = 0

            # Track whether the list completed successfully
            goodList = True

            # Retrieve the list length condition for the current list by looking in cond
            listLength = cond[session_num][list_num][0]

            # Length 12 lists have 2 pairs per bin, length 24 list have 4 pairs per bin
            pairs_per_bin = 2 if listLength == 12 else 4

            # Select two or four word pairs from each bin (based on list length)
            for sem_i in range(len(semRows)):
                # The pair for each semantic bin gets placed twice
                pair_i = 0

                while pair_i < pairs_per_bin:
                    # Get the indices (within the full word pool) of the words chosen for the current session
                    available_indices = [wp_tot.index(word) for word in sess_wp_allowed]

                    # Randomly choose indices/words from those in the current session until one is found that has one
                    # or more pairs in the current bin
                    index_word1 = random.choice(available_indices)
                    while index_word1 not in semRows[sem_i]:
                        index_word1 = random.choice(available_indices)

                    # Get the indices of all words whose pairing with the chosen word falls into the correct bin
                    good_second_indices = semCols[sem_i][semRows[sem_i] == index_word1]

                    # Eliminate the words that are not available in the session
                    good_second_indices = [i for i in good_second_indices if wp_tot[i] in sess_wp_allowed]

                    # Ensure that a word cannot be accidentally paired with itself
                    if index_word1 in good_second_indices:
                        del good_second_indices[good_second_indices.index(index_word1)]

                    # If there are no good words to choose from, restart
                    if len(good_second_indices) == 0:
                        list_tries += 1
                        if list_tries > 10:
                            goodList = False
                            break
                        else:
                            continue

                    # Choose the second word randomly
                    index_word2 = random.choice(good_second_indices)

                    # Add the pairs to list_pairs, delete them from the pool of allowed words
                    list_pairs.append([wp_tot[index_word1], wp_tot[index_word2]])
                    del sess_wp_allowed[sess_wp_allowed.index(wp_tot[index_word1])]
                    del sess_wp_allowed[sess_wp_allowed.index(wp_tot[index_word2])]

                    pair_i += 1

                # If the list is bad, add the words back to the pool of allowed words
                if not goodList:
                    sess_wp_allowed.extend([x[0] for x in list_pairs] + [x[1] for x in list_pairs])
                    break

            # If the list is good, add the list_pairs to sess_pairs,
            if goodList:
                sess_pairs.append(list_pairs)
                list_num += 1
            else:
                # Otherwise, try the session again (up to 50 times), then restart
                list_pairs = []
                sess_tries += 1
                if sess_tries > 50:
                    goodSess = False
                    break

        # If the whole session went successfully
        if goodSess:
            # Get the pairs from the lists, add them backwards and forwards to sess_pair_dict
            sess_pair_dict = dict(itertools.chain(*sess_pairs))
            sess_pair_dict.update(dict(zip(sess_pair_dict.values(), sess_pair_dict.keys())))
            pair_dicts.append(sess_pair_dict)
            pairs.append(sess_pairs)
            session_num += 1
        else:  # If the session did not go well, try again.
            sess_pairs = []
        print ''
    return pairs, pair_dicts, practice_lists


def place_pairs_ltpFR3(pairs, cond):
    """

    :param pairs:
    :param cond:
    :param config:
    :return:
    """
    # Load all valid list compositions for 12-item lists (small lists are too restrictive to use trial and error)
    with open('valid12.json', 'r') as f:
        valid12 = json.load(f)['3bin-valid12']

    # Loop through sessions
    subj_wo = []
    for (n, sess_pairs) in enumerate(pairs):
        sess_wo = []
        #print '\nPlacing session', n, ':',
        #sys.stdout.flush()
        # Loop through lists within each session
        for (m, list_pairs) in enumerate(sess_pairs):
            #print m,
            #sys.stdout.flush()

            # Create pairs of word pairs from the same bin -- one pair will have adjacent presentation, one distant
            grouped_pairs = [list(group) for group in
                             zip([list_pairs[i] for i in range(len(list_pairs)) if i % 2 == 0],
                                 [list_pairs[i] for i in range(len(list_pairs)) if i % 2 == 1])]

            # Retrieve list length for the current list
            list_length = cond[n][m][0]

            # For 12-item lists, select a random solution template and assign word pairs to the variables in the
            # template, such that one pair from each bin has adjacent presentation and one pair from each bin has
            # distant presentation
            if list_length == 12:

                # Randomize the ordering of the grouped pairs, as well as the orderings within each group and each pair
                adjacents = ['a', 'b', 'c']
                distants = ['d', 'e', 'f']
                random.shuffle(adjacents)
                random.shuffle(distants)
                key = {}
                for group in grouped_pairs:
                    random.shuffle(group)
                    random.shuffle(group[0])
                    random.shuffle(group[1])
                    key[adjacents.pop(0)] = group[0]
                    key[distants.pop(0)] = group[1]

                # Choose a random valid solution
                list_wo = copy.deepcopy(random.choice(valid12))

                # Each entry in the solution list is a string containing a letter followed by 0 or 1
                # The letter corresponds to the word pair and the number corresponds to the item in the pair.
                # Letters a, b, and c are adjacent presentation pairs; d, e, and f are distant presentation pairs.
                for i in range(len(list_wo)):
                    w = list_wo[i]
                    list_wo[i] = key[w[0]][int(w[1])]

            # For 24-item lists, create two 12-item lists based on random solution templates and concatenate them.
            elif list_length == 24:
                # Randomize the ordering of the grouped pairs, as well as the orderings within each group and each pair
                adjacents1 = ['a', 'b', 'c']
                distants1 = ['d', 'e', 'f']
                adjacents2 = ['a', 'b', 'c']
                distants2 = ['d', 'e', 'f']
                random.shuffle(adjacents1)
                random.shuffle(distants1)
                random.shuffle(adjacents2)
                random.shuffle(distants2)
                key1 = {}
                key2 = {}
                for group_num, group in enumerate(grouped_pairs):
                    random.shuffle(group)
                    random.shuffle(group[0])
                    random.shuffle(group[1])
                    if group_num % 2 == 0:
                        key1[adjacents1.pop(0)] = group[0]
                        key1[distants1.pop(0)] = group[1]
                    else:
                        key2[adjacents2.pop(0)] = group[0]
                        key2[distants2.pop(0)] = group[1]

                # Choose a random valid solution
                list_wo1 = copy.deepcopy(random.choice(valid12))
                list_wo2 = copy.deepcopy(random.choice(valid12))

                # Each entry in the solution list is a string containing a letter followed by 0 or 1
                # The letter corresponds to the word pair and the number corresponds to the item in the pair.
                # Letters a, b, and c are adjacent presentation pairs; d, e, and f are distant presentation pairs.
                for i in range(len(list_wo1)):
                    w = list_wo1[i]
                    list_wo1[i] = key1[w[0]][int(w[1])]
                    w = list_wo2[i]
                    list_wo2[i] = key2[w[0]][int(w[1])]

                list_wo = list_wo1 + list_wo2

            else:
                raise ValueError('Function place_pairs_ltpFR3() can only handle word lists of length 12 or 24!')

            # Add finalized list to the session
            sess_wo.append(list_wo)

        subj_wo.append(sess_wo)

    return subj_wo


def listgen_ltpFR3(n):
    """
    Generate all lists for a participant, including the conditions, word pairs
    and word ordering. This function saves the results to a json file labelled
    with the participant's number.

    """
    import config

    # Read in the semantic association matrix
    semMat = []
    with open(config.w2vfile) as w2vfile:
        for word in w2vfile:
            wordVals = []
            wordValsString = word.split()
            for val in wordValsString:
                thisVal = float(val)
                wordVals.append(thisVal)
            semMat.append(wordVals)

    semArray = numpy.array(semMat)
    # Create three semantic similarity bins and sort word pairs by bin
    semRows, semCols = make_bins_ltpFR3(semArray)

    # Read in the word pool
    with open(config.wpfile) as wpfile:
        wp_tot = [x.strip() for x in wpfile.readlines()]

    counts = numpy.zeros(len(wp_tot))
    for i in range(n):
        print '\nSubject ' + str(i) + '\n'
        # Randomize list conditions (list length, presentation rate, modality, distractor duration)
        condi = randomize_conditions_ltpFR3(config)

        # Choose all of the pairs to be used in the experiment
        pairs, pair_dicts, practice_lists = choose_pairs_ltpFR3(wp_tot, condi, config, semRows, semCols)

        # Create all lists by placing the word pairs in appropriate positions
        subj_wo = place_pairs_ltpFR3(pairs, condi)

        # Add practice lists
        subj_wo[0] = practice_lists + subj_wo[0]
        practice_condi = [[18, 1200, 'a', 18000], [18, 1200, 'v', 18000]]
        random.shuffle(practice_condi)
        condi[0] = practice_condi + condi[0]

        d = {'word_order': subj_wo, 'pairs': pair_dicts, 'conditions': condi}

        for sess_dict in pair_dicts:
            counts[numpy.array([wp_tot.index(w) for w in sess_dict])] += 1
            counts[numpy.array([wp_tot.index(w) for w in practice_lists[0]])] += 1
            counts[numpy.array([wp_tot.index(w) for w in practice_lists[1]])] += 1

        with open('/Users/jessepazdera/AtomProjects/ltpFR3_MTurk/static/pools/lists/%d.js' % i, 'w') as f:
            s = 'var sess_info = ' + json.dumps(d) + ';'
            f.write(s)

    with open('/Users/jessepazdera/AtomProjects/ltpFR3_MTurk/static/pools/lists/counts.json', 'w') as f:
        f.write(str([c for c in counts]))

    print max(counts), min(counts), len([wp_tot[i] for i in range(len(counts)) if counts[i] == 0])
    return counts


if __name__ == "__main__":
    nsess = input('How many sessions would you like to generate? ')
    counts = listgen_ltpFR3(nsess)
    print counts.mean()
    print counts.std()
    print counts.max()
    print counts.min()
