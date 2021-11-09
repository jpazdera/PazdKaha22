#!/usr/bin/env python2
import random
import itertools
import numpy
import sys
import os
import json
import copy

def makeSemBins(semArray, nBins):
    """
    Makes the semantic bins.
    A spot in semRows[i] and semCols[i] are the indices of words that
    fall in the ith semantic bin
    """

    # Split up the semantic space into equal segments
    semBins = list(numpy.linspace(semArray.min(),semArray.max(),nBins+1))
    # Creates boundaries for the segments
    semBins = zip(*[semBins[i:]+semBins[-1:i] for i in range(2)])
    semRows = []
    semCols = []
    for bin in semBins:
        # For words within those boundaries, append the indices to
        # semRows and semCols
        (i,j) = ((semArray > bin[0]) & (semArray < bin[1])).nonzero()
        semRows.append(i)
        semCols.append(j)

    return semRows, semCols

def choosePairs(semArray, wp_tot, wp_allowed_main, wp_allowed_last, config, semRows, semCols):
    """
    Chooses all of the pairs to be presented in all sessions
    """
    # Will hold all of the sessions' pairs.
    # pairs[x][y][z] gives the zth pair of words in the yth list on session x
    # the last item in pairs[x][y] will always contain 8 additional words 
    # that are not a part of any pair
    pairs = []
    # points to the other word in the pair for a given session
    pair_dicts = []
    # Take from all of the words in wp_allowed_main (except for the last session)
    full_wp_allowed = wp_allowed_main[:]
    session_num = 0
    # Go through each session
    while session_num <config.nSessions:
        print 'Making session', session_num,':',
        sys.stdout.flush()

        # words allowed for that session are taken from the full wordpool
        sess_wp_allowed = full_wp_allowed[:]
        # Pairs for a given session
        sess_pairs = []

        #number of times it's attempted to make the list
        sess_tries = 0

        list_num = 0
        # keeps track of whether the session completed successfully
        goodSess = True
        while list_num< config.nLists:
            print list_num,
            sys.stdout.flush()

            # If it's the last session, the second half of the lists
            # should contain the words for the last session
            if session_num==config.nSessions-1 and list_num==config.nLists/2:
                unused_sess_wp = sess_wp_allowed[:]
                sess_wp_allowed = wp_allowed_last[:]

            # Pairs within a given list
            list_pairs = [] 
            list_tries = 0
            goodList = True
            for sem_i in range(len(semRows)):
                # The pair for each semantic bin gets placed twice
                pair_i = 0
                while pair_i<2:
                    
                    # Get the indices of the words in sess_wp_allowed
                    available_indices = [wp_tot.index(word) for word in sess_wp_allowed]
                    #available_indices = [i for i in range(len(wp_tot)) if wp_tot[i] in sess_wp_allowed]
                    
                    # Randomly choose indices from available_indices until it falls in semRows[sem_i]
                    index_word1 = random.choice(available_indices)
                    while index_word1 not in semRows[sem_i]:
                        index_word1 = random.choice(available_indices)
                    
                    # Get all of the indices of words that correspond to the word chosen for word1
                    # that also fall in the correct sem_bin
                    good_second_indices = semCols[sem_i][semRows[sem_i]==index_word1]
                    
                    # Eliminate the words that are not available in the session
                    good_second_indices = [i for i in good_second_indices if wp_tot[i] in sess_wp_allowed]

                    # Get rid of the first word, if it does fall in the correct bin
                    if index_word1 in good_second_indices:
                        del good_second_indices[good_second_indices.index(index_word1)]

                    # if there are no good words to choose from, restart
                    if len(good_second_indices)==0:
                        list_tries+=1
                        if list_tries>10:
                            goodList = False
                            break
                        else:
                            continue
                    
                    # Choose the second word randomly
                    index_word2 = random.choice(good_second_indices)

                    # Not sure why this is here. Probably doesn't need to be.
                    while index_word2==index_word1:
                        index_word2 = random.choice(good_second_indices)
                    
                    # Add the pairs to list_pairs, delete them from the pool of allowed words
                    list_pairs.append((wp_tot[index_word1], wp_tot[index_word2]))
                    del sess_wp_allowed[sess_wp_allowed.index(wp_tot[index_word1])]
                    del sess_wp_allowed[sess_wp_allowed.index(wp_tot[index_word2])]
                    
                    pair_i +=1
                
                # If the list is bad, add the words back to the pool of allowed words
                if not goodList:
                    sess_wp_allowed.extend([x[0] for x in list_pairs]+[x[1] for x in list_pairs])
                    break
            
            # If the list is good, add the list_pairs to sess_pairs,
            if goodList:
                sess_pairs.append(list_pairs)
                list_num+=1
            else:
                # Otherwise, try the session again (up to 50 times), then restart
                list_pairs = []
                sess_tries += 1
                if sess_tries>50:
                    goodSess = False
                    break

        # If the whole session went sucessfully
        if goodSess:
            # Get the pairs from the lists, add them backwards and forwards to sess_pair_dict
            sess_pair_dict = dict(itertools.chain(*sess_pairs))
            sess_pair_dict.update(dict(zip(sess_pair_dict.values(), sess_pair_dict.keys())))
            pair_dicts.append(sess_pair_dict)

            # Add 8 extra words to the end of every list.
            for list in range(config.nLists):
                list_extra_words = []
                for i in range(config.listLength-8*2):
                    if session_num!=config.nSessions-1 or list<config.nLists/2:
                        list_extra_words.append(sess_wp_allowed.pop(random.randint(0,len(sess_wp_allowed)-1)))
                    else:
                        list_extra_words.append(unused_sess_wp.pop(random.randint(0,len(unused_sess_wp)-1)))
                sess_pairs[list].append(list_extra_words)
            
            # If it's the last session, sess_pairs contains old words for half, then new words for half.
            # This mixes up the lists
            if session_num==config.nSessions-1:
                print '\n Shuffling last session...'
                sess_pairs_mixed = []
                
                # If it's the last session, sess_pairs contains old pairs for the 
                # first half, and new pairs for the second
                old_pairs = sess_pairs[:config.nLists/2]
                new_pairs = sess_pairs[config.nLists/2:]
                
                list_types = (['OLD']*(config.nLists/3))+(['NEW']*(config.nLists/3))+\
                        (['MIXED']*(config.nLists/3))
                random.shuffle(list_types)

                mixed_pairs = []
                
                for (i,list_type) in enumerate(list_types):
                    
                    # If the list is all old or all new, just take one of those lists and add it on
                    if list_type=='OLD':
                        sess_pairs_mixed.append(old_pairs.pop())
                    elif list_type=='NEW':
                        sess_pairs_mixed.append(new_pairs.pop())
                     # If the list type is mixed, split an old and new list into half lists
                    elif list_type=='MIXED':
                        # If a mixed list already exists, use that
                        if len(mixed_pairs)>0:
                            sess_pairs_mixed.append(mixed_pairs.pop())
                        else:
                            # get one new and old old list
                            old_list = old_pairs.pop()
                            new_list = new_pairs.pop()

                            # Generates two mixed lists 
                            mixed_list_1 = []
                            mixed_list_2 = []

                            # Later, the first pair is placed close together, the second
                            # far apart. This makes it so that the temporal relatedness
                            # is independent of old/new.
                            choose_first = [random.randint(0,1) for _ in range((len(old_list)-1)/2)]

                            # Loop through the bins except for the last one, which contains random
                            # items. Will deal with that further down.
                            for i in range(len(old_list)-1):
                                
                                if i%2==choose_first[i/2]:
                                    mixed_list_1.append(old_list.pop(0))
                                    mixed_list_2.append(new_list.pop(0))
                                else:
                                    mixed_list_1.append(new_list.pop(0))
                                    mixed_list_2.append(old_list.pop(0))
                            
                            old_extra_words = old_list[len(old_list)-1]
                            new_extra_words = new_list[len(new_list)-1]

                            mixed_list_1.append(\
                                    old_extra_words[0:len(old_extra_words)/2] + \
                                    new_extra_words[0:len(new_extra_words)/2]
                                    )
                            mixed_list_2.append(\
                                    old_extra_words[len(old_extra_words)/2:] + \
                                    new_extra_words[len(new_extra_words)/2:]
                                    )
                            sess_pairs_mixed.append(mixed_list_1)
                            mixed_pairs.append(mixed_list_2)
                sess_pairs = sess_pairs_mixed
            pairs.append(sess_pairs) 
            session_num+=1
        else: # If the session did not go well, try again.
            sess_pairs = []
        print ''
    return pairs, pair_dicts

def placePairs(pairs, config):
    """
    Places each of the pairs
    """
    if config.listLength == 12 and config.numBins == 3:
        with open('valid12.json', 'r') as f:
            valid12 = json.load(f)['3bin-valid12']
    subj_wo = []
    # Loop through sessions
    # (n = sessionNum)
    for (n, sess_pairs) in enumerate(pairs):
        sess_wo = []
        print '\nPlacing session',n,':',
        sys.stdout.flush()
        # loop through lists
        # m = listNum
        if config.listLength == 12 and config.numBins == 3:
            for (m, list_pairs) in enumerate(sess_pairs):
                print m,
                sys.stdout.flush()

                # placeable pairs are the /actual/ pairs, as opposed to the last item
                # in list_pairs which is the random items
                placeable_pairs = [list(pair) for pair in list_pairs[:-1]]

                # Group the pairs, so that each pair of pairs is in a tuple
                grouped_pairs = [list(group) for group in zip([placeable_pairs[i] for i in range(len(placeable_pairs)) if i % 2 == 0],
                                [placeable_pairs[i] for i in range(len(placeable_pairs)) if i % 2 == 1])]

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

                for i in range(len(list_wo)):
                    w = list_wo[i]
                    # w is a letter from a-f followed by a number from 0-1. The letter corresponds to the word pair, the number corresponds to the item in the pair
                    list_wo[i] = key[w[0]][int(w[1])]

                sess_wo.append(list_wo)

        else:
            for (m, list_pairs) in enumerate(sess_pairs):
                print m,
                sys.stdout.flush()

                # All items in the list start out as None
                list_wo = [None]*config.listLength

                # placeable pairs are the /actual/ pairs, as opposed to the last item
                # in list_pairs which is the random items
                placeable_pairs = list_pairs[:-1]

                # Group the pairs, so that each pair of pairs is in a tuple
                grouped_pairs = zip([placeable_pairs[i] for i in range(len(placeable_pairs)) if i % 2 == 0],
                                    [placeable_pairs[i] for i in range(len(placeable_pairs)) if i % 2 == 1])

                useable_positions = range(config.listLength)
                placedAll = False

                # Loop until all of the pairs are placed
                while not placedAll:

                    # All items in the list start out as None
                    list_wo = [None]*config.listLength
                    for pairs in grouped_pairs:

                        # The close pair is always the first of the pairs, the far
                        # pair is the second
                        closePair = pairs[0]
                        farPair = pairs[1]

                        # Place close pairs
                        # list comprehension will chose places where
                        # there are adjacent usable_positions
                        possible_index1s = [i for i in range(len(useable_positions)-1) \
                                if useable_positions[i+1]==useable_positions[i]+1]

                        # If you can't place it, exit, try again
                        if len(possible_index1s)==0:
                            break

                        # Choose a position for the first of the pair randomly
                        index_1 = random.choice(possible_index1s)

                        # Place both items of the close pair in list_wo
                        list_wo[useable_positions[index_1]] = closePair[0]
                        list_wo[useable_positions[index_1+1]] = closePair[1]

                        # Those positions are no longer useable
                        del useable_positions[index_1:index_1+2]

                        #Place far pairs. Try 50 times.
                        farPlace_tries = 0
                        while farPlace_tries<50:
                            # Place the first item
                            place_1 = random.choice(useable_positions)
                            # Get all of the places that are at least 2 spots away
                            possible_place2s = [place for place in useable_positions if abs(place-place_1)>2]

                            # if there are none, try again
                            if len(possible_place2s)==0:
                                farPlace_tries+=1
                                continue
                            # Otherwise, choose one randomly
                            place_2 = random.choice(possible_place2s)
                            break
                        # I think this could be an else...
                        if farPlace_tries>=50:
                            break

                        # Place the two far pairs
                        list_wo[place_1] = farPair[0]
                        list_wo[place_2] = farPair[1]

                        # Those positions are no longer useable
                        del useable_positions[useable_positions.index(place_1)]
                        del useable_positions[useable_positions.index(place_2)]
                    else:
                        # Only runs if the loop is exited normally.
                        placedAll = True

                # For the remaining items in the list
                for i in range(len(list_wo)):
                    # If nothing has been placed there, put in one of the random words
                    if list_wo[i]==None:
                        list_wo[i] = list_pairs[len(list_pairs)-1].pop()
                sess_wo.append(list_wo)
        subj_wo.append(sess_wo)
    return subj_wo

def verifyFiles(files):
    """
    Verify that all the files specified in the config are there so
    that there is no random failure in the middle of the experiment.
    This will call sys.exit(1) if any of the files are missing.
    
    """

    for f in files:
        if not os.path.exists(f):
            print "\nERROR:\nPath/File does not exist: %s\n\nPlease verify the config.\n" % f
            sys.exit(1)

if __name__=="__main__":
    import config
    # The full wordpool (only used for indexing)
    wpfile = open(config.wpfile)
    # The wordpool for most sessions
    main_wp = open(config.wpfile_main)
    # Half of the wordpool for the last session
    # (the other half is taken from the main wordpool)
    last_sess_wp = open(config.wpfile_last_session)

    # Read in the semantic association values
    semMat = []
    wasfile = open(config.wasfile)
    for word in wasfile:
        wordVals = []
        wordValsString = word.split()
        for val in wordValsString:
            thisVal = float(val)
            wordVals.append(thisVal)
        semMat.append(wordVals)
    semArray = numpy.array(semMat)
    
    semBins = list(numpy.linspace(semArray.min(),semArray.max(),config.numBins+1))
    # Creates boundaries for the segments
    semBins = zip(*[semBins[i:]+semBins[-1:i] for i in range(2)])
    semRows = []
    semCols = []
    for bin in semBins:
        # For words within those boundaries, append the indices to
        # semRows and semCols
        (i,j) = ((semArray > bin[0]) & (semArray < bin[1])).nonzero()
        semRows.append(i)
        semCols.append(j)
    # semRows and semCols are 4xN lists, with each of the four slots in the list
    # being a semantic bin. a = semRows[x][y] and b = semCols[x][y] gives the 
    # indices (a and b) of a word in the xth semantic bin

    # Read in the wordpools
    wp_tot = numpy.array([x.strip() for x in wpfile.readlines()])
    numpy.random.shuffle(wp_tot)
    wp_allowed_main = wp_tot[:576]
    wp_allowed_last = wp_tot[-288:]
    # wp_allowed_main = [x.strip() for x in main_wp.readlines()]
    # wp_allowed_last = [x.strip() for x in last_sess_wp.readlines()]

    wp_tot = wp_tot.tolist()
    wp_allowed_main = wp_allowed_main.tolist()
    wp_allowed_last = wp_allowed_last.tolist()

    # This chooses all of the pairs to be used in the experiment
    pairs, pair_dicts = choosePairs(semArray, wp_tot, wp_allowed_main, wp_allowed_last, config, semRows, semCols)
    
    # This places the pairs in the correct location
    subj_wo = placePairs(pairs, config)
    
    #print wp_tot, subj_wo, pair_dicts, semMat


