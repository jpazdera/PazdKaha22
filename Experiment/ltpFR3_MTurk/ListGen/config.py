# Configuration for Long Term Participants Free Recall 3

"""
This module sets options for running ltpFR3.  This experiment was specifically designed to run in the PEERS project
at the Computational Memory Lab.  Most of the setting are fairly hardcoded in the file session_config.py.  Do not
change anything in either file unless you are sure of what you are doing.
"""

# Session configuration
nSessions = 1  # Number of sessions
reps = 1  # Number of repetitions of each list type in each session

# List Conditions
listLength = (12, 24)  # List lengths in numbers of words
presRate = (800, 1600)  # Presentation rates in milliseconds
modality = ('a', 'v')  # Modalities of presentation - auditory or visual
distDur = (12000, 24000)  # Distractor period durations in milliseconds

# Important file paths
wpfile = 'wordpool_ltpFR3.txt'  # ltpFR3 word pool
w2vfile = 'w2v_scores_ltpFR3.txt'  # WAS scores for all word pairs
