# Configuration for Long Term Participants Free Recall

"""
This module sets options for running ltpFR.  This experiment was
specifically designed to run in the ltp (Long Term Participants)
project at the Computational Memory Lab.  Most of the setting are
fairly hardcoded in the file session_config.py.  Do not change anything
in either file unless you are sure of what you are doing.
"""

# experiment structure
nSessions = 24
adjustTrials = [8, 16]  # after this trial, get chance to rewet the cap
listLength = 24
nLists = 24

# stimuli and related statistics
wpfile = '/Users/jessepazdera/svn/ltpFR2/trunk/exp/pools/wasnorm_wordpool.txt'
tnfile = '/Users/jessepazdera/svn/ltpFR2/trunk/exp/pools/wasnorm_task.txt'
wasfile = '/Users/jessepazdera/svn/ltpFR2/trunk/exp/pools/wasnorm_was.txt'
wpfile_main = '/Users/jessepazdera/svn/ltpFR2/trunk/exp/pools/wordpool_main.txt'
wpfile_last_session = '/Users/jessepazdera/svn/ltpFR2/trunk/exp/pools/wordpool_last_session.txt'
# list creation settings
trainSets = [[[2, 2, 4], [2, 3, 3]], [[2, 6], [3, 5], [4, 4]]]
ratingRange = (.3, .7)
WASthresh = .55
maxTries = 2500
numBins = 3

# stimulus display settings
taskText = ['Size', 'Living/Nonliving']
showTaskText = False

taskColors = [(.4226, 0.5871, 0), (0.5774, 0.4129, 1.0000)]
taskFonts = ['../fonts/amplitud.ttf', '../fonts/zerothre.ttf']
taskCases = ['upper', 'lower']
doTaskFonts = True
doTaskColors = True
doTaskCases = True
recallStartText = '*******'
wordHeight = .08  # Word Font size (percentage of vertical screen)
defaultFont = '../fonts/Verdana.ttf'

# study response keys
# K: big
# L: small
# ,: living
# .: nonliving
respPool = {'K': (0, 1), 'L': (0, 0), ',': (1, 1), '.': (1, 0)}

# button to pause during wait screen
pauseButton = 'SPACE'

# math distractor settings
doMathDistract = True
MATH_numVars = 3
MATH_maxNum = 9
MATH_minNum = 1
MATH_plusAndMinus = False
MATH_displayCorrect = True
MATH_textSize = .1
MATH_correctBeepDur = 500
MATH_correctBeepFreq = 400
MATH_correctBeepRF = 50
MATH_correctSndFile = None
MATH_incorrectBeepDur = 500
MATH_incorrectBeepFreq = 200
MATH_incorrectBeepRF = 50
MATH_incorrectSndFile = None
MATH_practiceDisplayCorrect = True

# Instructions text files
introFirstSess = 'text/introFirstSess.txt'
introLists = 'text/introLists.txt'
introMathPractice = 'text/introMathPractice.txt'
introRecall = 'text/introRecall.txt'
introSummary = 'text/introSummary.txt'
introSummaryMath = 'text/introSummaryMath.txt'
introGetReady = 'text/introGetReady.txt'
interimGetReady = 'text/interimGetReady.txt'
instructTask = 'text/instructTask.txt'
introOtherSess = 'text/introOtherSess.txt'

trialBreak = 'text/trialBreak.txt'
midSessionBreak = 'text/midSessionBreak.txt'
recogFastMsg = 'text/recogFastMsg.txt'

instructEFR = 'text/EFR_extra_instruct.txt'
introOtherSessEFR = 'text/introOtherSessEFR.txt'

# Files needed to run the experiment
files = (wpfile,
         tnfile,
         wasfile,
         wpfile_main,
         wpfile_last_session,
         introFirstSess,
         introLists,
         introMathPractice,
         introRecall,
         introSummary,
         introSummaryMath,
         introGetReady,
         introOtherSess,
         trialBreak,
         midSessionBreak,
         defaultFont)

# create empty file to indicate end of session
# used in conjunction with ltpFR.sh to automatically upload data to rhino
writeEndFile = True

# experiment timing
fastConfig = False

if fastConfig:  # run fast version to quickly check for errors
    # timing of recall trials
    wordDuration = 3
    msgDur = 3
    wordISI = 8
    jitter = 40

    trialBreakTime = 1000
    preListDelay = 150
    preRecallDelay = 120
    jitterBeforeRecall = 20
    recallDuration = 300

    # math timing
    MATH_minProblemTime = 10
    MATH_minDelay = 20
    MATH_practiceMaxDistracterLimit = 1000
    MATH_preDistractLength = 200
    MATH_eolDistractLength = 200
    # list of the distractor conditions; ISI, RI tuples

    distractLens = ((0, 0), (0, 8000), (0, 16000), (8000, 8000), (16000, 16000))

    # final free recall
    preffrDelay = 500
    ffrDuration = 5000


else:  # run actual experiment
    # timing of recall trials
    wordDuration = 1600
    msgDur = 1500
    wordISI = 800
    jitter = 400

    trialBreakTime = 10000
    preListDelay = 1500
    preRecallDelay = 1200
    jitterBeforeRecall = 200
    recallDuration = 75000

    # math timing
    MATH_minProblemTime = 1500
    MATH_minDelay = 6000
    MATH_practiceMaxDistracterLimit = 12000
    MATH_preDistractLength = 24000
    MATH_eolDistractLength = 24000

# Beep at start and end of recording (freq,dur,rise/fall)
startBeepFreq = 800
startBeepDur = 500
startBeepRiseFall = 100
stopBeepFreq = 400
stopBeepDur = 500
stopBeepRiseFall = 100
countdownBeepShortFreq = 400
countdownBeepShortDur = 400
countdownBeepShortRiseFall = 10
countdownBeepLongFreq = 600
countdownBeepLongDur = 1000
countdownBeepLongRiseFall = 100

# Realtime configuration
# ONLY MODIFY IF YOU KNOW WHAT YOU ARE DOING!
# HOWEVER, IT SHOULD BE TWEAKED FOR EACH MACHINE
doRealtime = False
rtPeriod = 120
rtComputation = 9600
rtConstraint = 1200

# Have subject recall all items coming to mind, pressing space if they belive an item they said wasn't on a list.
totalRecall = False
