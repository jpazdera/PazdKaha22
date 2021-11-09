This experiment is built using PsiTurk in conjunction with jsPsych.
For an overview of the structure of PsiTurk experiments:
https://psiturk.readthedocs.io/en/latest/index.html
For an overview of the structure of jsPsych experiments:
https://www.jspsych.org/

/static/
  audio/
    - wordpool/: A folder containing audio recordings of all words in the wordpool
    - 400Hz.wav: A 400 Hz tone that plays to signal the start of the recall period
  css/
    - bootstrap.min.css: Formatting from the Bootstrap library
    - custom_formatting.css: Custom reformatting for text in experiments
    - jspsych.css: jsPsych related formatting
    - style.css: PsiTurk related formatting
    - survey_style.css: Formatting for post-experiment survey pages
  js/
    - jspsych/: Folder containing the standard jsPsych library
    - audio-test.js: Custom jsPsych plugin for running the audio test used in both experiments
    - countdown.js: Custom jsPsych plugin for displaying 10-second countdowns
    - free-recall.js: Custom jsPsych plugin for collecting free recall responses
    - ltpFR3.js: Main script for running Experiment 1
    - ltpFR3v2.js: Main script for running Experiment 2
    - math-distractor.js: Custom jsPsych plugin for running math distractor tasks (A+B+C=?)
    - supplementary-functions.js: Contains four miscellaneous functions
      - randomInt: Generates a random integer in a given range (used for generating math problems)
      - saveData: Used for passing session data to the Python save() function in custom.py, in order
        to write log files
      - loadListsAndRun, loadListsAndRunV2: These functions are called from exp.html. They load
        the word lists for a given session number, and then trigger ltpFR3.js or ltpFR3v2.js only once
        the word lists have finished loading. If list-loading and the experiment were instead called
        directly in exp.html using script tags as shown below, ltpFR3.js would start running while
        the word lists were still loading, which would crash the experiment:
        <script src="static/pools/lists/0.js" type="text/javascript"></script>
        <script src="/static/js/ltpFR3.js" type="text/javascript"></script>
    - utils.js: A few utility functions used by PsiTurk
  php/
    - save_data.php: Deprecated function used for saving session data. Replaced by save() in custom.py.
  pools/
    - lists/: Folder containing JSON files with pre-generated word lists for each session
    - PEERS_wordpool.js: The full PEERS wordpool of 1768 items
    - wordpool_ltpFR3.txt: The 552-item subest of the PEERS wordpool that was used for
      this experiment

/templates/
  - ad.html: Template for the ad shown to participants when previewing the HIT.
  - complete.html: Unused except when running PsiTurk server in cabin mode.
  - consent.html: Template for the consent form page for the experiment.
  - default.html: Typically unused page that just redirects to ad.html
  - error.html: Page for displaying custom error messages. IMPORTANT: If you are hosting your ad
    through PsiTurk's ad servers, errors will direct participants to PsiTurk's own default error
    page, and your custom error messages WILL NOT be shown. The messages defined here only appear
    when you are running in cabin mode or if you are self-hosting your ad.
  - exp.html: Main page that loads all the JavaScript and CSS used in the experiment, then runs
    the script for the experiment via loadListsAndRun.
  - survey.html: Post-experiment survey page for Experiment 1
  - survey2.html: Post-experiment survey page for Experiment 2
  - thanks.html: I believe this file is unused, and is a holdover from the basic PsiTurk experiment
    template

/ListGen/: Contains the Python code we used for generating word lists for ltpFR3. We controlled the
semantic relatedness of words in the list, ensuring that every list had some highly related words
and some dissimilar words positioned both nearby and at distant serial positions.
