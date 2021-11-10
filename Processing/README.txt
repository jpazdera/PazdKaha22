PROCESSING PIPELINE SUMMARY

During data collection, we automatically ran process_ltpFR3_MTurk.py hourly. This script has six
major components. 

1) Extracting session logs from the SQLite database (if necessary) and sorting session logs 
    ("events" files). This process is handled by psiturk_tools.load_psiturk_data().

2) Parse session logs and automatically score recall performance. Organize presentation and recall
    data from the session log into behavioral matrices expected by the Python Behavioral Toolbox 
    (pybeh). Store this information in JSON files labeled "data". This process is handled by 
    psiturk_tools.process_psiturk_data().

3) Extract survey responses from session logs and organize them into a single spreadsheet. Identify
    any participants who reported writing notes as a memory strategy. This process is handled by
    survey_processing.process_survey().
    
4) Automatically calculate a wide range of behavioral performance statistics using a combination
    of pybeh functions and custom scripts. Store this information in JSON files labeled "stats".
    Then calculate average behavioral stats across all participants. This process is handled by 
    run_stats.run_stats(). (See https://github.com/pennmem/pybeh for the Python Behavioral Toolbox).

5) Generate a PDF report for each participant, along with an aggregate report across all
    participants. These reports contain a variety of the stats calculated in Component 4, and allow
    quick assessment of whether each participant properly focused on the task. This process is
    handled by report.ltpFR3_report().

6) Sort files for excluded, rejected, and bad session participants into appropriately named folders.
    This process is handled by sort_excluded_files.sort_excluded_files().
    
Note that Components 1 and 2 were modified from code by M. Karl Healey. The original code repository 
for the data processing pipleline can be found at https://github.com/pennmem/ltpFR3_MTurk_postprocess.
