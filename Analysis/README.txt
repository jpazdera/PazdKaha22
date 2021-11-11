ANALYSIS FILE SUMMARY

Plotting.ipynb:
Contains code for generating the figures found in the paper. Note that the code that calculates the 
behavioral metrics in the figures can be found in the Processing directory. The code that plots the 
figures reads the precalculated average stats from the all_v1_excl_wn.json and all_v2_excl_wn.json 
files in Data/stats/.

json_to_csv.ipynb:
Contains code for aggregating data from all participants' data and stats JSON files (see Data 
directory) into a table of recall data and a table of intrusion data, to be analyzed in R.

ltpFR3_stats.R:
Contains code for all statistical analyses of Experiments 1 and 2.
