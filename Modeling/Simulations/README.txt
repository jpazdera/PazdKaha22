SIMULATION FILE SUMMARY

Simulation Naming Scheme:

Files labeled "sim1" manage the fitting of CMR2 to the pooled-modality data. Files labeled "sim2" 
conduct grid searches to determine the pairing of beta_enc and gamma_CF that best fit the modality-
specific start-conditional SPCs. Files labeled "a", "b", and "c" handle simulations for 12, 24, and
6-item lists, respectively. Thus, sim1a and sim2a equate to Simulations 1 and 2 from our manuscript, 
respectively. Meanwhile sim1b, sim2b, sim1c, and sim2c collectively make up Simulation 3 from our
manuscript.

File Contents:

Note that the particle swarm optimization itself was run outside of these notebooks through bash
commands and Python scripts located and documented in the ../CMR2/ folder. The sim1* notebooks
just prepare the target behavior files and the random 24- and 6-item lists for the particle swarm,
read and plot its various outputs, and run the final comparison of the best parameter set identified
by each particle. The sim2* notebooks, in contrast, comprise the entirety of the modality effect
grid search. The sim1b_lists.json and sim1c_lists.json files contain the randomly generated word
lists used in Simulation 3.