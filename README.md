# Analyses and results of SignifiKANTE

## Data
All simulated datasets used in the analysis of SignifiKANTE can be found in the directory `data/sc_simulated`. GRNBoost2-based reference GRNs on all thirty GTEx tissues in combination with "groundtruth" (DIANE-like) P-values and approximate P-values computed by SignifiKANTE with 100 target gene clusters are publicly accesible on Zenodo (https://doi.org/10.5281/zenodo.17581025).

## Scripts
Code for reproducing our simulated datasets can be found under `scripts/simulate_sc_data`. All scripts used for running SignifiKANTE on our HPC systems are conatined in `scripts/fdr`. All notebooks and scripts used for carrying out experiments and analyses on SingifiKANTE's FDR controlled GRNs are located in `scripts/analysis`. All plots shown in our manuscript were generated with notebooks located in `scripts/plotting`.

## Results
All our produced results underlying the plots of our manuscript are stored in `results/`, while results on simulated data are separated from all GTEx-based results in the subdirectory `results/sc_simulated_data`.
