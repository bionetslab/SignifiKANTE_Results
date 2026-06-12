import pandas as pd
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr

if __name__ == "__main__":
    k_list = list(range(100, 1001, 100))
    result_dir = "misc_results/"
    
    bto_df = pd.read_csv("/home/hpc/iwbn/iwbn106h/Projects/SignifiKANTE_Results.git/scripts/analysis/tissue_distances_bto.csv", index_col=0)
    bto_idx = np.triu_indices_from(bto_df, k=1)
    bto_dists = bto_df.values[bto_idx]
    
    result_dict = {'k': [], 'type': [], 'spearman_corr': [], 'pvalue': []}
    
    for k in k_list:
        grnboost_file = os.path.join(result_dir, f"tissue_distances_all_edges_top_{k}.csv")
        grnboost_df = pd.read_csv(grnboost_file, index_col=0)
        
        signifikante_file = os.path.join(result_dir, f"tissue_distances_significant_approx_top_{k}.csv")
        signifikante_df = pd.read_csv(signifikante_file, index_col=0)
        
        groundtruth_file = os.path.join(result_dir, f"tissue_distances_significant_groundtruth_top_{k}.csv")
        groundtruth_df = pd.read_csv(groundtruth_file, index_col=0)
        
        # Extract upper diagonal from distance matrix.
        grnboost_idx = np.triu_indices_from(grnboost_df, k=1)
        grnboost_dists = grnboost_df.values[grnboost_idx]

        groundtruth_idx = np.triu_indices_from(groundtruth_df, k=1)
        groundtruth_dists = groundtruth_df.values[groundtruth_idx]

        signifikante_idx = np.triu_indices_from(signifikante_df, k=1)
        signifikante_dists = signifikante_df.values[signifikante_idx]
        
        # Compute correlation values.
        corr_grnboost = spearmanr(bto_dists, grnboost_dists)[0]
        corr_groundtruth = spearmanr(bto_dists, groundtruth_dists)[0]
        corr_signifikante = spearmanr(bto_dists, signifikante_dists)[0]
        
        pvalue_grnboost = spearmanr(bto_dists, grnboost_dists)[1]
        pvalue_groundtruth = spearmanr(bto_dists, groundtruth_dists)[1]
        pvalue_signifikante = spearmanr(bto_dists, signifikante_dists)[1]
        
        result_dict['k'].extend([k, k, k])
        result_dict['type'].extend(['GRNBoost2', 'DIANE-like', 'SignifiKANTE'])
        result_dict['spearman_corr'].extend([corr_grnboost, corr_groundtruth, corr_signifikante])
        result_dict['pvalue'].extend([pvalue_grnboost, pvalue_groundtruth, pvalue_signifikante])
    
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(os.path.join(result_dir, "tissue_distance_correlations_top_100_to_1000.csv"), index=False)
        
        
        
        