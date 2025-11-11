import pandas as pd
import os
import pickle
from statsmodels.stats.multitest import multipletests

def print_num_tfs(directory):
    for tissue_dir in os.listdir(directory):
        full_path = os.path.join(directory, tissue_dir)
        if os.path.isdir(full_path):
            tsv_file = os.path.join(full_path, f'{tissue_dir}.tsv')
            target_genes_file = os.path.join(full_path, f'{tissue_dir}_target_genes.tsv')

            if os.path.isfile(tsv_file) and os.path.isfile(target_genes_file):
                # Count columns in .tsv file (header line)
                exp_df = pd.read_csv(tsv_file, sep='\t', index_col=0)
                num_genes = len(exp_df.columns)

                # Count lines in target_genes file
                targets_df = pd.read_csv(target_genes_file, index_col=0)
                num_targets = len(targets_df['target_gene'])

                num_tfs = num_genes - num_targets

                print(f'{tissue_dir}: number of TFs = {num_tfs}')
            else:
                print(f"Missiang files for {tissue_dir}")

def compute_gtex_data_statistics(exp_data_dir,
                                 grn_data_dir):
    results_dict = {'tissue' : [], 'samples' : [], 'genes' : [], 'grnboost_edges' : [], 'signif_edges' : []}
    
    for tissue in os.listdir(exp_data_dir):
        print("Processing tissue ", tissue)
        full_path = os.path.join(exp_data_dir, tissue)
        if os.path.isdir(full_path):
            exp_file = os.path.join(full_path, f'{tissue}.tsv')

            if os.path.isfile(exp_file):
                # Count samples and genes based on expression matrix.
                exp_df = pd.read_csv(exp_file, sep='\t', index_col=0)
                num_samples = len(exp_df.index)
                num_genes = len(exp_df.columns)
                
                # Count number of edges and significant edges based on GRN files.
                approx_file = os.path.join(grn_data_dir, tissue, "random_targets_wasserstein_up_to_100", "fdr_grn_nontf_100_numtf_-1.csv")
                approx_df = pd.read_csv(approx_file)
                num_edges = len(approx_df)
                # Compute significant edges with BH-correction.
                _, pvals_corrected_approx, _, _ = multipletests(approx_df['pvalue'], method='fdr_bh')
                approx_df['pvalue_bh'] = pvals_corrected_approx
                signif_df = approx_df[approx_df['pvalue_bh']<=0.05]
                num_signif_edges = len(signif_df)

                results_dict['tissue'].append(tissue)
                results_dict['samples'].append(num_samples)
                results_dict['genes'].append(num_genes)
                results_dict['grnboost_edges'].append(num_edges)
                results_dict['signif_edges'].append(num_signif_edges)
    
    return results_dict


if __name__ == "__main__":
    exp_data_dir = '/home/woody/iwbn/iwbn106h/gtex'
    grn_data_dir = '/home/woody/iwbn/iwbn106h/gtex_fdr_results'
    
    gtex_stats_dict = compute_gtex_data_statistics(exp_data_dir,
                                                   grn_data_dir)
    
    gtex_stats_df = pd.DataFrame(gtex_stats_dict)
    gtex_stats_df.to_csv("gtex_tissue_statistics.csv", index=False)

