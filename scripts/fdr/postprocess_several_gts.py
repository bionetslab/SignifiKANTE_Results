import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, f1_score, precision_score, recall_score
import pickle

def aggregate_groundtruth_results(root_dir : str,
                                  tissue_list : list,
                                  ):
    # Iterate over selected tissues in root directory.
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
    
        # Check if it's a directory
        if os.path.isdir(subdir_path) and subdir in tissue_list:
            print("Processing tissue ", subdir)
            
            batch_dir = os.path.join(subdir_path, f'batch_wise_fdr_grns')
        
            # Navigate into FDR controlled batches.
            if os.path.isdir(batch_dir):
                #print(f"Processing directory: {batch_dir}")

                # Iterate over all .csv files inside 'batch_wise_fdr_grn'
                all_dfs = []
                for file in os.listdir(batch_dir):
                    if file.startswith("fdr"):
                        file_path = os.path.join(batch_dir, file)
                        #print(f"  Reading file: {file_path}")

                        # Read CSV into DataFrame
                        df = pd.read_csv(file_path)
                        all_dfs.append(df)
            
                # Concatenate all dataframes.
                combined_df = pd.concat(all_dfs, ignore_index=True)
                #combined_df_path = os.path.join(batch_dir, "groundtruth_grn.csv")
                #combined_df.to_csv(combined_df_path, index=False)
                if combined_df['importance'].nunique() != len(combined_df):
                    print(f'Warning: tissue {subdir} does not have unique importances...')
            
                # Iterate over all runtimes txt files and compute total runtime.
                total_runtime = 0.0
                for file in os.listdir(batch_dir):
                    if file.startswith("time"):
                        time_file = os.path.join(batch_dir, file)
                        #print(f"  Reading file: {time_file}")
                        # Read runtime from batch file.
                        with open(time_file, "r") as f:
                            runtime = float(f.read().strip())
                            total_runtime += runtime
                runtime_out_file = os.path.join(batch_dir, "groundtruth_runtime.txt")
                #with open(runtime_out_file, "w") as f:
                #    f.write(f'{total_runtime}\n')
            
                # Iterate over all emissions csv files and compute total emission.
                total_emissions = 0.0
                for file in os.listdir(batch_dir):
                    if file.startswith("emissions"):
                        em_file = os.path.join(batch_dir, file)
                        print(f"  Reading file: {em_file}")
                        # Read runtime from batch file.
                        em_df = pd.read_csv(em_file)
                        if len(em_df) > 1:
                            print("Warning: emissions df has more than one row!")
                        total_emissions += em_df['emissions'][0]
                em_file_out = os.path.join(batch_dir, "groundtruth_emissions.txt")
                #with open(em_file_out, "w") as f:
                #    f.write(f'{total_emissions}\n')

def compute_approx_fdr_metrics(root_dir : str,
                               tissue_list : list,
                               non_tf_list : list,
                               tf_list : list,
                               approx_dir_name : str,
                               groundtruth_list : list):
    all_tissues_dict = dict()
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
    
        # Check if it's a directory
        if os.path.isdir(subdir_path) and subdir in tissue_list:
            print("Processing tissue ", subdir)
            approx_dir = os.path.join(subdir_path, approx_dir_name)
        
            # Navigate into FDR controlled approximation.
            if os.path.isdir(approx_dir):
                
                # Iterate over all cluster sizes.
                tissue_dict = dict()
                for num_tfs in tf_list:
                    for num_non_tfs in non_tf_list:
                        for groundtruth_dir in groundtruth_list:
                            
                            fdr_file_name = f'fdr_grn_nontf_{num_non_tfs}_numtf_{num_tfs}.csv'
                            time_file_name = f'time_nontf_{num_non_tfs}_numtf_{num_tfs}.txt'
                            em_file_name = f'emissions_nontf_{num_non_tfs}_numtf_{num_tfs}.csv'
                            fdr_path = os.path.join(approx_dir, fdr_file_name)
                            time_path = os.path.join(approx_dir, time_file_name)
                            em_path = os.path.join(approx_dir, em_file_name)
                            
                            # Compute F1, Precision, Recall on approx. FDR controlled GRNs.
                            approx_grn = pd.read_csv(fdr_path)
                            gt_path = os.path.join(subdir_path, groundtruth_dir, "aggregated_fdr_grn.csv")
                            gt_grn = pd.read_csv(gt_path)
                            #approx_grn = approx_grn.sort_values(by='importance', ascending=False)
                            #gt_grn = gt_grn.sort_values(by='importance', ascending=False)
                            merged_grn = pd.merge(approx_grn, gt_grn, on=['TF', 'target'], how='inner', suffixes=('_approx', '_gt'))
                            # Sort GRNs by importance.
                            approx_pvals = list(merged_grn['pvalue_approx'])
                            gt_pvals = list(merged_grn['pvalue_gt'])
                            mae = mean_absolute_error(gt_pvals, approx_pvals)
                            # Threshold pvalues at 005 and 001.
                            approx_005 = [1 if pval <= 0.05 else 0 for pval in approx_pvals]
                            approx_001 = [1 if pval <= 0.01 else 0 for pval in approx_pvals]
                            gt_005 = [1 if pval <= 0.05 else 0 for pval in gt_pvals]
                            gt_001 = [1 if pval <= 0.01 else 0 for pval in gt_pvals]
                            f1_005 = f1_score(gt_005, approx_005)
                            f1_001 = f1_score(gt_001, approx_001)
                            
                            # Compute runtime savings.
                            with open(time_path, "r") as f:
                                approx_runtime = float(f.read().strip())
                            gt_time_path = os.path.join(subdir_path, groundtruth_dir, "aggregated_runtime.txt")
                            with open(gt_time_path, "r") as f:
                                gt_time = float(f.read().strip())
                            absolute_time_saving = gt_time - approx_runtime
                            time_savings_factor = gt_time / approx_runtime
                            
                            # Compute emissions savings.
                            em_df = pd.read_csv(em_path)
                            if len(em_df) > 1:
                                print("Warning: emissions df has more than one row!")
                            approx_em = em_df['emissions'][0]
                            gt_em_path = os.path.join(subdir_path, groundtruth_dir, "aggregated_emissions.txt")
                            with open(gt_em_path, "r") as f:
                                gt_em = float(f.read().strip())
                            absolute_em_saving = gt_em - approx_em
                            em_savings_factor = gt_em / approx_em
                            
                            # Save results.
                            tissue_dict[(num_non_tfs, num_tfs, groundtruth_dir)] = {'mae': mae, 
                                                                   'f1_005': f1_005, 
                                                                   'f1_001': f1_001,
                                                                   'abs_time_saving': absolute_time_saving,
                                                                   'factor_time_saving': time_savings_factor,
                                                                   'abs_emission_saving': absolute_em_saving,
                                                                   'factor_emission_saving': em_savings_factor,
                                                                   'total_runtime' : approx_runtime,
                                                                   }
                    
                tissue_res_file = os.path.join(approx_dir, "approx_fdr_results.pkl")
                with open(tissue_res_file, 'wb') as f:
                    pickle.dump(tissue_dict, f)
                all_tissues_dict[subdir] = tissue_dict
    return all_tissues_dict

def pickle_results_to_df(root_dir : str,
                        tissue_list : list,
                        approx_dir_name : str
                        ):
    # Iterate over selected tissues in root directory.
    all_results_dict = {
        'tissue': [],
        'num_non_tfs': [],
        'num_tfs': [],
        'mae': [],
        'groundtruth': [],
        'f1_005': [],
        'f1_001': [],
        'abs_time_saving': [],
        'rel_time_saving': [],
        'abs_emission_saving': [],
        'rel_emission_saving': [],
        'total_runtime': []
    }
    
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
    
        # Check if it's a directory
        if os.path.isdir(subdir_path) and subdir in tissue_list:
            print("Processing tissue ", subdir)
            approx_dir = os.path.join(subdir_path, approx_dir_name)
        
            # Open aggregated pickle results file.
            pickle_file = os.path.join(approx_dir, "approx_fdr_results.pkl")
            with open(pickle_file, "rb") as f:
                res_dict = pickle.load(f)
            tissue = subdir
            # Iterate through results dict and add entries to df.
            for key, val in res_dict.items():
                num_non_tfs = key[0]
                num_tfs = key[1]
                groundtruth = key[2]
                mae = val['mae']
                f1_005 = val['f1_005']
                f1_001 = val['f1_001']
                abs_time = val['abs_time_saving']
                rel_time = val['factor_time_saving']
                abs_em = val['abs_emission_saving']
                rel_em = val['factor_emission_saving']
                total_time = val['total_runtime']
                all_results_dict['tissue'].append(tissue)
                all_results_dict['num_non_tfs'].append(num_non_tfs)
                all_results_dict['num_tfs'].append(num_tfs)
                all_results_dict['mae'].append(mae)
                all_results_dict['f1_005'].append(f1_005)
                all_results_dict['f1_001'].append(f1_001)
                all_results_dict['abs_time_saving'].append(abs_time)
                all_results_dict['rel_time_saving'].append(rel_time)
                all_results_dict['abs_emission_saving'].append(abs_em)
                all_results_dict['rel_emission_saving'].append(rel_em)
                all_results_dict['total_runtime'].append(total_time)
                all_results_dict['groundtruth'].append(groundtruth)
    
    all_res_df = pd.DataFrame(all_results_dict)
    all_res_df.to_csv(f'./random_100_targets_wasserstein_against_ten_groundtruths.csv', index=False)


if __name__ == "__main__":
    root_dir = "/home/woody/iwbn/iwbn106h/gtex_fdr_results/"

    tissue_list = [
    "Adipose_Tissue", "Adrenal_Gland", "Bladder", "Blood", "Blood_Vessel",
    "Brain", "Breast", "Cervix_Uteri", "Colon", "Esophagus",
    "Fallopian_Tube", "Heart", "Kidney", "Liver", "Lung",
    "Muscle", "Nerve", "Ovary", "Pancreas", "Pituitary",
    "Prostate", "Salivary_Gland", "Skin", "Small_Intestine", "Spleen",
    "Stomach", "Testis", "Thyroid", "Uterus", "Vagina"
    ]

    three_tissues = ['Breast', 'Kidney', 'Testis']

    # Code for aggreagting groundtruth results - only needs to be done once.
    #aggregate_groundtruth_results(root_dir,
    #                              tissue_list)
    #print("Aggregated GT results!")

    # Parameters for approx. FDR results aggregation.
    non_tf_list = [100]
    tf_list = [-1]
    groundtruth_list = [f'groundtruth_batches_{i}' for i in range(10)]
    approx_dir_name = "random_targets_wasserstein_up_to_100"
    res = compute_approx_fdr_metrics(root_dir,
                                     three_tissues,
                                     non_tf_list, 
                                     tf_list,
                                     approx_dir_name,
                                     groundtruth_list)
    
    # Transform dict-based results to DF-based plottable format.
    pickle_results_to_df(root_dir,
                         three_tissues,
                         approx_dir_name)
    
