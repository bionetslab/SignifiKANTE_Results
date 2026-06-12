import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, f1_score, precision_score, recall_score
import pickle
from statsmodels.stats.multitest import multipletests

def aggregate_groundtruth_results(root_dir : str,
                                  tissue_list : list,
                                  num_groundtruths : int = 10
                                  ):
    # Iterate over selected tissues in root directory.
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
    
        # Check if it's a directory
        if os.path.isdir(subdir_path) and subdir in tissue_list:
            print("Processing tissue ", subdir)
            
            for i in range(num_groundtruths):
                print("Processing groundtruth ", i)
                batch_dir = os.path.join(subdir_path, f'groundtruth_ridge_{i}')
            
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
                    combined_df_path = os.path.join(batch_dir, "groundtruth_grn.csv")
                    combined_df.to_csv(combined_df_path, index=False)
                
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
                    with open(runtime_out_file, "w") as f:
                        f.write(f'{total_runtime}\n')
                
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
                    with open(em_file_out, "w") as f:
                        f.write(f'{total_emissions}\n')

def compute_approx_fdr_metrics(
    root_dir: str,
    tissue_list: list,
    num_non_tfs: int,
    num_tfs: int,
    approx_dir_name: str,
    groundtruth_list: list
):
    import os
    import pickle
    import pandas as pd
    from statsmodels.stats.multitest import multipletests
    from sklearn.metrics import f1_score

    all_tissues_dict = {}

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        # Only process selected tissue directories
        if not (os.path.isdir(subdir_path) and subdir in tissue_list):
            continue

        print("Processing tissue", subdir)

        tissue_dict = {}

        # Preload all groundtruth GRNs once for this tissue
        gt_cache = {}

        for gt_idx, gt_dir in enumerate(groundtruth_list):
            gt_path = os.path.join(subdir_path, gt_dir, "groundtruth_grn.csv")

            if not os.path.exists(gt_path):
                print(f"Missing groundtruth file: {gt_path}")
                continue

            gt_grn = pd.read_csv(gt_path)

            if 'pvalue' not in gt_grn.columns:
                print(f"Missing pvalue column in {gt_path}")
                continue

            # Apply BH correction
            _, pvals_corrected_gt, _, _ = multipletests(
                gt_grn["pvalue"], method="fdr_bh"
            )
            gt_grn["pvalue_with_bh"] = pvals_corrected_gt

            gt_cache[gt_idx] = gt_grn

        # Process each groundtruth as reference
        for firstGT in gt_cache.keys():
            gt_grn = gt_cache[firstGT]

            # -------------------------------------------------
            # Compare approximate GRN vs this groundtruth
            # -------------------------------------------------
            approx_dir = os.path.join(subdir_path, approx_dir_name)
            fdr_path = os.path.join(
                approx_dir,
                f"fdr_grn_nontf_{num_non_tfs}_numtf_{num_tfs}.csv"
            )

            if os.path.exists(fdr_path):
                approx_grn = pd.read_csv(fdr_path)

                if 'pvalue' in approx_grn.columns:
                    _, pvals_corrected_approx, _, _ = multipletests(
                        approx_grn["pvalue"], method="fdr_bh"
                    )
                    approx_grn["pvalue_with_bh"] = pvals_corrected_approx

                    merged_grn = pd.merge(
                        approx_grn,
                        gt_grn,
                        on=["TF", "target"],
                        how="inner",
                        suffixes=("_approx", "_gt")
                    )

                    if not merged_grn.empty:
                        # Raw p-value threshold
                        approx_005 = (
                            merged_grn["pvalue_approx"] <= 0.05
                        ).astype(int)
                        gt_005 = (
                            merged_grn["pvalue_gt"] <= 0.05
                        ).astype(int)

                        f1_005 = f1_score(
                            gt_005,
                            approx_005,
                            zero_division=0
                        )

                        # BH corrected threshold
                        approx_005_bh = (
                            merged_grn["pvalue_with_bh_approx"] <= 0.05
                        ).astype(int)
                        gt_005_bh = (
                            merged_grn["pvalue_with_bh_gt"] <= 0.05
                        ).astype(int)

                        f1_005_bh = f1_score(
                            gt_005_bh,
                            approx_005_bh,
                            zero_division=0
                        )

                        tissue_dict[(f"gt-{firstGT}", "approx")] = {
                            "f1_005": f1_005,
                            "f1_005_bh": f1_005_bh
                        }

            else:
                print(f"Missing approximation file: {fdr_path}")

            # -------------------------------------------------
            # Compare this groundtruth to all other groundtruths
            # -------------------------------------------------
            for secondGT in gt_cache.keys():
                if firstGT == secondGT:
                    continue

                comparison_grn = gt_cache[secondGT]

                merged_grn = pd.merge(
                    comparison_grn,
                    gt_grn,
                    on=["TF", "target"],
                    how="inner",
                    suffixes=("_approx", "_gt")
                )

                if merged_grn.empty:
                    continue

                # Raw p-value threshold
                approx_005 = (
                    merged_grn["pvalue_approx"] <= 0.05
                ).astype(int)
                gt_005 = (
                    merged_grn["pvalue_gt"] <= 0.05
                ).astype(int)

                f1_005 = f1_score(
                    gt_005,
                    approx_005,
                    zero_division=0
                )

                # BH corrected threshold
                approx_005_bh = (
                    merged_grn["pvalue_with_bh_approx"] <= 0.05
                ).astype(int)
                gt_005_bh = (
                    merged_grn["pvalue_with_bh_gt"] <= 0.05
                ).astype(int)

                f1_005_bh = f1_score(
                    gt_005_bh,
                    approx_005_bh,
                    zero_division=0
                )

                tissue_dict[(f"gt-{firstGT}", f"gt-{secondGT}")] = {
                    "f1_005": f1_005,
                    "f1_005_bh": f1_005_bh
                }

        # -------------------------------------------------
        # Save results for this tissue
        # -------------------------------------------------
        approx_dir = os.path.join(subdir_path, approx_dir_name)
        os.makedirs(approx_dir, exist_ok=True)

        tissue_res_file = os.path.join(
            approx_dir,
            "gt_gt_results.pkl"
        )

        with open(tissue_res_file, "wb") as f:
            pickle.dump(tissue_dict, f)

        all_tissues_dict[subdir] = tissue_dict

def pickle_results_to_df(root_dir : str,
                        tissue_list : list,
                        approx_dir_name : str
                        ):
    # Iterate over selected tissues in root directory.
    all_results_dict = {
        'tissue' : [],
        'groundtruth': [],
        'approximation': [],
        'f1_005': [],
        'f1_005_bh': []
    }
    
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
    
        # Check if it's a directory
        if os.path.isdir(subdir_path) and subdir in tissue_list:
            print("Processing tissue ", subdir)
            approx_dir = os.path.join(subdir_path, approx_dir_name)
        
            # Open aggregated pickle results file.
            pickle_file = os.path.join(approx_dir, "gt_gt_results.pkl")
            with open(pickle_file, "rb") as f:
                res_dict = pickle.load(f)
            tissue = subdir
            # Iterate through results dict and add entries to df.
            for key, val in res_dict.items():
                groundtruth = key[0]
                approximation = key[1]
                f1_005 = val['f1_005']
                f1_005_bh = val['f1_005_bh']

                all_results_dict['tissue'].append(tissue)
                all_results_dict['groundtruth'].append(groundtruth)
                all_results_dict['approximation'].append(approximation)
                all_results_dict['f1_005'].append(f1_005)
                all_results_dict['f1_005_bh'].append(f1_005_bh)
    
    all_res_df = pd.DataFrame(all_results_dict)
    return all_res_df


if __name__ == "__main__":
    root_dir = "/home/woody/iwbn/iwbn106h/gtex_fdr_results/"

    #tissue_list = [
    #"Adipose_Tissue", "Adrenal_Gland", "Bladder", "Blood", "Blood_Vessel",
    #"Brain", "Breast", "Cervix_Uteri", "Colon", "Esophagus",
    #"Fallopian_Tube", "Heart", "Kidney", "Liver", "Lung",
    #"Muscle", "Nerve", "Ovary", "Pancreas", "Pituitary",
    #"Prostate", "Salivary_Gland", "Skin", "Small_Intestine", "Spleen",
    #"Stomach", "Testis", "Thyroid", "Uterus", "Vagina"
    #]

    single_tissue = ['Kidney']

    # Code for aggreagting groundtruth results - only needs to be done once.
    #aggregate_groundtruth_results(root_dir,
    #                              single_tissue)
    #print("Aggregated GT results!")
    #quit()

    # Parameters for approx. FDR results aggregation.
    num_non_tfs = 100
    num_tfs = -1
    groundtruth_list = [f'groundtruth_ridge_{i}' for i in range(10)]
    approx_dir_name = "random_100_targets_ridge"
    res = compute_approx_fdr_metrics(root_dir,
                                     single_tissue,
                                     num_non_tfs, 
                                     num_tfs,
                                     approx_dir_name,
                                     groundtruth_list)
    
    # Transform dict-based results to DF-based plottable format.
    all_res_df = pickle_results_to_df(root_dir,
                         single_tissue,
                         approx_dir_name)
    all_res_df.to_csv('gt_variation_ridge_results.csv', index=False)
    
