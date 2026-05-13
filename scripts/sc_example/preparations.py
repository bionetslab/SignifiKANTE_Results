
def check_datasets():
    """
    Check available datasets for total size and size of annotated cell subpopulations.
    """

    import scanpy as sc

    data_paths = [
        './data/10x-rep1-kallisto-cellbender/10x-rep1-kallisto-cellbender.h5ad',
        './data/10x-rep2-kallisto-cellbender/10x-rep2-kallisto-cellbender.h5ad',
        './data/bd-rhap-rep1/bd-rhap-rep1.h5ad',
        './data/bd-rhap-rep2/bd-rhap-rep2.h5ad',
    ]

    for data_path in data_paths:

        adata = sc.read_h5ad(data_path)

        print(data_path)
        print(adata.shape)
        print(adata.obs['celltype_semi_manual'].value_counts())
        print('\n')


def curate_data():
    """
    For selected dataset (bd-rhap-rep1), subset to 3 cell subpopulations (NK, DC, T),
    scale to zero mean and unit variance, and save to CSV.
    """

    import os
    import scanpy as sc
    from scipy.sparse import issparse

    save_path = './data/processed'
    os.makedirs(save_path, exist_ok=True)

    data_path = './data/bd-rhap-rep1/bd-rhap-rep1.h5ad'

    sub_populations = ['nk_cells', 'DC', 'cd8+_tcells']

    adata = sc.read_h5ad(data_path)

    for sub_population in sub_populations:

        # Subset
        adata_sub = adata[adata.obs['celltype_semi_manual'] == sub_population, :].copy()

        # Convert to dense
        if issparse(adata_sub.X):
            adata_sub.X  =adata_sub.X.toarray()

        # Z-score normalization
        sc.pp.scale(adata_sub, zero_center=True, max_value=None)

        # Save to CSV
        df = adata_sub.to_df()
        df.to_csv(os.path.join(save_path, sub_population.lower() + '.csv'), index=False)


def compute_input_grns():
    """
    Compute inout GRNs for downstream analyses.
    """

    import os
    import pandas as pd

    from signifikante.algo import grnboost2

    num_grns = 10
    sub_populations = ['nk_cells', 'dc', 'cd8+_tcells']

    data_dir = './data/processed'
    res_path = './input_grns'
    os.makedirs(res_path, exist_ok=True)

    for sub_population in sub_populations:

        # Load processed expression data
        expression_df = pd.read_csv(os.path.join(data_dir, sub_population + '.csv'))

        # Load the TF list obtained from (https://resources.aertslab.org/cistarget/tf_lists/)
        with open('./data/allTFs_hg38.txt', 'r', encoding='utf-8') as f:
            tf_list = [line.rstrip('\n') for line in f]

        # Check overlap with genes present in dataset
        genes_list = set(expression_df.columns.tolist())
        intersection = set(tf_list).intersection(genes_list)

        if len(intersection) == 0:
            print(f'No TFs found for {sub_population}. Continue.')
            continue

        for grn_id in range(num_grns):

            print(f'# --- ({grn_id}/{num_grns}) Computing GRN for {sub_population} --- #')

            grn = grnboost2(
                expression_data=expression_df.copy(),
                gene_names=None,
                tf_names=tf_list,
                seed=42 + grn_id,
                verbose=False,
            )

            grn_fn = f'grn_{sub_population}_{grn_id:02d}.csv'
            grn.to_csv(os.path.join(res_path, grn_fn), index=False)


def generate_configs_ground_truth():
    """
    Generate the config files for the groundtruth computation for each dataset, for each input GRN.
    """

    import os
    import yaml

    num_runs = 10
    sub_populations = ['nk_cells', 'dc', 'cd8+_tcells']
    num_permutations = [1000, 2000, 3000, 4000]

    config_dir = './configs_ground_truth'
    os.makedirs(config_dir, exist_ok=True)

    data_dir = './data/processed'
    grn_dir = './input_grns'
    results_dir = './results_ground_truth'

    for sub_population in sub_populations:
        for run_id in range(num_runs):
            for k in num_permutations:

                config = {
                    'dataset_name': sub_population,
                    'run_id': run_id,
                    'num_permutations': k,
                    'data_path': os.path.join(data_dir, sub_population + '.csv'),
                    'grn_path': os.path.join(grn_dir, f'grn_{sub_population}_00.csv'),
                    'result_dir': results_dir,
                }

                config_fn = f'config_{sub_population}_num_permut_{k:04d}_{run_id:02d}.yaml'

                with open(os.path.join(config_dir, config_fn), 'w') as f:
                    yaml.dump(config, f)


def generate_configs_approx():
    """
    Generate the config files for the approximate FDR computation
    for each dataset, for each input GRN, and for each number of clusters.
    """

    import os
    import yaml

    num_grns = 10
    sub_populations = ['nk_cells', 'dc', 'cd8+_tcells']
    num_clusters = list(range(1, 11)) + list(range(20, 101, 10))
    num_permutations = [1000, 2000, 3000, 4000]

    config_dir = './configs_approx'
    os.makedirs(config_dir, exist_ok=True)

    data_dir = './data/processed'
    grn_dir = './input_grns'
    results_dir = './results_approx'

    for sub_population in sub_populations:
        for grn_id in range(num_grns):
            for l in num_clusters:
                for k in num_permutations:
                    config = {
                        'dataset_name': sub_population,
                        'grn_id': grn_id,
                        'num_clusters': l,
                        'num_permutations': k,
                        'data_path': os.path.join(data_dir, sub_population + '.csv'),
                        'grn_path': os.path.join(grn_dir, f'grn_{sub_population}_{grn_id:02d}.csv'),
                        'result_dir': results_dir,
                    }

                    config_fn = f'config_{sub_population}_num_clust_{l:03d}_num_permut_{k:04d}_grn_id_{grn_id:02d}.yaml'

                    with open(os.path.join(config_dir, config_fn), 'w') as f:
                        yaml.dump(config, f)


if __name__ == '__main__':

    # check_datasets()

    # curate_data()

    # compute_input_grns()

    # generate_configs_ground_truth()

    # generate_configs_approx()

    print('done')






