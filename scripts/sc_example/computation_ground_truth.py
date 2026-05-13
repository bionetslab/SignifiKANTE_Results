
import os
import yaml
import argparse
import pandas as pd

from typing import Dict
from distributed import LocalCluster, Client
from codecarbon import OfflineEmissionsTracker
from tracking import scalability_wrapper
from signifikante.algo import signifikante_fdr


def compute_classical_fdr(config: Dict):
    """
    Run classical FDR according to settings specified in config.
    """

    dataset_name = config['dataset_name']
    run_id = config['run_id']
    k = config['num_permutations']
    data_path = config['data_path']
    grn_path = config['grn_path']
    result_dir = config['result_dir']

    os.makedirs(result_dir, exist_ok=True)

    expression_mat = pd.read_csv(data_path)
    grn_input = pd.read_csv(grn_path)

    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    local_cluster = LocalCluster(
        n_workers=n_cpus,
        threads_per_worker=1,
        processes=True,
    )
    client = Client(local_cluster)

    print(client)
    print(client.ncores())
    print(f'Number of workers: {len(client.scheduler_info()["workers"])}')

    def run():
        grn_fdr = signifikante_fdr(
            expression_data=expression_mat,
            cluster_representative_mode='all_genes',
            num_target_clusters=-1,
            num_tf_clusters=-1,
            input_grn=grn_input,
            tf_names=None,
            target_subset=None,
            client_or_address=client,
            seed=42 + run_id,
            verbose=False,
            num_permutations=k,
            output_dir=None,
            scale_for_tf_sampling=True,
            inference_mode='grnboost2',
            apply_bh_correction=True,
            normalize_gene_expression=False,
        )
        return grn_fdr

    fn_emissions = os.path.join(result_dir, f'emissions_{dataset_name}_num_permut_{k:04d}_{run_id:02d}.csv')

    try:

        with OfflineEmissionsTracker(
                country_iso_code='DEU', output_file=fn_emissions, log_level='error', measure_power_secs=600
        ) as tracker:

            wall_time, mem_samples, grn = scalability_wrapper(
                function=run, function_params=None, tracking_interval=0.1
            )

            grn.to_csv(os.path.join(result_dir, f'grn_{dataset_name}_num_permut_{k:04d}_{run_id:02d}.csv'), index=False)

            mem_peak = max(mem_samples)
            num_samples = len(mem_samples)
            mem_avg = sum(mem_samples) / num_samples

            tracking_df = pd.DataFrame([{
                'dataset': dataset_name,
                'run_id': run_id,
                'num_permutations': k,
                'wall_time': wall_time,
                'mem_peak': mem_peak,
                'mem_avg': mem_avg,
                'num_samples': num_samples,
            }])
            tracking_df.to_csv(os.path.join(result_dir, f'tracking_{dataset_name}_num_permut_{k:04d}_{run_id:02d}.csv'), index=False)

    finally:
        client.close()
        local_cluster.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.f, 'r') as f:
        cfg = yaml.safe_load(f)

    compute_classical_fdr(config=cfg)

    print('done')






