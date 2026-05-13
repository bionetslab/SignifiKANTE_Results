import pandas as pd
from arboreto.fdr_utils import compute_wasserstein_distance_matrix, cluster_genes_to_dict
from sklearn.metrics import silhouette_score
import os
import matplotlib.pyplot as plt
import numpy as np

def compute_silhouette_scores(expression_df,
                              target_list : list[str],
                              num_tf_cluster_list : list[int],
                              num_target_cluster_list : list[int]):

    non_tf_silhouettes = []
    tf_silhouettes = []

    all_genes = list(expression_df.columns)

    # Compute full distance matrix between all pairs of input genes.
    dist_matrix_all = compute_wasserstein_distance_matrix(expression_df, num_threads=-1)
    #dist_matrix_all.to_csv('testis_distance_matrix.csv', index=True)
    #quit()

    # Separate TF and non-TF distances and cluster both types individually.
    #tf_bool = [True if gene in tf_names else False for gene in dist_matrix_all.columns]
    dist_mat_targets = dist_matrix_all
    #dist_mat_tfs = dist_matrix_all.loc[tf_bool, tf_bool]

    #for num_tf_clusters in num_tf_cluster_list:
    #    tf_to_clust = cluster_genes_to_dict(dist_mat_tfs, num_clusters=num_tf_clusters)
    #    # Prepare input for TF clusters to silhouette score computation.
    #    tf_cluster_labels = [tf_to_clust[gene] for gene in dist_mat_tfs.columns]
    #    dist_mat_tfs_numpy = dist_mat_tfs.copy().to_numpy()
    #    tf_silhouette_score = silhouette_score(X=dist_mat_tfs_numpy, labels=tf_cluster_labels, metric='precomputed')
    #    tf_silhouettes.append(tf_silhouette_score)

    for num_target_clusters in num_target_cluster_list:
        non_tf_to_clust, _ = cluster_genes_to_dict(dist_mat_targets, num_clusters=num_target_clusters, mode="distance")
        # Prepare input for non-TF clusters to silhouette score computation.
        target_cluster_labels = [non_tf_to_clust[gene] for gene in dist_mat_targets.columns]
        dist_mat_non_tfs_numpy = dist_mat_targets.copy().to_numpy()
        non_tf_silhouette_score = silhouette_score(X=dist_mat_non_tfs_numpy, labels=target_cluster_labels, metric='precomputed')
        non_tf_silhouettes.append(non_tf_silhouette_score)

    return tf_silhouettes, non_tf_silhouettes

def compute_diameters(expression_df,
                      tissue_dir,
                      num_target_cluster_list : list[int]):

    # Compute full distance matrix between all pairs of input genes.
    dist_mat_path = os.path.join("/data/bionets/xa39zypy/gtex/", tissue_dir, "distance_matrix.csv")
    if not os.path.exists(dist_mat_path):
        dist_matrix_all = compute_wasserstein_distance_matrix(expression_df, num_threads=-1)
    else:    
        dist_matrix_all = pd.read_csv(dist_mat_path, index_col=0)
    
    dist_matrix_all.index = dist_matrix_all.columns

    dist_mat_targets = dist_matrix_all

    cluster_max_dists = {}  # Dictionary: {k: [max_dists_per_cluster]}.

    for num_target_clusters in num_target_cluster_list:
        non_tf_to_clust, _ = cluster_genes_to_dict(
            dist_mat_targets, num_clusters=num_target_clusters, mode="distance"
        )

        clusters = {}
        for gene, cluster_id in non_tf_to_clust.items():
            clusters.setdefault(cluster_id, []).append(gene)

        # Compute max intra-cluster distance for each cluster
        max_dists_per_cluster = []
        for cluster_id, genes in clusters.items():
            if len(genes) < 2:
                # Skip or mark empty/singleton clusters.
                max_dists_per_cluster.append(0.0)
                continue
            submat = dist_mat_targets.loc[genes, genes]
            max_dist = submat.to_numpy().max()
            max_dists_per_cluster.append(max_dist)

        cluster_max_dists[num_target_clusters] = max_dists_per_cluster

    return cluster_max_dists

def compute_mean_distances(expression_df,
                      tissue_dir,
                      num_target_cluster_list : list[int]):

    # Compute full distance matrix between all pairs of input genes.
    dist_mat_path = os.path.join("/data/bionets/xa39zypy/gtex/", tissue_dir, "distance_matrix.csv")
    if not os.path.exists(dist_mat_path):
        dist_matrix_all = compute_wasserstein_distance_matrix(expression_df, num_threads=-1)
    else:    
        dist_matrix_all = pd.read_csv(dist_mat_path, index_col=0)
    
    dist_matrix_all.index = dist_matrix_all.columns

    dist_mat_targets = dist_matrix_all

    cluster_max_dists = {}  # Dictionary: {k: [max_dists_per_cluster]}.

    for num_target_clusters in num_target_cluster_list:
        non_tf_to_clust, _ = cluster_genes_to_dict(
            dist_mat_targets, num_clusters=num_target_clusters, mode="distance"
        )

        clusters = {}
        for gene, cluster_id in non_tf_to_clust.items():
            clusters.setdefault(cluster_id, []).append(gene)

        # Compute max intra-cluster distance for each cluster
        max_dists_per_cluster = []
        for cluster_id, genes in clusters.items():
            if len(genes) < 2:
                # Skip or mark empty/singleton clusters.
                max_dists_per_cluster.append(0.0)
                continue
            submat = dist_mat_targets.loc[genes, genes]
            # Extract upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(len(genes), k=1)
            mean_dist = submat.to_numpy()[triu_indices].mean()
            max_dists_per_cluster.append(mean_dist)


        cluster_max_dists[num_target_clusters] = max_dists_per_cluster

    return cluster_max_dists

def plot_silhouette_scores(num_clusters, silhouette_scores, x_label, file_name):
    plt.figure(figsize=(8, 5))
    plt.plot(num_clusters, silhouette_scores, marker='o', linestyle='-', color='blue')

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel('Averaged Silhouette Score')
    #plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file_name}.png')

def compute_silhouettes_for_all_tissues(all_tissues_dir, output_dir, subset_tissues : list[str]):

    target_cluster_list = list(range(2,10,1)) + list(range(10, 100, 10))
    tf_cluster_list = []

    results_dict = {'tissue' : [], 'num_clusters': [], 'gene_type': [], 'avg_silhouette': []}
    # Iterate over subdirectories in the parent directory
    for tissue_dir in os.listdir(all_tissues_dir):
        if len(subset_tissues) > 0 and tissue_dir not in subset_tissues:
            continue
        full_dir_path = os.path.join(all_tissues_dir, tissue_dir)
        
        print(f'Procesing tissue {tissue_dir}...')
        if os.path.isdir(full_dir_path):
            tsv_file = os.path.join(full_dir_path, f'{tissue_dir}.tsv')
            exp_mat = pd.read_csv(tsv_file, sep='\t', index_col=0)
            target_file = os.path.join(full_dir_path, f'{tissue_dir}_target_genes.tsv')
            target_df = pd.read_csv(target_file, index_col=0)
            target_list = list(target_df['target_gene'])


            tf_silhouettes, target_silhouettes = compute_silhouette_scores(
                exp_mat,
                target_list,
                tf_cluster_list,
                target_cluster_list
            )

            # Save silhouette scores in dictionary format.
            for tf_index in range(len(tf_cluster_list)):
                num_tf_clusters = tf_cluster_list[tf_index]
                score = tf_silhouettes[tf_index]
                gene_type = 'tf'
                tissue_name = tissue_dir
                results_dict['tissue'].append(tissue_name)
                results_dict['gene_type'].append(gene_type)
                results_dict['num_clusters'].append(num_tf_clusters)
                results_dict['avg_silhouette'].append(score)
            for non_tf_index in range(len(target_cluster_list)):
                num_nontf_clusters = target_cluster_list[non_tf_index]
                score = target_silhouettes[non_tf_index]
                gene_type = 'target'
                tissue_name = tissue_dir
                results_dict['tissue'].append(tissue_name)
                results_dict['gene_type'].append(gene_type)
                results_dict['num_clusters'].append(num_nontf_clusters)
                results_dict['avg_silhouette'].append(score)

    return results_dict

def compute_diameters_for_all_tissues(all_tissues_dir, subset_tissues : list[str]):

    target_cluster_list = list(range(1,10,1)) + list(range(10, 101, 10))
    
    result_dicts = []

    # Iterate over subdirectories in the parent directory
    for tissue_dir in os.listdir(all_tissues_dir):
        if len(subset_tissues) > 0 and tissue_dir not in subset_tissues:
            continue
        full_dir_path = os.path.join(all_tissues_dir, tissue_dir)
        
        print(f'Procesing tissue {tissue_dir}...')
        if os.path.isdir(full_dir_path):
            tsv_file = os.path.join(full_dir_path, f'{tissue_dir}.csv')
            exp_mat = pd.read_csv(tsv_file)

            cluster_max_dists = compute_diameters(
                exp_mat,
                tissue_dir,
                target_cluster_list
            )
            
            for k, max_dists in cluster_max_dists.items():
                for cluster_idx, max_dist in enumerate(max_dists, start=0):
                    result_dicts.append({
                        "tissue" : tissue_dir,
                        "num_clusters": k,
                        "cluster_id": cluster_idx,
                        "mean_intra_dist": max_dist
                    })

    return result_dicts


if __name__ == "__main__":
    #all_tissue_dir = "/data/bionets/xa39zypy/gtex"
    #output_dir = "/data/bionets/xa39zypy/gtex"
    #subset_tissues = []
    
    all_tissue_dir = "/data/bionets/xa39zypy/sc_blood"
    subset_tissues = []

    results_dict = compute_diameters_for_all_tissues(all_tissue_dir,
                                        subset_tissues)
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv('diameters_sc_blood.tsv', sep='\t', index=True)
