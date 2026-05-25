
def summarize_resource_usage():
    """
    For each dataset and number of clusters, compute speedup of approximate P-value computation
    versus ground truth P-value computation. Also compare emissions and memory.
    """

    import os
    import pandas as pd

    save_path = 'results'
    os.makedirs(save_path, exist_ok=True)

    num_clusters = list(range(1, 11)) + list(range(20, 101, 10))
    num_permutations = [1000, 3000, 5000, 7000, 10000]
    sub_populations = ['nk_cells', 'dc', 'cd8+_tcells']

    dfs = []
    for sub_population in sub_populations:
        for num_permut in num_permutations:

            # Load ground truth results
            df_time_mem = pd.read_csv(os.path.join('./results_ground_truth', f'tracking_{sub_population}_num_permut_{num_permut:05d}_00.csv'))
            df_em = pd.read_csv(os.path.join('./results_ground_truth', f'emissions_{sub_population}_num_permut_{num_permut:05d}_00.csv'))

            time_gt = df_time_mem['wall_time'].iloc[0]
            peak_mem_gt = df_time_mem['mem_peak'].iloc[0]
            emissions_gt = df_em['emissions'].iloc[0]

            for l in num_clusters:

                # Load the approximation results
                df_time_mem_approx = pd.read_csv(os.path.join(
                    './results_approx',
                    f'tracking_{sub_population}_num_clust_{l:03d}_num_permut_{num_permut:05d}_grn_id_00.csv')
                )
                df_em_approx = pd.read_csv(os.path.join(
                    './results_approx',
                    f'emissions_{sub_population}_num_clust_{l:03d}_num_permut_{num_permut:05d}_grn_id_00.csv')
                )

                time_approx = df_time_mem_approx['wall_time'].iloc[0]
                peak_mem_approx = df_time_mem_approx['mem_peak'].iloc[0]
                emissions_approx = df_em_approx['emissions'].iloc[0]

                df_time_mem_approx['wall_time_gt'] = time_gt
                df_time_mem_approx['time_diff_abs'] = time_gt - time_approx
                df_time_mem_approx['time_speedup'] = time_gt / time_approx

                df_time_mem_approx['mem_peak_gt'] = peak_mem_gt
                df_time_mem_approx['peak_mem_diff_abs'] = peak_mem_gt - peak_mem_approx
                df_time_mem_approx['peak_mem_ratio'] = peak_mem_gt / peak_mem_approx

                df_time_mem_approx['emissions_gt'] = emissions_gt
                df_time_mem_approx['emissions_diff_abs'] = emissions_gt - emissions_approx
                df_time_mem_approx['emission_ratio'] = emissions_gt / emissions_approx

                df_time_mem_approx['dataset'] = sub_population
                df_time_mem_approx['num_clusters'] = l
                df_time_mem_approx['num_permutations'] = num_permut

                dfs.append(df_time_mem_approx)

    res_df = pd.concat(dfs, axis=0, ignore_index=True)
    res_df.to_csv(os.path.join(save_path, 'time_peak_mem_emissions.csv'), index=False)


def compute_performance_metrics():
    """
    For each dataset and number of clusters, compute performance metrics
    for approximate P-values (raw and BH corrected) against ground truth P-values:
        - Precision, Recall and F1 score at 0.05
        - MAE
    """

    import os
    import warnings
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error

    save_path = './results'
    os.makedirs(save_path, exist_ok=True)

    num_clusters = list(range(1, 11)) + list(range(20, 101, 10))
    num_permutations = [1000, 3000, 5000, 7000, 10000]
    sub_populations = ['nk_cells', 'dc', 'cd8+_tcells']

    alphas = [0.01, 0.05]

    metric_to_score_fct = {
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
    }

    rows = []
    for sub_population in sub_populations:

        for num_permut in num_permutations:

            # Load ground truth results
            grn_gt = pd.read_csv(os.path.join('./results_ground_truth', f'grn_{sub_population}_num_permut_{num_permut:05d}_00.csv'))

            # Subset to relevant columns
            grn_gt = grn_gt.loc[:, ['TF', 'target', 'pvalue', 'pvalue_bh']].copy()

            for l in num_clusters:

                # Load the approximation results
                grn_approx = pd.read_csv(os.path.join(
                    './results_approx',
                    f'grn_{sub_population}_num_clust_{l:03d}_num_permut_{num_permut:05d}_grn_id_00.csv')
                )

                # Subset to relevant columns
                grn_approx = grn_approx.loc[:, ['TF', 'target', 'pvalue', 'pvalue_bh']].copy()

                # Rename
                grn_approx.rename(columns={'pvalue': 'pvalue_approx', 'pvalue_bh': 'pvalue_bh_approx'}, inplace=True)

                # Align with ground truth GRN
                grn_merged = grn_gt.merge(
                    grn_approx,
                    on=['TF', 'target'],
                    how='inner'
                )
                if grn_merged.shape[0] != grn_gt.shape[0] or grn_merged.shape[0] != grn_approx.shape[0]:
                    warnings.warn(
                        'Edges were dropped during merge. There might be an error!'
                    )

                # Compute performance metrics
                for mode in ['raw', 'bh']:

                    if mode == 'raw':
                        pvals = grn_merged['pvalue']
                        pvals_approx = grn_merged['pvalue_approx']
                    else:
                        pvals = grn_merged['pvalue_bh']
                        pvals_approx = grn_merged['pvalue_bh_approx']

                    # Compute MAE
                    mae = mean_absolute_error(pvals, pvals_approx)

                    rows.append({
                        'dataset': sub_population,
                        'mode': mode,
                        'alpha': -1.0,
                        'num_permutations': num_permut,
                        'num_clusters': l,
                        'metric': 'mae',
                        'score': mae,
                        'num_sig': -1,
                        'num_sig_approx': -1,
                    })

                    for alpha in alphas:

                        y_true = (pvals <= alpha).astype(int)
                        y_pred = (pvals_approx <= alpha).astype(int)

                        print(
                            f'# dataset: {sub_population}, num clust: {l}, mode: {mode}, alpha: {alpha}'
                        )
                        # print(y_true.shape[0])
                        # print(y_true.sum())
                        # print(y_pred.sum())

                        for metric, score_fct in metric_to_score_fct.items():

                            score = score_fct(y_true, y_pred, zero_division=0.0)

                            rows.append({
                                'dataset': sub_population,
                                'mode': mode,
                                'alpha': alpha,
                                'num_permutations': num_permut,
                                'num_clusters': l,
                                'metric': metric,
                                'score': score,
                                'num_sig': y_true.sum(),
                                'num_sig_approx': y_pred.sum(),
                            })

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(save_path, 'approximation_quality.csv'), index=False)


def compute_performance_gt_vs_gt():
    """
    For each dataset, compute performance metrics for pairwise ground truth runs and approximate versus ground truth:
        - Precision, Recall and F1 score at 0.05
        - MAE
    """

    import os
    import warnings
    import pandas as pd
    from itertools import combinations
    from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error

    save_path = './results'
    os.makedirs(save_path, exist_ok=True)

    sub_populations = ['nk_cells', 'dc', 'cd8+_tcells']
    num_runs = 10
    num_clusters = 10
    num_permutations = [1000, 3000, 5000, 7000, 10000]
    alphas = [0.01, 0.05]

    metric_to_score_fct = {
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
    }

    rows = []
    for sub_population in sub_populations:
        for num_permut in num_permutations:
            # Load the approximation results
            grn_approx = pd.read_csv(os.path.join(
                './results_approx',
                f'grn_{sub_population}_num_clust_{num_clusters:03d}_num_permut_{num_permut:05d}_grn_id_00.csv')
            )
            grn_approx = grn_approx.loc[:, ['TF', 'target', 'pvalue', 'pvalue_bh']].copy()

            # Load the ground truth runs
            grn_id_to_grn = dict()
            for run_id in range(num_runs):

                grn_gt = pd.read_csv(os.path.join('./results_ground_truth', f'grn_{sub_population}_num_permut_{num_permut:05d}_{run_id:02d}.csv'))
                grn_gt = grn_gt.loc[:, ['TF', 'target', 'pvalue', 'pvalue_bh']].copy()

                grn_id_to_grn[run_id] = grn_gt

            grn_id_to_grn[-1] = grn_approx

            # Compare ground truth to approx and ground truth to ground truth
            for id1, id2 in combinations(sorted(grn_id_to_grn.keys()), 2):

                print(id1, id2)

                grn_approx = grn_id_to_grn[id1].loc[:, ['TF', 'target', 'pvalue', 'pvalue_bh']].copy()
                grn_gt = grn_id_to_grn[id2].loc[:, ['TF', 'target', 'pvalue', 'pvalue_bh']].copy()

                grn_approx.rename(columns={'pvalue': 'pvalue_approx', 'pvalue_bh': 'pvalue_bh_approx'}, inplace=True)

                grn_merged = grn_gt.merge(
                    grn_approx,
                    on=['TF', 'target'],
                    how='inner'
                )
                if grn_merged.shape[0] != grn_gt.shape[0] or grn_merged.shape[0] != grn_approx.shape[0]:
                    warnings.warn(
                        'Edges were dropped during merge. There might be an error!'
                    )

                for mode in ['raw', 'bh']:

                    if mode == 'raw':
                        pvals = grn_merged['pvalue']
                        pvals_approx = grn_merged['pvalue_approx']
                    else:
                        pvals = grn_merged['pvalue_bh']
                        pvals_approx = grn_merged['pvalue_bh_approx']

                    # Compute MAE
                    mae = mean_absolute_error(pvals, pvals_approx)

                    rows.append({
                        'dataset': sub_population,
                        'num_permutations': num_permut,
                        'grn1_id': id1,
                        'grn2_id': id2,
                        'mode': mode,
                        'alpha': -1.0,
                        'num_clusters': num_clusters,
                        'metric': 'mae',
                        'score': mae,
                    })

                    for alpha in alphas:

                        y_true = pvals <= alpha
                        y_pred = pvals_approx <= alpha

                        for metric, score_fct in metric_to_score_fct.items():

                            score = score_fct(y_true, y_pred, zero_division=0.0)

                            rows.append({
                                'dataset': sub_population,
                                'num_permutations': num_permut,
                                'grn1_id': id1,
                                'grn2_id': id2,
                                'mode': mode,
                                'alpha': alpha,
                                'num_clusters': num_clusters,
                                'metric': metric,
                                'score': score,
                            })

        res_df = pd.DataFrame(rows)
        res_df.to_csv('./results/gt_vs_gt.csv', index=False)


def robustness_analysis():
    """
    For 10 different input GRNs (computed with grnboost2) analyze whether robustness increases for FDR controlled GRNs.
    """

    import os
    import pandas as pd
    from itertools import combinations

    save_path = './results'
    os.makedirs(save_path, exist_ok=True)

    sub_populations = ['nk_cells', 'dc', 'cd8+_tcells']
    num_grns = 10
    num_clusters = 10
    num_permutations = [1000, 3000, 5000, 7000, 10000]
    alphas = [0.01, 0.05, 0.1, 0.15, 0.2]

    def jaccard_similarity(g1: pd.DataFrame, g2: pd.DataFrame) -> float:
        edges1 = set(zip(g1['TF'], g1['target']))
        edges2 = set(zip(g2['TF'], g2['target']))
        intersection = len(edges1 & edges2)
        union = len(edges1 | edges2)
        jaccard = intersection / union if union > 0 else 0.0
        return jaccard

    rows = []
    for sub_population in sub_populations:

        for num_permut in num_permutations:

            # Load the GRNs
            grn_id_to_grn = dict()
            for grn_id in range(num_grns):
                grn = pd.read_csv(os.path.join(
                    './results_approx',
                    f'grn_{sub_population}_num_clust_{num_clusters:03d}_num_permut_{num_permut:05d}_grn_id_{grn_id:02d}.csv'
                ))
                grn_id_to_grn[grn_id] = grn

            # Iterate over pairs of GRNs
            for id1, id2 in combinations(sorted(grn_id_to_grn.keys()), 2):

                print(id1, id2)

                grn1 = grn_id_to_grn[id1].loc[:, ['TF', 'target', 'importance', 'pvalue']].copy()
                grn2 = grn_id_to_grn[id2].loc[:, ['TF', 'target', 'importance', 'pvalue']].copy()

                grn1_top50 = (
                    grn1.sort_values('importance', ascending=False)
                    .groupby('TF', group_keys=False)
                    .head(50)
                    .reset_index(drop=True)
                )

                grn2_top50 = (
                    grn2.sort_values('importance', ascending=False)
                    .groupby('TF', group_keys=False)
                    .head(50)
                    .reset_index(drop=True)
                )

                js = jaccard_similarity(grn1_top50, grn2_top50)

                rows.append({
                    'dataset': sub_population,
                    'num_permutations': num_permut,
                    'grn1_id': id1,
                    'grn2_id': id2,
                    'alpha': 'scenic',
                    'jaccard_similarity': js,
                    'grn1_size': grn1_top50.shape[0],
                    'grn2_size': grn2_top50.shape[0],
                })

                for pval_mode in ['pvalue', 'pvalue_bh', 'pvalue_westfall_young']:

                    pvals_1 = grn1[pval_mode].to_numpy().flatten()
                    pvals_2 = grn2[pval_mode].to_numpy().flatten()

                    for alpha in alphas:

                        grn1_sub = grn1[pvals_1 <= alpha].copy()
                        grn2_sub = grn2[pvals_2 <= alpha].copy()

                        if grn1_sub.shape[0] > 0 and grn2_sub.shape[0] > 0:
                            js = jaccard_similarity(grn1_sub, grn2_sub)
                        elif (
                                (grn1_sub.shape[0] > 0 and grn2_sub.shape[0] == 0)
                                or (grn1_sub.shape[0] == 0 and grn2_sub.shape[0] > 0)
                        ):
                            js = 0
                        else:
                            js = 1

                        rows.append({
                            'dataset': sub_population,
                            'num_permutations': num_permut,
                            'grn1_id': id1,
                            'grn2_id': id2,
                            'alpha': str(alpha),
                            'jaccard_similarity': js,
                            'grn1_size': grn1_sub.shape[0],
                            'grn2_size': grn2_sub.shape[0],
                        })

        res_df = pd.DataFrame(rows)
        res_df.to_csv('./results/robustness.csv', index=False)


def robustness_analysis2():
    """
    For 10 different input GRNs (computed with grnboost2) analyze whether robustness increases for FDR controlled GRNs.
    """

    import os
    import pandas as pd
    from itertools import combinations

    save_path = './results'
    os.makedirs(save_path, exist_ok=True)

    sub_populations = ['nk_cells', 'dc', 'cd8+_tcells']
    num_grns = 10
    num_clusters = 10
    num_permutations = [1000, 3000, 5000, 7000, 10000]
    alphas = [0.01, 0.05, 0.1, 0.15, 0.2]

    def jaccard_similarity(g1: pd.DataFrame, g2: pd.DataFrame) -> float:
        edges1 = set(zip(g1['TF'], g1['target']))
        edges2 = set(zip(g2['TF'], g2['target']))
        intersection = len(edges1 & edges2)
        union = len(edges1 | edges2)
        jaccard = intersection / union if union > 0 else 0.0
        return jaccard

    rows = []
    for sub_population in sub_populations:
        for num_permut in num_permutations:

            # Load the GRNs
            grn_id_to_grn = dict()
            for grn_id in range(num_grns):
                grn = pd.read_csv(os.path.join(
                    './results_approx',
                    f'grn_{sub_population}_num_clust_{num_clusters:03d}_num_permut_{num_permut:05d}_grn_id_{grn_id:02d}.csv'
                ))
                grn_id_to_grn[grn_id] = grn

            # Iterate over pairs of GRNs
            for id1, id2 in combinations(sorted(grn_id_to_grn.keys()), 2):

                print(id1, id2)

                grn1 = grn_id_to_grn[id1].loc[:, ['TF', 'target', 'importance', 'pvalue']].copy().sort_values('importance', ascending=False).reset_index(drop=True)
                grn2 = grn_id_to_grn[id2].loc[:, ['TF', 'target', 'importance', 'pvalue']].copy().sort_values('importance', ascending=False).reset_index(drop=True)

                for pval_mode in ['pvalue', 'pvalue_bh', 'pvalue_westfall_young']:

                    pvals_1 = grn1[pval_mode].to_numpy().flatten()
                    pvals_2 = grn2[pval_mode].to_numpy().flatten()

                    for alpha in alphas:
                        grn1_sub = grn1[pvals_1 <= alpha].copy()
                        grn2_sub = grn2[pvals_2 <= alpha].copy()

                        num_selected_edges_1 = grn1_sub.shape[0]
                        num_selected_edges_2 = grn2_sub.shape[0]

                        grn1_sub_importance_base = grn1.iloc[0:num_selected_edges_1, :].copy()
                        grn2_sub_importance_base = grn2.iloc[0:num_selected_edges_2, :].copy()

                        if num_selected_edges_1 > 0 and num_selected_edges_2 > 0:
                            js = jaccard_similarity(grn1_sub, grn2_sub)
                            js_importance_base = jaccard_similarity(grn1_sub_importance_base, grn2_sub_importance_base)
                        elif (
                                (num_selected_edges_1 > 0 and num_selected_edges_2 == 0)
                                or (num_selected_edges_1 == 0 and num_selected_edges_2 > 0)
                        ):
                            js = 0
                            js_importance_base = 0
                        else:
                            js = 1
                            js_importance_base = 1

                        rows.append({
                            'dataset': sub_population,
                            'num_permutations': num_permut,
                            'grn1_id': id1,
                            'grn2_id': id2,
                            'alpha': str(alpha),
                            'pval_mode': pval_mode,
                            'jaccard_similarity_fdr_based': js,
                            'jaccard_similarity_importance_based': js_importance_base,
                            'grn1_size': grn1_sub.shape[0],
                            'grn2_size': grn2_sub.shape[0],
                        })

        res_df = pd.DataFrame(rows)
        res_df.to_csv('./results/robustness2.csv', index=False)


if __name__ == '__main__':

    print('# Summarizing resource usage ...')
    summarize_resource_usage()
    
    print('# Computing performace metrics ...')
    compute_performance_metrics()
    
    print('# Computing performance metrics (gt vs gt) ...')
    compute_performance_gt_vs_gt()
    
    print('# Robusteness 0 ...')
    robustness_analysis()
    
    print('# Robustness 1 ...')
    robustness_analysis2()

    print('done')
