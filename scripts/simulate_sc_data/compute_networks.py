# %%
import sys
sys.path.append('/data_nfs/og86asub/SignifiKANTE_Results/')

import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from signifikante.algo import grnboost2_fdr, grnboost2, diy, genie3
import numpy as np
import scanpy as sc
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import numpy as np


import pingouin
import itertools
import random



import os.path as op
import os

np.random.seed(12)
random.seed(12)


def precision_recall(top_n, ground_truth_set):
    true_positives = sum(
            tuple((row['TF'], row['target'])) in ground_truth_set
            for _, row in top_n.iterrows())
    precision = true_positives / top_n.shape[0] if top_n.shape[0] > 0 else 0
    recall = true_positives / len(ground_truth_set) if top_n.shape[0] > 0 else 0
    return precision, recall, true_positives

def compute_precision(ranked_df: pd.DataFrame, thresholds: List[int], ground_truth_set: set, absolute=False) -> pd.DataFrame:

    ranked_df = ranked_df.sort_values(by='importance', ascending=False)
    results = []
    if not absolute:
        absolute_thresholds = [int(t/100 * ranked_df.shape[0]) for t in thresholds]
        print(absolute_thresholds)
        mythresholds = absolute_thresholds
    else:
        mythresholds = thresholds
        mythresholds = [ranked_df.shape[0]]+mythresholds
    
    for t in range(len(mythresholds)):
        subnet = ranked_df.iloc[0:mythresholds[t]]
        precision, recall,true_positives = precision_recall(subnet, ground_truth_set)
        results.append({'Top N Predictions': mythresholds[t], 'thresholds': thresholds[t], 'True Positives': true_positives, 'Precision': precision, 'Recall': recall, 'comparison': 'GRNBoost2'})
        precision, recall,true_positives = precision_recall(subnet[subnet.pvalue<0.05], ground_truth_set)
        results.append({'Top N Predictions': mythresholds[t], 'thresholds': thresholds[t],'True Positives': true_positives, 'Precision': precision, 'Recall': recall, 'comparison': 'SignifiKANTE'})
        precision, recall,true_positives = precision_recall(subnet[subnet.p_adj<0.05], ground_truth_set)
        results.append({'Top N Predictions': mythresholds[t],'thresholds': thresholds[t], 'True Positives': true_positives, 'Precision': precision, 'Recall': recall, 'comparison': 'SignifiKANTE (BH)'})
    return pd.DataFrame(results)


def compute_metrics(orig_net, fdr_net, groundtruth, data_trial = 'data', p_cutoff=0.05):

    ground_truth_set = set()
    for _, row in groundtruth.iterrows():
        ground_truth_set.add(tuple((row['source'], row['target'])))

    results = []

    thresholds: List[int] = [0.5,1,2,3,4, 5,6,7,8,9, 10, 15, 20, 50, 75, 100]
    thresholds = list(np.sort(thresholds))

    precision_all = compute_precision(orig_net, thresholds, ground_truth_set)
    precision_all['dataset'] = data_trial
    return precision_all






def preprocess_data(gex_data):
    gex_data = gex_data.T
    var=pd.DataFrame(gex_data.columns.T)
    var = var.set_index('gene')
    obs = pd.DataFrame(gex_data.index)
    ad = sc.AnnData(gex_data.values, var=var, obs = obs)
    sc.pp.scale(ad)
    new_d = pd.DataFrame(ad.X, columns=gex_data.columns)
    return new_d


def compute_nets(new_d, tfs, num_target_clusters=10, iterations=1):

        nets = []
        for k in range(iterations):
                orig_net = grnboost2(expression_data=prepr_data, tf_names=list(groundtruth.source.unique()))
                nets.append(orig_net)
        
        nets1 = pd.concat(nets)
        orig_net = nets1.groupby(['TF', 'target']).mean('importance').reset_index()
        #nets1['zscore'] = nets1[('importance','mean')]/nets1[('importance','std')]
        #compute input GRN
        #orig_net = grnboost2(expression_data=new_d, tf_names=tfs)
        # Run approximate FDR control.
        #orig_net = orig_net.loc[:, ['TF', 'target']]
        print(orig_net.columns)
        fdr_net = grnboost2_fdr(
                tf_names= tfs,
                expression_data=new_d,
                cluster_representative_mode="random",
                num_target_clusters=num_target_clusters,
                num_tf_clusters=-1,
                input_grn=orig_net,
                num_permutations= 1000,
                scale_for_tf_sampling=True)


        fdr_net['p_adj'] = fdrcorrection(fdr_net['pvalue'])[1]

        return orig_net, fdr_net
                

                

def jaccard(s1, s2):
    try:
        jac = len(s1.intersection(s2))/len(s1.union(s2))
    except:
        jac = 0
    return(jac)

def compute_jaccards(collect_networks, thresholds =  [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90, 100], absolute=False):
    jaccards = []
    thresholds = list(np.sort(thresholds))
    net_counter = 0
    for net in collect_networks:
        if not absolute:
            absolute_thresholds = [int(t/100 * net.shape[0]) for t in thresholds]
            print(absolute_thresholds)
            mythresholds = absolute_thresholds
        else:
            mythresholds = thresholds
            mythresholds = [net.shape[0]]+mythresholds

        net['edge_keys']  = net['TF']+'_'+net['target']
        jaccard_local = []
        for t in range(len(mythresholds)):
            print(t)
            subnet = net.iloc[0:mythresholds[t]].sort_values('importance')
            
            s1 = set(subnet[subnet['p_adj']<0.05]['edge_keys'])
            s2 = set(subnet['edge_keys'])
            s3 = set(subnet[subnet['pvalue']<0.05]['edge_keys'])
            j = jaccard(s1, s2)
            j2 = jaccard(s3,s2)
            jaccard_local.append([mythresholds[t],thresholds[t], j, net_counter, 'SignifiKANTE (BH-adj)'])
            jaccard_local.append([mythresholds[t],thresholds[t], j2, net_counter, 'SignifiKANTE'])

        net_counter+=1
        jaccards.append(jaccard_local)
    jaccards = np.concatenate(jaccards)
    return pd.DataFrame(jaccards)


if __name__ == '__main__':

    net_collect = {}
    p_net_collect = {}
    metrics_collector =  []
    groundtruth_collector = {}
    for s in ['5_sources', '10_sources', '20_sources']:
    #for s in ['5_sources']:
        os.makedirs(f'/data/bionets/og86asub/SignifiKANTE_Results/results/sc_simulated_data/{s}', exist_ok=True)
        net_collect[s] = []
        p_net_collect[s] = []
        groundtruth_collector[s] = []
        for i in range(1, 11):
            # Load expression matrix - in this case simulate one.
            gex_data = pd.read_csv(f'/data/bionets/og86asub/SignifiKANTE_Results/data/sc_simulated_data/{s}/data/data_{i}.tsv', index_col=0, sep='\t')
            groundtruth = pd.read_csv(f'/data/bionets/og86asub/SignifiKANTE_Results/data/sc_simulated_data/{s}/nets/network_{i}.tsv',  sep='\t')
            groundtruth_collector[s].append(groundtruth)
            prepr_data = preprocess_data(gex_data)
            tfs = list(groundtruth.source.unique())
            orig_net, fdr_net = compute_nets(prepr_data, tfs)
            fdr_net.to_csv(f'/data/bionets/og86asub/SignifiKANTE_Results/results/sc_simulated_data/{s}/grn_{i}.tsv', sep='\t')
            net_collect[s].append(orig_net)
            p_net_collect[s].append(fdr_net)
            metrics = compute_metrics(orig_net, fdr_net, groundtruth, data_trial = f'data_{i}')
            metrics['data_configuration'] = s
            metrics_collector.append(metrics)


    metrics_collector =  []
    for s in ['5_sources', '10_sources', '20_sources']:
        print(s)
        for i in range(0, 10):
            metrics = compute_metrics(net_collect[s][i], p_net_collect[s][i], groundtruth_collector[s][i], data_trial = f'data_{i}', p_cutoff=0.05)
            metrics['data_configuration'] = s
            metrics_collector.append(metrics)

    metrics_collector = pd.concat(metrics_collector)
    metrics_collector.to_csv(f'/data/bionets/og86asub/SignifiKANTE_Results/results/sc_simulated_data/aggregated_metrics.tsv', sep = '\t')


    myjac_collector = []
    for s in ['5_sources', '10_sources', '20_sources']:
        myjac = compute_jaccards(collect_networks=p_net_collect[s])
        myjac.columns = ['top_n','thresholds', 'jaccard_index', 'net_counter', 'type']
        myjac['jaccard_index']  = pd.to_numeric(myjac['jaccard_index'] )
        myjac['top_n']  = pd.to_numeric(myjac['top_n'] )
        myjac['thresholds']  = pd.to_numeric(myjac['thresholds'] )

        myjac.to_csv(f'/data/bionets/og86asub/SignifiKANTE_Results/results/sc_simulated_data/{s}/jaccard_indices.tsv')
        myjac_collector.append(myjac)

    myjac = pd.concat(myjac_collector)

    myjac.to_csv(f'/data/bionets/og86asub/SignifiKANTE_Results/results/sc_simulated_data/jaccard_indices.tsv')
# %%
