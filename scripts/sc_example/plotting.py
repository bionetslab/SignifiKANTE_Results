
def plot_performance():
    """
    4 Panels:
        - MAE
        - F1 @ alpha
        - GT vs GT
        - Robustness
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.transforms as mtransforms

    alpha = 0.05

    fig, axd = plt.subplot_mosaic(
        '''
        AB
        CD
        ''',
        figsize=(6, 6),
        dpi=300,
        constrained_layout=True,
    )

    # Load results
    res_df_perf = pd.read_csv('./results/approximation_quality.csv')
    res_df_gt_vs_gt = pd.read_csv('./results/gt_vs_gt.csv')
    res_df_robustness = pd.read_csv('./results/robustness.csv')

    # Define cmap
    colors = sns.color_palette('rocket', 3)
    cmap = dict(zip(res_df_perf['dataset'].unique(), colors))

    legend_fontsize = 8

    # --- Plot the MAE across number of target clusters
    plot_df = res_df_perf[
        (res_df_perf['mode'] == 'raw')
        & (res_df_perf['metric'] == 'mae')
    ].copy()
    ax = axd['A']
    sns.lineplot(
        data=plot_df,
        x='num_clusters',
        y='score',
        hue='dataset',
        palette=cmap,
        ax=ax,
    )
    ax.set_xscale('log')
    ax.legend(fontsize=legend_fontsize, loc='upper left', title=None)
    ax.set_ylabel('MAE')
    ax.set_xlabel('Number of target clusters')

    # --- Plot the F1@alpha across number of target clusters
    plot_df = res_df_perf[
        (res_df_perf['mode'] == 'raw')
        & (res_df_perf['metric'] == 'f1')
        & (res_df_perf['alpha'] == alpha)
    ].copy()
    ax = axd['B']
    sns.lineplot(
        data=plot_df,
        x='num_clusters',
        y='score',
        hue='dataset',
        palette=cmap,
        ax=ax,
    )
    ax.set_xscale('log')
    ax.legend(fontsize=legend_fontsize, loc='upper left', title=None)
    ax.set_ylabel(f'F1 score @ {alpha}')
    ax.set_xlabel('Number of target clusters')

    # --- Plot F1@alpha for gt vs gt and approx vs gt
    plot_df = res_df_gt_vs_gt[
        (res_df_gt_vs_gt['mode'] == 'raw')
        & (res_df_gt_vs_gt['metric'] == 'f1')
        & (res_df_gt_vs_gt['alpha'] == alpha)
    ].copy()
    plot_df['mode'] = 'Groundtruth\nvs. groundtruth'
    plot_df.loc[
        (plot_df['grn1_id'] == -1) | (plot_df['grn2_id'] == -1),
        'mode'
    ] = 'Approximation\nvs. groundtruth'
    ax = axd['C']
    sns.boxplot(
        data=plot_df,
        x='dataset',
        y='score',
        hue='mode',
        hue_order=['Groundtruth\nvs. groundtruth', 'Approximation\nvs. groundtruth'],
        palette=dict(zip(
            ['Groundtruth\nvs. groundtruth', 'Approximation\nvs. groundtruth'],
            sns.color_palette('Set2', 2)
        )),
        ax=ax,
    )
    ax.legend(fontsize=legend_fontsize, loc='upper right', title=None)
    ax.set_xlabel('')
    ax.set_ylabel(f'F1 score @ {alpha}')

    ax = axd['D']
    sns.boxplot(
        data=res_df_robustness,
        x='alpha',
        y='jaccard_similarity',
        hue='dataset',
        palette=cmap,
        ax=ax,
    )
    ax.legend(fontsize=legend_fontsize, loc='upper left', title=None)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Pairwise Jaccard similarity')

    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            -0.02,
            1.02,
            label,
            transform=ax.transAxes + trans,
            fontsize=14,
            va='bottom',
            fontfamily='sans-serif',
            fontweight='bold'
        )

    fig.savefig('./results/fig_performance.png', dpi=fig.dpi)


def plot_resources():
    """
    6 Panels:
        - Absolute values in row 1, factors in row 2
        - Columns: Wall time, peak memory, Emissions
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.transforms as mtransforms

    fig, axd = plt.subplot_mosaic(
        '''
        ABC
        DEF
        ''',
        figsize=(6, 6),
        dpi=300,
        constrained_layout=True,
    )

    # Load results

    # Todo ...


    for label, ax in axd.items():

        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            -0.02,
            1.02,
            label,
            transform=ax.transAxes + trans,
            fontsize=14,
            va='bottom',
            fontfamily='sans-serif',
            fontweight='bold'
        )



if __name__ == '__main__':

    # plot_performance()

    print('done')


