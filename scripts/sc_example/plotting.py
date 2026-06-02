
# def plot_performance():
#     """
#     4 Panels:
#         - MAE
#         - F1 @ alpha
#         - GT vs GT
#         - Robustness
#     """
#     import os
#     import matplotlib
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import matplotlib.transforms as mtransforms
#     from matplotlib.ticker import FuncFormatter
#     from matplotlib.patches import Patch
#     from matplotlib.lines import Line2D
#
#     matplotlib.use('Agg')
#
#     alpha = 0.05
#
#     window_size = 5
#
#     save_path = './plots'
#     os.makedirs(save_path, exist_ok=True)
#
#     fig, axd = plt.subplot_mosaic(
#         '''
#         ABC
#         LL.
#         DEF
#         ''',
#         figsize=(8, 5.5),
#         dpi=600,
#         constrained_layout=True,
#         gridspec_kw={
#             'height_ratios': [1, 0.05, 1]
#         }
#     )
#
#     # Load results
#     res_df_perf = pd.read_csv('./results/approximation_quality.csv')
#     res_df_gt_vs_gt = pd.read_csv('./results/gt_vs_gt.csv')
#     res_df_num_sig = pd.read_csv('./results/num_sig_edges.csv')
#
#     # Rename cell types
#     mapper = {'nk_cells': 'NK cell', 'dc': 'Dendritic cell', 'cd8+_tcells': 'T cell (CD8+)'}
#     res_df_perf['dataset'] = res_df_perf['dataset'].map(mapper)
#     res_df_gt_vs_gt['dataset'] = res_df_gt_vs_gt['dataset'].map(mapper)
#
#     # Define cmap
#     colors = sns.color_palette('rocket', 3)
#     cmap = dict(zip(sorted(res_df_perf['dataset'].unique()), colors))
#
#     legend_fontsize = 8
#
#     # --- Plot the MAE across number of target clusters
#     plot_df = res_df_perf[
#         (res_df_perf['num_permutations'] == 1000)
#         & (res_df_perf['mode'] == 'raw')
#         & (res_df_perf['metric'] == 'mae')
#     ].copy()
#
#     # plot_df = plot_df.sort_values(['dataset', 'num_clusters'])
#     # plot_df['smoothed_score'] = (
#     #     plot_df.groupby('dataset')['score']
#     #     .transform(
#     #         lambda s: s.rolling(
#     #             window=window_size,
#     #             center=True,
#     #             min_periods=1
#     #         ).mean()
#     #     )
#     # )
#
#     ax = axd['A']
#     sns.lineplot(
#         data=plot_df,
#         x='num_clusters',
#         y='score',
#         hue='dataset',
#         palette=cmap,
#         hue_order=sorted(cmap.keys()),
#         ax=ax,
#     )
#     ax.set_xscale('log')
#     ax.legend(fontsize=legend_fontsize, loc='upper left', title=None)
#     ax.set_ylabel('MAE')
#     ax.set_xlabel('Number of target clusters')
#
#     # --- Plot the F1@alpha across number of target clusters
#     plot_df = res_df_perf[
#         (res_df_perf['num_permutations'] == 1000)
#         & (res_df_perf['mode'] == 'raw')
#         & (res_df_perf['metric'] == 'f1')
#         & (res_df_perf['alpha'] == alpha)
#     ].copy()
#
#     plot_df = plot_df.sort_values(['dataset', 'num_clusters'])
#     plot_df['smoothed_score'] = (
#         plot_df.groupby('dataset')['score']
#         .transform(
#             lambda s: s.rolling(
#                 window=window_size,
#                 center=True,
#                 min_periods=1
#             ).mean()
#         )
#     )
#
#     ax = axd['B']
#     sns.lineplot(
#         data=plot_df,
#         x='num_clusters',
#         y='score',
#         hue='dataset',
#         palette=cmap,
#         hue_order=sorted(cmap.keys()),
#         ax=ax,
#     )
#     ax.set_xscale('log')
#     ax.legend(fontsize=legend_fontsize, loc='upper left', title=None)
#     ax.set_ylabel(f'F1 score @ {alpha}')
#     ax.set_xlabel('Number of target clusters')
#
#     # --- Plot F1@alpha for gt vs gt and approx vs gt
#     plot_df = res_df_gt_vs_gt[
#         (res_df_gt_vs_gt['num_permutations'] == 1000)
#         & (res_df_gt_vs_gt['mode'] == 'raw')
#         & (res_df_gt_vs_gt['metric'] == 'f1')
#         & (res_df_gt_vs_gt['alpha'] == alpha)
#     ].copy()
#     plot_df['comparison'] = 'Groundtruth\nvs. groundtruth'
#     plot_df.loc[
#         (plot_df['grn1_id'] == -1) | (plot_df['grn2_id'] == -1),
#         'comparison'
#     ] = 'Approximation\nvs. groundtruth'
#     ax = axd['C']
#     sns.boxplot(
#         data=plot_df,
#         x='dataset',
#         y='score',
#         hue='comparison',
#         hue_order=['Groundtruth\nvs. groundtruth', 'Approximation\nvs. groundtruth'],
#         palette=dict(zip(
#             ['Groundtruth\nvs. groundtruth', 'Approximation\nvs. groundtruth'],
#             sns.color_palette('Set2', 2)
#         )),
#         ax=ax,
#     )
#     ax.legend(fontsize=legend_fontsize, loc='lower center', title=None)
#     ax.set_xlabel('')
#     ax.set_ylabel(f'F1 score @ {alpha}')
#     ax.tick_params(axis='x', labelrotation=25)
#     for label in ax.get_xticklabels():
#         label.set_ha('right')
#         label.set_va('top')
#
#     # --- Plot MAE across number of permutations for fixed number of clusters
#     plot_df = res_df_perf[
#         (res_df_perf['num_clusters'] == 10)
#         & (res_df_perf['mode'] == 'raw')
#         & (res_df_perf['metric'] == 'mae')
#     ].copy()
#
#     ax = axd['D']
#     sns.lineplot(
#         data=plot_df,
#         x='num_permutations',
#         y='score',
#         hue='dataset',
#         palette=cmap,
#         hue_order=sorted(cmap.keys()),
#         marker='o',
#         ax=ax,
#     )
#     ax.set_xticks([1000, 3000, 5000, 7000, 10000])
#     ax.xaxis.set_major_formatter(
#         FuncFormatter(lambda x, _: rf'${x / 1000:g}$')
#     )
#     ax.legend(fontsize=legend_fontsize, loc='upper left', title=None)
#     ax.set_ylabel('MAE')
#     ax.set_xlabel(r'Permutations $\times 10^3$')
#
#     # --- Plot F1 across number of permutations for fixed number of clusters
#     plot_df = res_df_perf[
#         (res_df_perf['num_clusters'] == 10)
#         & (res_df_perf['mode'] == 'raw')
#         & (res_df_perf['metric'] == 'f1')
#         & (res_df_perf['alpha'] == alpha)
#     ].copy()
#
#     ax = axd['E']
#     sns.lineplot(
#         data=plot_df,
#         x='num_permutations',
#         y='score',
#         hue='dataset',
#         palette=cmap,
#         hue_order=sorted(cmap.keys()),
#         marker='o',
#         ax=ax,
#     )
#     ax.set_xticks([1000, 3000, 5000, 7000, 10000])
#     ax.xaxis.set_major_formatter(
#         FuncFormatter(lambda x, _: rf'${x / 1000:g}$')
#     )
#     ax.legend(fontsize=legend_fontsize, loc='upper left', title=None)
#     ax.set_ylabel('F1 score @ 0.05')
#     ax.set_xlabel(r'Permutations $\times 10^3$')
#
#     # --- Plot number of significant edges depending on number of permutations
#     plot_df = res_df_num_sig[
#         (res_df_num_sig['alpha'] == alpha)
#         & (res_df_num_sig['num_clusters'].isin([-1, 10]))
#         & (res_df_num_sig['pval_mode'].isin(['pvalue_bh', 'pvalue_westfall_young']))
#     ].copy()
#     plot_df['pval_mode'] = plot_df['pval_mode'].map({'pvalue': 'raw', 'pvalue_bh': 'BH', 'pvalue_westfall_young': 'WY'})
#
#     plot_df['mode_comb'] = 'GT BH'
#     plot_df.loc[
#         (plot_df['mode'] == 'approx') & (plot_df['pval_mode'] == 'BH'),
#         'mode_comb'
#     ] = 'Approx\nBH'
#     plot_df.loc[
#         (plot_df['mode'] == 'approx') & (plot_df['pval_mode'] == 'WY'),
#         'mode_comb'
#     ] = 'Approx\nWY'
#
#     ax = axd['F']
#     sns.lineplot(
#         data=plot_df,
#         x='num_permutations',
#         y='num_sig',
#         hue='mode_comb',
#         palette='Set2',
#         # style='dataset',
#         errorbar=None,
#         marker='o',
#         ax=ax,
#     )
#     ax.set_xticks([1000, 3000, 5000, 7000, 10000])
#     ax.xaxis.set_major_formatter(
#         FuncFormatter(lambda x, _: rf'${x / 1000:g}$')
#     )
#     handles, labels = ax.get_legend_handles_labels()
#     dot_handles = [
#         Line2D(
#             [0], [0],
#             marker='o',
#             linestyle='None',
#             markersize=3,
#             color=h.get_color(),
#             label=lab,
#         )
#         for h, lab in zip(handles, labels)
#     ]
#     ax.legend(
#         handles=dot_handles,
#         labels=labels,
#         fontsize=legend_fontsize,
#         loc='center left',
#         title=None,
#     )
#     # ax.legend(fontsize=legend_fontsize, loc='center left', title=None)
#     ax.set_ylabel(f'Number of edges @ {alpha}')
#     ax.set_xlabel(r'Permutations $\times 10^3$')
#
#
#     # Create rectangular legend patches
#     handles = [
#         Patch(facecolor=cmap[name], edgecolor='lightgray', linewidth=0.8, label=name)
#         for name in sorted(cmap.keys())
#     ]
#     # Plot legend
#     ax = axd['L']
#     ax.axis('off')
#     ax.legend(
#         handles=handles,
#         loc='center',
#         frameon=False,
#         fontsize=legend_fontsize,
#         handlelength=2.0,
#         handleheight=1.5,
#         ncol=3,
#     )
#     for label, ax in axd.items():
#         if label in list('LCF'):
#             continue
#         ax.legend_.remove()
#
#     for label, ax in axd.items():
#         if label == 'L':
#             continue
#         trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
#         ax.text(
#             -0.02,
#             1.02,
#             label,
#             transform=ax.transAxes + trans,
#             fontsize=14,
#             va='bottom',
#             fontfamily='sans-serif',
#             fontweight='bold'
#         )
#
#     fig.savefig('./plots/fig_performance.pdf', dpi=fig.dpi)
#
#
# def plot_resources():
#     """
#     6 Panels:
#         - Absolute values in row 1, factors in row 2
#         - Columns: Wall time, peak memory, Emissions
#     """
#     import os
#     import matplotlib
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import matplotlib.transforms as mtransforms
#
#     matplotlib.use('Agg')
#
#     save_path = './plots'
#     os.makedirs(save_path, exist_ok=True)
#
#     # Load results
#     res_df = pd.read_csv('./results/time_peak_mem_emissions.csv')
#
#     mapper = {'nk_cells': 'NK', 'dc': 'DC', 'cd8+_tcells': 'T cell'}
#     res_df['dataset'] = res_df['dataset'].map(mapper)
#
#     res_df['wall_time'] = res_df['wall_time'] / 60
#     res_df['mem_peak'] = res_df['mem_peak'] / 1024
#     res_df['emissions'] = (res_df['emissions_gt'] + res_df['emissions_diff_abs']) * 1000
#
#     col_to_ylabel = {
#         'wall_time': 'Wall time [minutes]',
#         'mem_peak': 'Peak memory [GiB]',
#         'emissions': r'Emissions [$\text{kg CO}_2$ eq]',
#         'time_speedup': 'Speedup factor',
#         'peak_mem_ratio': 'Peak memory ratio\n[GT / Approx.]',
#         'emission_ratio': 'Emissions ratio\n[GT / Approx.]'
#     }
#     res_df.rename(columns=col_to_ylabel, inplace=True)
#
#     col_to_fmt = {
#         'wall_time': '.1f', 'mem_peak': '.1f', 'emissions': '.1f',
#         'time_speedup': '.1f', 'peak_mem_ratio': '.1f', 'emission_ratio': '.1f'
#     }
#
#     label_fontsize = 15
#
#     fig, axd = plt.subplot_mosaic(
#         '''
#         ABC
#         DEF
#         ''',
#         figsize=(14, 7),
#         dpi=300,
#         constrained_layout=True,
#     )
#
#     for ax, (key, col) in zip(axd.values(), col_to_ylabel.items()):
#
#         plot_df = (
#             res_df[res_df['num_permutations'] == 1000]
#             .pivot(
#                 index='dataset',
#                 columns='num_clusters',
#                 values=col
#             )
#         )
#
#         heat_map = sns.heatmap(
#             plot_df,
#             cmap='mako',
#             linewidths=0.5,
#             linecolor='white',
#             annot=True,
#             fmt=col_to_fmt[key],
#             cbar_kws={'label': col},
#             ax=ax,
#             annot_kws={
#                 'size': label_fontsize - 2,
#                 'rotation': 90,  # 🔹 rotate text inside cells (degrees)
#                 'ha': 'center',  # horizontal alignment (e.g., 'center', 'left', 'right')
#                 'va': 'center'  # vertical alignment (e.g., 'center', 'top', 'bottom')
#             }
#         )
#         ax.set_xlabel('Number of target clusters', size=label_fontsize)
#         ax.set_ylabel(None)
#         ax.tick_params(axis='y', labelsize=label_fontsize)
#
#         cbar = heat_map.collections[0].colorbar
#         # cbar.set_ticks([100, 150, 200, 250])
#         cbar.ax.yaxis.label.set_size(label_fontsize)
#         cbar.ax.tick_params(labelsize=label_fontsize - 4)
#
#     for label, ax in axd.items():
#
#         trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
#         ax.text(
#             -0.02,
#             1.02,
#             label,
#             transform=ax.transAxes + trans,
#             fontsize=18,
#             va='bottom',
#             fontfamily='sans-serif',
#             fontweight='bold'
#         )
#
#     fig.savefig('./plots/fig_resources.pdf', dpi=fig.dpi)


def plot_sc_results():

    import os
    import matplotlib
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Patch

    matplotlib.use('Agg')

    save_path = './plots'
    os.makedirs(save_path, exist_ok=True)

    # Load results
    res_df_perf = pd.read_csv('./results/approximation_quality.csv')
    res_df_gt_vs_gt = pd.read_csv('./results/gt_vs_gt.csv')
    res_df_resources = pd.read_csv('./results/time_peak_mem_emissions.csv')

    # Rename cell types
    mapper_perf = {'nk_cells': 'NK cell', 'dc': 'Dendritic cell', 'cd8+_tcells': 'T cell (CD8+)'}
    res_df_perf['dataset'] = res_df_perf['dataset'].map(mapper_perf)
    res_df_gt_vs_gt['dataset'] = res_df_gt_vs_gt['dataset'].map(mapper_perf)

    mapper_resources = {'nk_cells': 'NK', 'dc': 'D', 'cd8+_tcells': 'T'}
    res_df_resources['dataset'] = res_df_resources['dataset'].map(mapper_resources)

    # Convert units (s -> min, MiB -> GiB)
    res_df_resources['wall_time'] = res_df_resources['wall_time'] / 60
    res_df_resources['mem_peak'] = res_df_resources['mem_peak'] / 1024

    col_to_ylabel = {
        'wall_time': 'Wall time [minutes]',
        'mem_peak': 'Peak memory [GiB]',
        'time_speedup': 'Speedup factor',
        'peak_mem_ratio': 'Peak memory ratio\n[GT / Approx.]',
    }
    res_df_resources.rename(columns=col_to_ylabel, inplace=True)

    col_to_fmt = {'wall_time': '.1f', 'mem_peak': '.1f', 'time_speedup': '.1f', 'peak_mem_ratio': '.1f'}

    # Define colormaps
    cmap0 = dict(zip(sorted(res_df_perf['dataset'].unique()), sns.color_palette('rocket', 3)))
    cmap1 = dict(zip(
            ['Approximation\nvs. groundtruth', 'Groundtruth\nvs. groundtruth'],
            sns.color_palette('Set2', 2)
    ))

    panel_label_fs = 18
    ax_label_fs = 12
    legend_fs = 12
    heatmap_annot_fs = 12
    heatmap_y_label_fs = 12
    cbar_tick_label_fs = 12
    cbar_ax_label_fs = 12

    # Initialize mosaic
    mosaic = '''
                AB
                CX
                CY
                DE
                FG
                '''

    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(10, 11),
        dpi=600,
        constrained_layout=True,
        gridspec_kw={
            'height_ratios': [1.0, 0.5, 0.5, 1.0, 1.0]
        }
    )

    # --- Plot MAE across number of target clusters
    plot_df = res_df_perf[
        (res_df_perf['num_permutations'] == 1000)
        & (res_df_perf['mode'] == 'raw')
        & (res_df_perf['metric'] == 'mae')
    ].copy()
    ax = axd['A']
    sns.lineplot(
        data=plot_df,
        x='num_clusters',
        y='score',
        hue='dataset',
        palette=cmap0,
        hue_order=sorted(cmap0.keys()),
        ax=ax,
    )
    ax.set_xscale('log')
    ax.set_ylabel('MAE', fontsize=ax_label_fs)
    ax.set_xlabel('Number of target clusters', fontsize=ax_label_fs)
    ax.grid(True)

    # --- Plot F1@alpha across number of target clusters
    alpha = 0.05
    plot_df = res_df_perf[
        (res_df_perf['num_permutations'] == 1000)
        & (res_df_perf['mode'] == 'raw')
        & (res_df_perf['metric'] == 'f1')
        & (res_df_perf['alpha'] == alpha)
        ].copy()
    ax = axd['B']
    sns.lineplot(
        data=plot_df,
        x='num_clusters',
        y='score',
        hue='dataset',
        palette=cmap0,
        hue_order=sorted(cmap0.keys()),
        ax=ax,
    )
    ax.set_xscale('log')
    ax.set_ylabel(f'F1 score @ {alpha}', fontsize=ax_label_fs)
    ax.set_xlabel('Number of target clusters', fontsize=ax_label_fs)
    ax.grid(True)

    # --- Plot F1@alpha for gt vs gt and approx vs gt
    plot_df = res_df_gt_vs_gt[
        (res_df_gt_vs_gt['num_permutations'] == 1000)
        & (res_df_gt_vs_gt['num_clusters'] == 100)
        & (res_df_gt_vs_gt['mode'] == 'raw')
        & (res_df_gt_vs_gt['metric'] == 'f1')
        & (res_df_gt_vs_gt['alpha'] == alpha)
        ].copy()
    plot_df['comparison'] = 'Groundtruth\nvs. groundtruth'
    plot_df.loc[
        (plot_df['grn1_id'] == -1) | (plot_df['grn2_id'] == -1),
        'comparison'
    ] = 'Approximation\nvs. groundtruth'
    ax = axd['C']
    sns.boxplot(
        data=plot_df,
        x='dataset',
        y='score',
        hue='comparison',
        hue_order=['Approximation\nvs. groundtruth', 'Groundtruth\nvs. groundtruth'],
        palette=cmap1,
        ax=ax,
    )
    ax.set_xlabel('')
    ax.set_ylabel(f'F1 score @ {alpha}', fontsize=ax_label_fs)
    ax.tick_params(axis='x', labelrotation=15, labelsize=ax_label_fs)
    for label in ax.get_xticklabels():
        label.set_ha('right')
        label.set_va('top')

    # Plot legend A, B
    handles = [
        Patch(facecolor=cmap0[name], edgecolor='gray', linewidth=1.0, label=name)
        for name in sorted(cmap0.keys())
    ]
    ax = axd['X']
    ax.axis('off')
    ax.legend(
        handles=handles,
        loc='center',
        frameon=True,
        fontsize=legend_fs,
        handlelength=2.0,
        handleheight=1.5,
        ncol=1,
    )

    # Plot legend C
    handles = [
        Patch(facecolor=cmap1[name], edgecolor='gray', linewidth=1.0, label=name)
        for name in sorted(cmap1.keys())
    ]
    ax = axd['Y']
    ax.axis('off')
    ax.legend(
        handles=handles,
        loc='center',
        frameon=True,
        fontsize=legend_fs,
        handlelength=2.0,
        handleheight=1.5,
        ncol=1,
    )

    for key, ax in axd.items():
        if key in list('ABC'):
            ax.legend_.remove()

    # --- Plot resource usage
    for ax_key, (key, col) in zip(list('DEFG'), col_to_ylabel.items()):

        ax = axd[ax_key]

        plot_df = (
            res_df_resources[res_df_resources['num_permutations'] == 1000]
            .pivot(
                index='dataset',
                columns='num_clusters',
                values=col
            )
        )

        heat_map = sns.heatmap(
            plot_df,
            cmap='mako',
            linewidths=0.5,
            linecolor='white',
            annot=True,
            fmt=col_to_fmt[key],
            cbar_kws={'label': col},
            ax=ax,
            annot_kws={
                'size': heatmap_annot_fs,
                'rotation': 90,
                'ha': 'center',
                'va': 'center',
            }
        )
        ax.set_xlabel('Number of target clusters', fontsize=ax_label_fs)
        ax.set_ylabel(None)
        ax.tick_params(axis='y', labelsize=heatmap_y_label_fs)

        cbar = heat_map.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(cbar_ax_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_label_fs)

    for label, ax in axd.items():
        if label in list('XY'):
            continue
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            -0.02,
            1.02,
            label,
            transform=ax.transAxes + trans,
            fontsize=panel_label_fs,
            va='bottom',
            fontfamily='sans-serif',
            fontweight='bold'
        )

    fig.savefig('./plots/fig_sc_results.pdf', dpi=fig.dpi)


def plot_num_permutations():

    import os
    import matplotlib
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.transforms as mtransforms
    from matplotlib.ticker import FuncFormatter
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    matplotlib.use('Agg')

    save_path = './plots'
    os.makedirs(save_path, exist_ok=True)

    fig, axd = plt.subplot_mosaic(
        '''
        ABC
        XXY
        ''',
        figsize=(6, 2.3),
        dpi=600,
        constrained_layout=True,
        gridspec_kw={
            'height_ratios': [1, 0.05]
        }
    )

    # Load results
    res_df_perf = pd.read_csv('./results/approximation_quality.csv')
    res_df_num_sig = pd.read_csv('./results/num_sig_edges.csv')

    # Rename cell types
    mapper = {'nk_cells': 'NK cell', 'dc': 'Dendritic cell', 'cd8+_tcells': 'T cell (CD8+)'}
    res_df_perf['dataset'] = res_df_perf['dataset'].map(mapper)
    res_df_num_sig['dataset'] = res_df_num_sig['dataset'].map(mapper)

    # Define cmap
    colors = sns.color_palette('rocket', 3)
    cmap = dict(zip(sorted(res_df_perf['dataset'].unique()), colors))

    legend_fontsize = 8

    # --- Plot MAE across number of permutations for fixed number of clusters
    plot_df = res_df_perf[
        # (res_df_perf['num_clusters'] == 100)
        (res_df_perf['mode'] == 'raw')
        & (res_df_perf['metric'] == 'mae')
        ].copy()
    ax = axd['A']
    sns.lineplot(
        data=plot_df,
        x='num_permutations',
        y='score',
        hue='dataset',
        palette=cmap,
        hue_order=reversed(sorted(cmap.keys())),
        marker='o',
        ax=ax,
    )
    ax.set_xticks([1000, 3000, 5000, 7000, 10000])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: rf'${x / 1000:g}$')
    )
    ax.set_ylabel('MAE')
    ax.set_xlabel(r'Permutations $\times 10^3$')

    # --- Plot F1 across number of permutations for fixed number of clusters
    alpha = 0.05
    plot_df = res_df_perf[
        # (res_df_perf['num_clusters'] == 10)
        (res_df_perf['mode'] == 'raw')
        & (res_df_perf['metric'] == 'f1')
        & (res_df_perf['alpha'] == alpha)
    ].copy()
    ax = axd['B']
    sns.lineplot(
        data=plot_df,
        x='num_permutations',
        y='score',
        hue='dataset',
        palette=cmap,
        hue_order=reversed(sorted(cmap.keys())),
        marker='o',
        ax=ax,
    )
    ax.set_xticks([1000, 3000, 5000, 7000, 10000])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: rf'${x / 1000:g}$')
    )
    ax.set_ylabel(f'F1 score @ {alpha}')
    ax.set_xlabel(r'Permutations $\times 10^3$')

    # --- Plot number of significant edges depending on number of permutations
    plot_df = res_df_num_sig[
        (res_df_num_sig['alpha'] == alpha)
        # & (res_df_num_sig['num_clusters'].isin([100]))
        & (res_df_num_sig['num_clusters'] != -1)  # Exclude -1 = GT
        & (res_df_num_sig['pval_mode'].isin(['pvalue_bh', 'pvalue_westfall_young']))
        ].copy()

    plot_df['pval_mode'] = plot_df['pval_mode'].map({'pvalue': 'raw', 'pvalue_bh': 'BH', 'pvalue_westfall_young': 'WY'})

    ax = axd['C']
    # sns.lineplot(
    #     data=plot_df,
    #     x='num_permutations',
    #     y='num_sig',
    #     hue='pval_mode',
    #     palette='Set2',
    #     # style='dataset',
    #     # errorbar=None,
    #     marker='o',
    #     ax=ax,
    # )
    sns.lineplot(
        data=plot_df,
        x='num_permutations',
        y='num_sig',
        hue='dataset',
        hue_order=reversed(sorted(cmap.keys())),
        palette=cmap,
        style='pval_mode',
        markers={
            'BH': 'X',
            'WY': '^',
        },
        dashes={
            'BH': '',
            'WY': (1.5, 1.0),
        },
        ax=ax,
    )
    ax.set_xticks([1000, 3000, 5000, 7000, 10000])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: rf'${x / 1000:g}$')
    )
    ax.set_ylabel(f'Number of edges @ {alpha}')
    ax.set_xlabel(r'Permutations $\times 10^3$')

    # Plot legend A, B
    handles = [
        Patch(facecolor=cmap[name], edgecolor='lightgray', linewidth=0.8, label=name)
        for name in sorted(cmap.keys())
    ]
    ax = axd['X']
    ax.legend(
        handles=handles,
        loc='center',
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.0,
        handleheight=1.5,
        ncol=3,
    )
    ax.axis('off')

    # Plot legend C
    style_handles = [
        Line2D(
            [0], [0],
            color='black',
            linestyle='-',
            marker='X',
            label='BH',
        ),
        Line2D(
            [0], [0],
            color='black',
            linestyle=(0, (1.5, 1.0)),
            marker='^',
            label='WY',
        ),
    ]
    ax = axd['Y']
    ax.legend(
        handles=style_handles,
        fontsize=8,
        loc='center',
        ncol=2,
        frameon=False,
    )
    ax.axis('off')

    for label, ax in axd.items():
        if label in list('ABC'):
            ax.legend_.remove()

    for label, ax in axd.items():
        if label in list('XY'):
            continue
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            -0.02,
            1.01,
            label,
            transform=ax.transAxes + trans,
            fontsize=14,
            va='bottom',
            fontfamily='sans-serif',
            fontweight='bold'
        )

    fig.savefig('./plots/fig_sc_num_permutations.pdf', dpi=fig.dpi)


def retrieve_table_data():

    import os
    import pandas as pd

    for dataset in ['dc.csv', 'cd8+_tcells.csv', 'nk_cells.csv']:
        df = pd.read_csv(os.path.join('./data/processed', dataset))
        print(f'# --- {dataset}: num cells: {df.shape[0]}, num genes: {df.shape[1]}')





if __name__ == '__main__':

    # plot_sc_results()

    # plot_num_permutations()

    # retrieve_table_data()

    print('done')


