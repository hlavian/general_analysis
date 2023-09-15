import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree, to_tree, set_link_color_palette


def cluster_id_search(tree):
    nodes_list = []
    if tree.is_leaf():
        nodes_list.append(tree.get_id())
    else:
        nodes_list += cluster_id_search(tree.get_left())
        nodes_list += cluster_id_search(tree.get_right())

    return nodes_list


def find_trunc_dendro_clusters(linkage_mat, dendro):
    tree, branches = to_tree(linkage_mat, rd =True)
    ids = np.empty(linkage_mat.shape[0 ] +1, dtype=int)

    for i, clust in enumerate(dendro["leaves"]):
        ids[cluster_id_search(branches[clust])] = i

    return ids


def shade_plot(stim, ax=None, gamma=1 / 6, shade_range=(0.6, 0.9)):
    if type(stim) == list:  # these would be transitions
        _shade_plot(stim, ax=ax, gamma=gamma, shade_range=shade_range)

    elif type(stim) == Data:  # fish data
        transitions = find_transitions(Data.resampled_stim, Data.time_im_rep)
        _shade_plot(transitions, ax=ax, gamma=gamma, shade_range=shade_range)

    elif type(stim) == np.ndarray:  # stimulus array
        transitions = find_transitions(stim[:, 1], stim[:, 0])
        _shade_plot(transitions, ax=ax, gamma=gamma, shade_range=shade_range)

    elif type(stim) == tuple:  # time, lum tuple
        transitions = find_transitions(stim[1], stim[0])
        _shade_plot(transitions, ax=ax, gamma=gamma, shade_range=shade_range)


def _shade_plot(lum_transitions, ax=None, gamma=1 / 6, shade_range=(0.6, 0.9)):
    if ax is None:
        ax = plt.gca()
    shade = lum_transitions[0][1]
    for i in range(len(lum_transitions) - 1):
        shade = shade + lum_transitions[i][1]
        new_shade = shade_range[0] + np.power(np.abs(shade), gamma) * (shade_range[1] - shade_range[0])
        ax.axvspan(lum_transitions[i][0], lum_transitions[i + 1][0], color=(new_shade,) * 3)


def _find_thr(linked, n_clust):
    interval = [0, 2000]
    new_height = np.mean(interval)
    clust = 0
    n_clust = n_clust
    while clust != n_clust:
        new_height = np.mean(interval)
        clust = cut_tree(linked, height=new_height).max()
        if clust > n_clust:
            interval[0] = new_height
        elif clust < n_clust:
            interval[1] = new_height

    return new_height


def find_plot_thr(linked, n_clust):
    min_thr = _find_thr(linked, n_clust - 1)
    return min_thr


def plot_clusters_dendro(traces, stim, linkage_mat, labels, dendrolims=(900, 30),
                         thr=None, f_lim=2, gamma=1):
    fig_clust, ax = plt.subplots(3, 1, figsize=(10, 10))
    hexac = cluster_cols()

    n_clust = labels.max() + 1

    ##################
    ### Dendrogram ###
    # Compute and plot first dendrogram.
    if thr is None:
        thr = find_plot_thr(linkage_mat, n_clust)

    set_link_color_palette(hexac)

    ax_dendro = ax[2]
    ax_traces = ax[1]
    ax_clusters = ax[0]

    panel_dendro = dendrogram(linkage_mat,
                              color_threshold=thr,
                              # orientation='left',
                              distance_sort='descending',
                              show_leaf_counts=False,
                              no_labels=True,
                              above_threshold_color='#%02x%02x%02x' % (
                                  120, 120, 120))

    ax_dendro.axhline(thr, linewidth=0.7, color="k")
    ax_dendro.axis("off")

    # Plot traces matrix.
    im = ax_traces.imshow(traces[panel_dendro["leaves"], :],
                          aspect='auto', origin='lower', cmap=cm.RdBu_r,
                          vmin=-f_lim, vmax=f_lim)
    ax_traces.axes.spines['left'].set_visible(False)
    ax_traces.set_yticks([])

    # Time bar:
    dt = stim[1, 0]
    barlength = 10
    bounds = np.array([traces.shape[1] - barlength / dt,
                       traces.shape[1]])

    ##################
    # Cluster sizes ##
    # Calculate size of each defined cluster to put colored labels on the side.
    # Find indervals spanned by each cluster in the sorted traces matrix.
    # Add percentages spanned by each cluster.
    sizes = np.cumsum(np.array([np.sum(labels == i) for i in range(np.max(labels) + 1)]))
    intervals = np.insert(sizes, 0, 0)

    ##################
    # Cluster means ##

    for i in range(n_clust):
        ax_clusters.plot(np.nanmean(traces[labels == i, :], 0) +
                         i * 5, label=i, color=hexac[i])
    ax_clusters.axes.spines['left'].set_visible(False)
    ax_clusters.set_yticks([])

    barlength = 10
    ax_traces.axis("off")
    ax_clusters.axis("off")

    return fig_clust


def cluster_cols():
    color_list = ["lightblue", "lightcoral", "orange", "springgreen", "deepskyblue", "mediumpurple", "gold", "cyan",
                  "crimson",
                  "deeppink", "lawngreen", "darkviolet", "Darkgreen", "blue", "brown", "dodgerblue", "hotpink",
                  "OliveDrab", "gray", "seagreen"][0:k]
    # color_list = ["lightblue", "lightcoral", "orange", "springgreen", "deepskyblue", "mediumpurple","gold", "cyan", "crimson", "deeppink", "lawngreen", "darkviolet"]
    return color_list

