import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_pipeline import load_data
import scipy.stats
from statsmodels.stats.multitest import multipletests
from clustering import Cluster
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, cophenet


# ---------------------
# CORRELATION UTILITIES
# ---------------------

def pearson_correlation(x, y):
    """
    Computes similarity measure between each pair of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :return: Matrix with shape (nb_genes_1, nb_genes_2) containing the similarity coefficients
    """

    def standardize(a):
        a_off = np.mean(a, axis=0)
        a_std = np.std(a, axis=0)
        return (a - a_off) / a_std

    assert x.shape[0] == y.shape[0]
    x_ = standardize(x)
    y_ = standardize(y)
    return np.dot(x_.T, y_) / x.shape[0]


def upper_diag_list(m_):
    """
    Returns the condensed list of all the values in the upper-diagonal of m_
    :param m_: numpy array of float. Shape=(N, N)
    :return: list of values in the upper-diagonal of m_ (from top to bottom and from
             left to right). Shape=(N*(N-1)/2,)
    """
    m = np.triu(m_, k=1)  # upper-diagonal matrix
    tril = np.zeros_like(m_) + np.nan
    tril = np.tril(tril)
    m += tril
    m = np.ravel(m)
    return m[~np.isnan(m)]


def correlations_list(x, y, corr_fun=pearson_correlation):
    """
    Generates correlation list between all pairs of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :param corr_fun: correlation function taking x and y as inputs
    """
    corr = corr_fun(x, y)
    return upper_diag_list(corr)



# ---------------------
# CLUSTERING UTILITIES
# ---------------------


def hierarchical_clustering(data, corr_fun=pearson_correlation):
    """
    Performs hierarchical clustering to cluster genes according to a gene similarity
    metric.
    Reference: Cluster analysis and display of genome-wide expression patterns
    :param data: numpy array. Shape=(nb_samples, nb_genes)
    :param corr_fun: function that computes the pairwise correlations between each pair
                     of genes in data
    :return scipy linkage matrix
    """
    # Perform hierarchical clustering
    y = 1 - correlations_list(data, data, corr_fun)
    l_matrix = linkage(y, 'complete')  # 'correlation'
    return l_matrix


def compute_silhouette(data, l_matrix):
    """
    Computes silhouette scores of the dendrogram given by l_matrix
    :param data: numpy array. Shape=(nb_samples, nb_genes)
    :param l_matrix: Scipy linkage matrix. Shape=(nb_genes-1, 4)
    :return: list of Silhouette scores
    """
    nb_samples, nb_genes = data.shape

    # Form dendrogram and compute Silhouette score at each node
    clusters = {i: Cluster(index=i) for i in range(nb_genes)}
    scores = []
    for i, z in enumerate(l_matrix):
        c1, c2, dist, n_elems = z
        clusters[nb_genes + i] = Cluster(c_left=clusters[c1],
                                         c_right=clusters[c2])
        c1_indices = clusters[c1].indices
        c2_indices = clusters[c2].indices
        labels = [0] * len(c1_indices) + [1] * len(c2_indices)
        if len(labels) == 2:
            scores.append(0)
        else:
            expr = data[:, clusters[nb_genes + i].indices]
            m = 1 - pearson_correlation(expr, expr)
            m[m<0]=0
            score = silhouette_score(m, labels, metric='precomputed')
            scores.append(score)

    return scores


# ---------------------
# PLOTTING UTILITIES
# ---------------------

def overlap_score(real, sint):
    real_dens, _ = np.histogram(real, bins='auto', density=True)
    sint_dens, _ = np.histogram(sint, bins=real_dens.shape[0], density=True)
    sum_min=0
    sum_max=0
    for i in range(real_dens.shape[0]):
        _min = min(real_dens[i], sint_dens[i])
        _max = max(real_dens[i], sint_dens[i])
        sum_min+=_min
        sum_max+=_max
    
    print('Overlap score :', sum_min/sum_max)

def plot_distribution(data, label='MTB_overexpr', color='royalblue', linestyle='-', ax=None, plot_legend=True,
                      xlabel=None, ylabel=None):
    """
    Plot a distribution
    :param data: data for which the distribution of its flattened values will be plotted
    :param label: label for this distribution
    :param color: line color
    :param linestyle: type of line
    :param ax: matplotlib axes
    :param plot_legend: whether to plot a legend
    :param xlabel: label of the x axis (or None)
    :param ylabel: label of the y axis (or None)
    :return matplotlib axes
    """
    x = np.ravel(data)
    ax = sns.distplot(x,
                      hist=False,
                      kde_kws={'linestyle': linestyle, 'color': color, 'linewidth': 2, 'bw': .15},
                      label=label,
                      ax=ax)
    if plot_legend:
        plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    return ax


def plot_intensities(expr, plot_quantiles=True, dataset_name='MTB_overexpr', color='royalblue', ax=None):
    """
    Plot intensities histogram
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param plot_quantiles: whether to plot the 5 and 95% intensity gene quantiles
    :param dataset_name: name of the dataset
    :param color: line color
    :param ax: matplotlib axes
    :return matplotlib axes
    """
    x = np.ravel(expr)
    ax = sns.distplot(x,
                      hist=False,
                      kde_kws={'color': color, 'linewidth': 2, 'bw': .15},
                      label=dataset_name,
                      ax=ax)

    if plot_quantiles:
        stds = np.std(expr, axis=-1)
        idxs = np.argsort(stds)
        cut_point = int(0.05 * len(idxs))

        q95_idxs = idxs[-cut_point:]
        x = np.ravel(expr[q95_idxs, :])
        ax = sns.distplot(x,
                          ax=ax,
                          hist=False,
                          kde_kws={'linestyle': ':', 'color': color, 'linewidth': 2, 'bw': .15},
                          label='Alta variânça {}'.format(dataset_name))

        q5_idxs = idxs[:cut_point]
        x = np.ravel(expr[q5_idxs, :])
        sns.distplot(x,
                     ax=ax,
                     hist=False,
                     kde_kws={'linestyle': '--', 'color': color, 'linewidth': 2, 'bw': .15},
                     label='Baixa variânça {}'.format(dataset_name))
    plt.legend()
    plt.xlabel('Níveis absolutos')
    plt.ylabel('Densidade')
    return ax


def plot_gene_ranges(expr, dataset_name='MTB_overexpr', color='royalblue', ax=None):
    """
    Plot gene ranges histogram
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param dataset_name: name of the dataset
    :param color: line color
    :param ax: matplotlib axes
    :return matplotlib axes
    """
    nb_samples, nb_genes = expr.shape
    sorted_expr = [np.sort(expr[:, gene]) for gene in range(nb_genes)]
    sorted_expr = np.array(sorted_expr)  # Shape=(nb_genes, nb_samples)
    cut_point = int(0.05 * nb_samples)
    diffs = sorted_expr[:, -cut_point] - sorted_expr[:, cut_point]

    ax = sns.distplot(diffs,
                      hist=False,
                      kde_kws={'color': color, 'linewidth': 2, 'bw': .15},
                      label=dataset_name,
                      ax=ax)

    plt.xlabel('Gamas de genes')
    plt.ylabel('Densidade')

    return ax, diffs
