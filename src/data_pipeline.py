import numpy as np
import pandas as pd
import pickle
import random
import csv

DATA_DIR = '../data/MTB'
SYNTHETIC_DIR = '../data/artificial'
DEFAULT_DATAFILE = 'GSE59086_overexpressed.csv'  #(n_genes, n_samples)



def _parse(lines):
    """
    Parse lines from expression file
    :param lines: list of lines. First row and column have the sample and the gene names, respectively.
    :return: expression np.array with Shape=(nb_samples, nb_genes), gene names with Shape=(nb_genes,) and sample
             names with Shape=(nb_samples,)
    """
    # Parse lines containing gene expressions
    lines_iter = iter(lines)
    line = next(lines_iter, None)
    sample_names = line.split(',')[1:]
    gene_names = []
    expression_values = []
    line = next(lines_iter, None)
    while line:
        split = line.split(',')
        gene_name, expr = split[0], split[1:]
        gene_names.append(gene_name)
        expression_values.append(expr)
        line = next(lines_iter, None)
    expression_values = np.array(expression_values, dtype=np.float64).T

    return expression_values, gene_names, sample_names


def load_data(name=DEFAULT_DATAFILE):
    """
    Loads data from the file with the given name in DATA_DIR. 
    :param name: name from file in DATA_DIR containing the expression data
    
    :return: expression np.array with Shape=(nb_samples, nb_genes), gene symbols with Shape=(nb_genes,) and sample
             names with Shape=(nb_samples,)
    """
    # Parse data file
    with open('{}/{}'.format(DATA_DIR, name), 'r') as f:
        lines = f.readlines()
    #df = pd.read_csv(name, header=0, index_col=0)
    expr, gene_names, sample_names = _parse(lines)
    #expr, gene_names, sample_names = np.array(df), list(df.index),list(df.columns)
    print('Found {} genes in datafile'.format(len(gene_names)))

    return expr, gene_names, sample_names


def save_synthetic(name, expr, gene_symbols):
    """
    Saves expression data with Shape=(nb_samples, nb_genes) to pickle file with the given name in SYNTHETIC_DIR.
    :param name: name of the file in SYNTHETIC_DIR where the expression data will be saved
    :param expr: np.array of expression data with Shape=(nb_samples, nb_genes)
    :param gene_symbols: list of gene symbols matching the columns of expr
    """
    file = '{}/{}.pkl'.format(SYNTHETIC_DIR, name)
    data = {'expr': expr,
            'gene_symbols': gene_symbols}
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_synthetic(name):
    """
    Loads expression data from pickle file with the given name (produced by save_synthetic function)
    :param name: name of the pickle file in SYNTHETIC_DIR containing the expression data
    :return: np.array of expression with Shape=(nb_samples, nb_genes) and list of gene symbols matching the columns
    of expr
    """
    file = '{}/{}.pkl'.format(SYNTHETIC_DIR, name)
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data['expr'], data['gene_symbols']


def write_csv(name, expr, gene_symbols, sample_names=None, nb_decimals=5):
    """
    Writes expression data to a CSV file
    :param name: file name
    :param expr: expression matrix. Shape=(nb_samples, nb_genes)
    :param gene_symbols: list of gene symbols. Shape=(nb_genes,)
    :param sample_names: list of gene samples or None. Shape=(nb_samples,)
    :param nb_decimals: number of decimals for expression values
    """
    expr_rounded = np.around(expr, decimals=nb_decimals)
    with open('{}/{}'.format(CSV_DIR, name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow([' '] + [g for g in gene_symbols])
        for i, e in enumerate(expr_rounded):
            sample_name = ' '
            if sample_names is not None:
                sample_name = sample_names[i]
            writer.writerow([sample_name] + list(e))