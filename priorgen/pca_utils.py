'''
pca_utils.py

Module containing functions to work out optimal number of principal components
to contain a given percentage of variance, and generate diagnostic plots
'''

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import numpy as np


def run_PCA(parameters, observables, n_components):
    '''
    Runs a principal component analysis to reduce dimensionality of
    observables.

    Parameters
    ----------
    parameters : array_like, shape (N, M)
        The physical parameter values for each point we are training the
        ML classifier on. N is the number of points, whilst M is the
        physical value for each parameter. These are all assumed to be in
        the same order. We assume that there are M variables in the model,
        and that none of them are constants.
    observables : array_like, shape (N, X)
        The observables associated with each of the parameters. We assume
        that the observables are 1D arrays where each entry is directly
        comparable. For example, it could be F(t), but where each entry is
        at the same value of t.
    n_components : int
        The number of principal components to keep

    Returns
    -------
    pca : sklearn.decomposition.PCA
        The scikit-learn PCA object
    reduced_d_observables : array_like, shape(N, n_components)
        The observables after PCA has been applied to them
    '''

    pca = PCA(n_components=n_components)
    fitted_pca = pca.fit(observables)
    reduced_d_observables = fitted_pca.transform(observables)

    return pca, reduced_d_observables


def pca_plot(parameters, observables, n_components, save=True,
             save_path='PCA_plot.pdf'):
    '''
    Produces a plot of the explained variance of the first n_components
    principal components, along with a cumulative variance

    Parameters
    ----------
    parameters : array_like, shape (N, M)
        The physical parameter values for each point we are training the
        ML classifier on. N is the number of points, whilst M is the
        physical value for each parameter. These are all assumed to be in
        the same order. We assume that there are M variables in the model,
        and that none of them are constants.
    observables : array_like, shape (N, X)
        The observables associated with each of the parameters. We assume
        that the observables are 1D arrays where each entry is directly
        comparable. For example, it could be F(t), but where each entry is
        at the same value of t.
    n_components : int
        The number of principal components to keep
    save : bool, optional:
        If True, will save the output figure to save_path. Default is True.
    save_path : str, optional
        If save is True, this is the path that the figures will
        be saved to. Default is 'PCA_plot.pdf'.

    Returns
    -------
    fig : matplotlib.Figure
        The pca plot
    '''

    pca, _ = run_PCA(parameters, observables, n_components)
    variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance).round(4)

    fig, ax = plt.subplots(2,1, sharex=True)

    # Plot the
    ax[0].bar(np.arange(n_components), variance, label='Associated variance')
    #ax[0].set_xlabel('Principal component')
    ax[0].set_ylabel('Fractional variance')
    ax[0].set_yscale('log')

    ax[1].plot(np.arange(n_components), cumulative_variance, 'r', label='Cumulative variance')
    ax[1].set_xlabel('Principal component')
    ax[1].set_ylabel('Cumulative variance')
    ax[1].margins(x=0.01)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    if save:
        fig.savefig(save_path)

    return fig


def find_required_components(parameters, observables, variance):
    '''
    Calculates the number of principal components required for reduced
    dimensionality obserables to contain a given fraction of explained variance

    Parameters
    ----------
    parameters : array_like, shape (N, M)
        The physical parameter values for each point we are training the
        ML classifier on. N is the number of points, whilst M is the
        physical value for each parameter. These are all assumed to be in
        the same order. We assume that there are M variables in the model,
        and that none of them are constants.
    observables : array_like, shape (N, X)
        The observables associated with each of the parameters. We assume
        that the observables are 1D arrays where each entry is directly
        comparable. For example, it could be F(t), but where each entry is
        at the same value of t.
    variance : float
        The fraction of explained variance you want the principal components
        to contain

    Returns
    -------
    n_components : int
        The smallest number of principal comonents required to contain the
        specified fraction of explained variance
    '''
    if not 0 <= variance < 1:
        raise ValueError('variance must be between 0 and 1')

    # run PCA and keep all components
    pca, _ = run_PCA(parameters, observables, None)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # The +1 is required because the first part finds an index where the
    # cumulative explained variance ratio is larger than the threshold
    # and the indices start from 0
    return np.where(cumulative_variance >= variance)[0][0] + 1
