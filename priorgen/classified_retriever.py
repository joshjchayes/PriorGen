'''
ClassifiedRetriever

A general retriever to use a PriorGen classifier in retrieval.

The current build only uses nested sampling provided by dynesty.
'''

from .classifier import Classifier
import dynesty
import numpy as np
import csv


class ClassifiedRetriever:
    def __init__(self, training_parameters, training_observables, n_classes=50,
                 variance=0.999, n_components=None, n_bins=20, n_nuisance=0,
                 nuisance_limits=None):
        '''
        The ClassifiedRetriever is built to use a Classifier to run retrievals
        using the informed priors generated by the Classifer

        Parameters
        ----------
        training_parameters : array_like, shape (N, M)
            The physical parameter values for each point we are training the
            ML classifier on. N is the number of points, whilst M is the
            physical value for each parameter. These are all assumed to be in
            the same order. We assume that there are M variables in the model,
            and that none of them are constants.
        training_observables : array_like, shape (N, X)
            The observables associated with each of the parameters. We assume
            that the observables are 1D arrays where each entry is directly
            comparable. For example, it could be F(t), but where each entry is
            at the same value of t.
        n_classes : int, optional
            The number of different classes to use when training the classifier
            Default is 50.
        variance : float, optional
            The fraction of explained variance to keep in the principal
            components. Default is 0.999
        n_components : int or None, optional
            If provided, will override the `variance` kwarg and specify the
            number of principal components to use when conducting PCA on
            the observables. Default is None.
        n_bins : int, optional
            The number of bins to split each maginalised parameter distribution
            into. The more bins you have, the more detail you will have on the
            shape of each class' prior. However, if you have too many, you will
            encounter issues of undersampling. Default is 20.
        n_nuisance : int, optional
            The number of nuisance parameters to include in fitting. Can be
            changed by calling set_nuisance_parameters. Default is 0.
        nuisance_limits : None, or array_like, shape (n_nuisance, 2)
            The limits for each nuisance parameter, provided as (lower, upper)
            pairs. If None, defaults to (-1000, 1000) for each parameter.
        '''
        self.classifier = Classifier(training_parameters, training_observables,
                                     n_classes, variance, n_components, n_bins)

        self.set_nuisance_parameters(n_nuisance)

    def run_dynesty(self, data_to_fit, lnprob, nlive=200, bound='multi',
                    sample='auto', maxiter=None, maxcall=None, dlogz=None,
                    filepath='output.csv', **dynesty_kwargs):
        '''
        Runs nested sampling retrieval through Dynesty using the Classifier
        to inform priors

        Parameters
        ----------
        data_to_fit : array_like, shape (X,)
            The data you want to fit. Required for classification purposes.
        lnprob : function
            A function which must be passed a set of parameters and returns
            their ln likelihood. Signature should be `lnprob(params)` where
            params is an array with shape (n_variables, ). Note that you will
            need to have hard-coded the data and associated uncertainties into
            the `lnprob` function.
        nlive : int, optional
            The number of live points to use in the nested sampling. Default is
            200.
        bound : str, optional
            Method used to approximately bound the prior using the current set
            of live points. Conditions the sampling methods used to propose new
            live points. Choices are no bound ('none'), a single bounding
            ellipsoid ('single'), multiple bounding ellipsoids ('multi'), balls
            centered on each live point ('balls'), and cubes centered on each
            live point ('cubes'). Default is 'multi'.
        sample : str, optional
            Method used to sample uniformly within the likelihood constraint,
            conditioned on the provided bounds. Unique methods available are:
            uniform sampling within the bounds('unif'), random walks with fixed
            proposals ('rwalk'), random walks with variable (“staggering”)
            proposals ('rstagger'), multivariate slice sampling along preferred
            orientations ('slice'), “random” slice sampling along all
            orientations ('rslice'), and “Hamiltonian” slices along random
            trajectories ('hslice'). 'auto' selects the sampling method based
            on the dimensionality of the problem (from ndim). When ndim < 10,
            this defaults to 'unif'. When 10 <= ndim <= 20, this defaults to
            'rwalk'. When ndim > 20, this defaults to 'hslice' if a gradient is
            provided and 'slice' otherwise. 'rstagger' and 'rslice' are
            provided as alternatives for 'rwalk' and 'slice', respectively.
            Default is 'auto'.
        maxiter : int or None, optional
            The maximum number of iterations to run. If None, will run until
            stopping criterion is met. Default is None.
        maxcall : int or None, optional
            If not None, sets the maximum number of calls to the likelihood
            function. Default is None.
        **dynesty_kwargs : optional
            kwargs to be passed to the dynesty.NestedSampler() initialisation

        Returns
        -------
        results : dict
            The dynesty results dictionary, with the addition of the following
            attributes:
            weights - normalised weights for each sample
            cov - the covariance matrix
            uncertainties - the uncertainty on each fitted parameter,
                calculated from the square root of the diagonal of the
                covariance matrix.
        '''

        # First up, we need to define some variables for the Retriever
        # Number of dimensions we are retrieving
        n_dims = self.classifier.n_variables + self.n_nuisance

        # Make the prior transform function
        prior_transform = self.classifier.create_dynesty_prior_transform(
            data_to_fit, self.n_nuisance, self.nuisance_limits)

        # Set up and run the sampler here!!
        sampler = dynesty.NestedSampler(lnprob, prior_transform,
                                n_dims, bound=bound, sample=sample,
                                update_interval=float(n_dims), nlive=nlive,
                                **dynesty_kwargs)

        sampler.run_nested(maxiter=maxiter, maxcall=maxcall, dlogz=dlogz)

        results = sampler.results

        # Get some normalised weights
        results.weights = np.exp(results.logwt - results.logwt.max()) / \
            np.sum(np.exp(results.logwt - results.logwt.max()))

        # Calculate a covariance matrix for these results to get uncertainties
        cov = np.cov(results.samples, rowvar=False, aweights=results.weights)

        # Get the uncertainties from the diagonal of the covariance matrix
        diagonal = np.diag(cov)
        uncertainties = np.sqrt(diagonal)

        # Add the covariance matrix and uncertainties to the results object
        results.cov = cov
        results.uncertainties = uncertainties

        self._print_best(results)
        self._save_results(results, filepath)


        return results

    def _save_results(self, results, filepath):
        '''
        Saves the results to a file
        '''
        write_dict = []

        best_results = results.samples[np.argmax(results.logl)]

        for i in range(self.classifier.n_variables):
            value = best_results[i]
            unc = results.uncertainties[i]

            write_dict.append({'Variable':i, 'Best value' : value,
                               'Uncertainty' : unc})


        with open(filepath, 'w') as f:
            columns = ['Variable', 'Best value', 'Uncertainty']
            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            writer.writerows(write_dict)



    def _print_best(self, results):
        '''
        Prints the best results to terminal

        Parameters
        ----------
        results : dynesty.results.Results
            The Dynesty results object, but must also have weights, cov and
            uncertainties as entries.
        '''
        best_results = results.samples[np.argmax(results.logl)]

        print('Best results:')

        for i in range(self.classifier.n_variables):
            value = round(best_results[i], 4)
            unc = round(results.uncertainties[i], 4)

            print('Variable {}: {}±{}'.format(i, value, unc))

    def set_nuisance_parameters(self, n_nuisance, nuisance_limits=None):
        '''
        Sets n nusiance parameters to fitting. The nusiance parameters must be
        included in the lnprob function.

        Parameters
        ----------
        n_nuisance : int
            The number of nuisance parameters
        nuisance_limits : None, or array_like, shape (n_nuisance, 2)
            The limits for each nuisance parameter, provided as (lower, upper)
            pairs. If None, defaults to (-1000, 1000) for each parameter.
        '''

        if type(n_nuisance) is not int:
            raise ValueError('n_nuisance must be an integer!')
        if n_nuisance < 0:
            raise ValueError("Can't have negative nuisance parameters!")

        if nuisance_limits is None:
            nuisance_limits = np.array([[-1000, 1000] for i in range(n_nuisance)])


        if not n_nuisance == 0:
            nlimshape = nuisance_limits.shape
            if not len(nlimshape) == 2:
                raise ValueError('Invalid nuisance_limits shape {}'.format(nlimshape))
            if not nlimshape[0] == n_nuisance:
                raise ValueError('{} limits provided for {} nuisance parameters'.format(nlimshape[0], n_nuisance))
            if not nlimshape[1] == 2:
                raise ValueError('Limits need to be provided as (lower, upper) pairs.')
        self.n_nuisance = n_nuisance
        self.nuisance_limits = nuisance_limits
