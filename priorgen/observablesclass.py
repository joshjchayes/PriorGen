'''
cluster.py

PriorGen ObservablesClasses are objectswhich contain functions useful for
a class within the PriorGen.Classifier, including the distributions for each

'''

import numpy as np
from scipy.stats import rv_discrete

class ObservablesClass:
    def __init__(self, center, distribution):
        '''
        A class of observable. This contains information on a class within a
        Classifier.

        Parameters
        ----------
        center : array_like, shape (n_components, )
            The center of the cluster in PCA space
        distribution : tuple output from np.histogram, length [n_variables]
            The marginalised distributions of each of the variables in
            parameter space. Sizes should be ((n_bins,), (n_bins + 1, ))

        Notes
        -----
        The center is in PCA space, and the distribution is in parameter space.
        This is because a new set of data will be run through the Classifier by
        undergoing the PCA transform, but we are actually interested in
        acquiring the parameter distribution for the new set of data.
        '''

        self.center = center

        # For each variable, produce a distribution function which can be drawn
        # from
        n_bins = len(distribution[0][0])

        self.distribution_functions = np.zeros(len(distribution), object)
        self.bin_centers = np.zeros((len(distribution), n_bins))

        # The edges of the bins in dimensionless space [0-1)
        self.bin_edges = np.zeros((len(distribution), n_bins, 2))

        # These are used to uniformly distribute within bins, since the
        # distribution functions only provide a bin label, rather than a
        # value from within them. They are the coefficients needed to turn
        # any value in a bin onto a linear distribution between 0 and 1.
        self.linear_coeffs = np.zeros((len(distribution), n_bins, 2))

        for i, di in enumerate(distribution):
            vals = di[0]
            flat_bin_edges = di[1]

            # Calculate bin centres
            bin_centers = (flat_bin_edges[1:] - flat_bin_edges[:-1])/2

            # Normalise the histogram - we are using this for probabilities!
            vals = vals/sum(vals)

            # Work out the cumulative distribution - this will give us the
            # unitless bin edge values!
            unitless_bin_edges = np.hstack((0, np.cumsum(vals)))

            # need to work out the linear coefficients now
            # First: the gradient
            m = (flat_bin_edges[1:] - flat_bin_edges[:-1])/(unitless_bin_edges[1:] - unitless_bin_edges[:-1])

            # Second: y-intercept
            c = flat_bin_edges[1:] - unitless_bin_edges[1:] * (flat_bin_edges[1:] - flat_bin_edges[:-1])/(unitless_bin_edges[1:] - unitless_bin_edges[:-1])

            self.linear_coeffs[i] = np.vstack((m, c)).T

            # make the distribution functions
            xk = np.arange(n_bins)
            self.distribution_functions[i] = rv_discrete(values=(xk, vals))
            self.bin_centers[i] = bin_centers

    def convert_from_unit_interval(self, param_idx, unitless_value):
        '''
        Converts a parameter from a unitless value in range [0,1) to a
        physical value using the parameter distributions. This is used as the
        prior defining function in retrieval.

        Parameters
        ----------
        param_idx : int
            The index of the parameter to be converted
        unitless_value : float
            The unitless value to convert to a physical value

        Returns
        -------
        physical_value : float
            The physical value.
        '''

        # Obtain the bin index
        bin_idx = self.distribution_functions[param_idx].ppf(unitless_value)

        # Now we convert the unitless value into a physical value
        m = self.linear_coeffs[bin_idx, 0]
        c = self.linear_coeffs[bin_idx, 1]
        return m * unitless_value + c
