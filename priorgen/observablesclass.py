'''
cluster.py

PriorGen ObservablesClasses are objectswhich contain functions useful for
a class within the PriorGen.Classifier, including the distributions for each

'''

import numpy as np
from scipy.stats import rv_discrete

import matplotlib
matplotlib.use('Agg')


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.stats import kde

#plt.ioff()

class ObservablesClass:
    def __init__(self, parameters, center, n_bins):
        '''
        A class of observable. This contains information on a class within a
        Classifier.

        Parameters
        ----------
        parameters : np.array, shape (n_members, n_parameters)
            The physical parameters of each member of the ObservablesClass.
        center : array_like, shape (n_components, )
            The center of the cluster in PCA space.
        n_bins : int
            The number of bins to split each maginalised parameter distribution
            into. The more bins you have, the more detail you will have on the
            shape of each class' prior. However, if you have too many, you will
            encounter issues of undersampling.

        Notes
        -----
        If a bin in the marginalised parameter distribution is empty, a
        RuntimeWarning will be raised. This can be safely ignored as an
        empty bin will never be selected by the nested sampler routines
        '''
        self.parameters = parameters
        self.center = center
        self.n_bins = n_bins
        self.n_variables = self.parameters.shape[1]

        # Generate the marginalised distributions
        distribution = self._generate_distributions(n_bins)

        # The histogrammed data from _generate_distributions, stored in
        # physical parameter space
        self._distributions = distribution

        # Make some blank arrays we will populate in a minute
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

            # NOTE: If a bin is empty, then a RuntimeWarning will be raised
            # due to a divide by zero. This is okay, as these bins will never
            # be selected by the random selection anyway.

            self.linear_coeffs[i] = np.vstack((m, c)).T

            # make the distribution functions
            xk = np.arange(n_bins)
            self.distribution_functions[i] = rv_discrete(values=(xk, vals))
            self.bin_centers[i] = bin_centers

    def _generate_distributions(self, n_bins):
        '''
        Generates the marginalised distribution for each of the variables

        Parameters
        ----------
        n_bins : int
            The number of bins to use in marginalisation histograms

        Returns
        -------
        binned_data : array_like, shape (n_variables, )
            The marginalised distributions. Each entry in the above array is a
            tuple output from numpy.histogram with shape (n_bins, n_bins + 1).
        '''
        binned_data = []

        for vi in range(self.n_variables):
            vals = self.parameters[:, vi]
            h = np.histogram(vals, n_bins)
            binned_data.append(h)

        return binned_data

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
        bin_idx = int(self.distribution_functions[param_idx].ppf(unitless_value))

        # Now we convert the unitless value into a physical value
        m = self.linear_coeffs[param_idx, bin_idx, 0]
        c = self.linear_coeffs[param_idx, bin_idx, 1]

        return m * unitless_value + c

    def plot_distribution(self, axis_labels=None, axis_lims=None, savepath=None):
        '''
        Generates a corner plot for the parameter distributions, along with the
        marginalised distributions which are subsequently used as informed
        priors

        Parameters
        ----------
        axis_labels : None or array_like, optional
            Option to provide axis labels. Should be variable names in a list,
            with names for all variables. If None, will default to using P_n
            where n is in range (0, n_variables - 1). Default is None.
        axis_lims : None or array_like, shape (n_variables, 2), optional
            The lower and upper limits of each axis for each variable. If
            provided, the range of the relevant axes will be changed. Default
            is None.
        savepath : str or None, optional
            If provided, will save the figure to the specified path. Default
            is None.
        '''
        plt.ioff()

        if axis_labels is None:
            axis_labels = ['P_{}'.format(i) for i in range(self.n_variables)]
        else:
            if not len(axis_labels) == self.n_variables:
                raise ValueError('{} axis labels provided for {} variables'.format(len(axis_labels), self.n_variables))

        if axis_lims is not None:
            axis_lims = np.array(axis_lims)
            if not axis_lims.shape == (self.n_variables, 2):
                raise ValueError('axis_lims has shape {} but should have shape {}'.format(axis_lims.shape, (self.n_variables, 2)))

        fig = plt.figure(figsize=(10,10))

        gs = gridspec.GridSpec(self.n_variables, self.n_variables)



        for i in range(self.n_variables):
            for j in range(i + 1):
                ax = fig.add_subplot(gs[i, j])

                xlabel = axis_labels[j]
                ylabel = axis_labels[i]

                # DO SOME THINGS WITH THE AXES, LABELS AND TICKS.
                if i == self.n_variables - 1:
                    # add in the x labels
                    ax.set_xlabel(xlabel)
                else:
                    # Remove x labels
                    ax.tick_params(axis='x',
                                   which='both',
                                   labelbottom=False)
                if j == 0:
                    # Add in the y axis and labels
                    ax.set_ylabel(ylabel)
                else:
                    # Remove y labels
                    ax.tick_params(axis='y',
                                   which='both',
                                   labelleft=False)

                if axis_lims is not None:
                    # Rescale the axes
                    ax.set_xlim(*axis_lims[j])
                    if not i == j:
                        # Avoid messing with the y axis on the marginal plots
                        ax.set_ylim(*axis_lims[i])

                # PLOT
                if i == j:
                    # Marginalised plot.
                    vals = self.parameters[:, i]
                    h = ax.hist(vals, bins=self.n_bins, histtype='stepfilled', color=plt.cm.BuGn(220), edgecolor='k', linewidth=2.5)

                else:
                    # Contour plot
                    x = self.parameters[:,j]
                    y = self.parameters[:,i]

                    # Set up a meshgrid
                    k = kde.gaussian_kde(np.array([x,y]))

                    if axis_lims is None:
                        xi, yi = np.mgrid[x.min():x.max():self.n_bins*1j, y.min():y.max():self.n_bins*1j]

                    else:
                        # Need to use differentlimits for meshgrid as we are changing the axis lims
                        xmin = axis_lims[j, 0]
                        xmax = axis_lims[j, 1]
                        ymin = axis_lims[i, 0]
                        ymax = axis_lims[i, 1]
                        xi, yi = np.mgrid[xmin:xmax:self.n_bins*1j, ymin:ymax:self.n_bins*1j]

                    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

                    # Make the plot itself!
                    pcm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
                    cont = ax.contour(xi, yi, zi.reshape(xi.shape))

        plt.close()
        return fig

    def are_parameters_in_class(self, parameters):
        '''
        Checks to see if a set of parameters are possible within the
        parameter distribution of the ObservablesClass

        Parameters
        ----------
        parameters : array_like, shape (n_parameters, )
            The parameter values to check.

        Returns
        -------
        parameters_in_class : bool
            Returns True if parameters are possible with the parameter
            distributions, False otherwise
        '''

        # Loop over each parameter
        for i, p in enumerate(parameters):
            # Check parameter is within in min and max values of distribution
            if p < self._distributions[i][1][0] or p > self._distributions[i][1][-1]:
                return False

            # Find the histogram bin containing parameter
            for j in range(self.n_bins):
                if self._distributions[i][1][j] < p < self._distributions[i][1][j + 1]:
                    bin_idx = j
                    break

            # Check histogram bin is non-zero
            if self._distributions[i][0][bin_idx] == 0:
                return False

        return True
