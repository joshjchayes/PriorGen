'''
classifier.py

Module containing the PriorGen Classifier. This classifier deals with
generating the informed priors
'''

import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

from .observablesclass import ObservablesClass
from .pca_utils import find_required_components, run_PCA, pca_plot


class Classifier:
    def __init__(self, parameters, observables, n_classes=50, variance=0.999,
                 n_components=None, n_bins=20):
        '''
        The Classifier is the main PriorGen interface, which generates the
        classes of observables, as well as classifies a new observable and
        provides an informed prior which can be used in retrieval codes.

        Parameters
        ----------
        parameters : array_like, shape (N, n_parameters)
            The physical parameter values for each point we are training the
            ML classifier on. N is the number of points, whilst n_parameters is
            the number of parameters. These are all assumed to be in the same
            order. We assume that there are M variables in the model, and that
            none of them are constants.
        observables : array_like, shape (N, X)
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
        '''

        # Convert parameters and observables into np arrays
        parameters = np.array(parameters)
        observables = np.array(observables)

        # Here are some sanity checks
        if not parameters.shape[0] == observables.shape[0]:
            raise ValueError('Number of parameters given does not match number of observables. {} vs {}'.format(
                parameters.shape[0], observables.shape[0]))

        if not len(observables.shape) == 2:
            raise ValueError('Each observable should be a 1D array!!')

        # Store the parameters and observables
        self.parameters = parameters
        self.observables = observables

        # Store some other useful info
        self.n_classes = n_classes

        if n_components is None:
            # find the ideal number of components
            n_components = find_required_components(parameters, observables, variance)

        self.n_components = n_components
        self.n_bins = n_bins

        # The number of varibles in the model used to generate the observables
        self.n_variables = parameters.shape[1]

        # Do a PCA reduction on the observables. This is required because
        # k-means clustering gets very time-consuming at high dimensionality
        self._pca, self._pca_observables = self._run_PCA(n_components)

        # Now PCA is completed, we can run a k-means clustering algorithm to
        # generate a different classes of observable
        self._clustering, self._scaler = self._run_kmeans_clustering(n_classes)

        # Now we generate an ObservablesClass for each of the classes in
        # the classification.
        self.classes = self._create_observables_classes(n_bins)

    def classify_observable(self, observable):
        '''
        Classifies an observable using the trained classifier.

        Parameters
        ----------
        observable : array_like, shape (X, )
            The observable to be classified

        Returns
        -------
        label : int
            The classified label of the observable. This can be used as an
            index for referencing the ObservablesClass the observable is
            classified as.
        '''
        observable = np.array([observable])

        # Run PCA
        pca_obs = self._pca.transform(observable)

        # Scale the pbc_observable
        scaled_obs = self._scaler.transform(pca_obs)

        # Now run the prediction of the cluster
        label = self._clustering.predict(scaled_obs)
        return label[0]

    def create_dynesty_prior_transform(self, observable):
        '''
        When given an observable, creates a function which will convert a
        unit cube from dynesty's nested sampling to a physical value set which
        can be used in a log-likelihood calculation.

        Parameters
        ----------
        observable : array_like, shape (X, )
            The observable to be classified

        Returns
        -------
        transform_prior : function
            The unit cube conversion function, which uses the parameter
            distributions from the ObservablesClass which the observable is
            classified as
        '''

        data_class = self.classes[int(self.classify_observable(observable))]

        def prior_transform(unit_cube):
            '''
            A unit cube conversion function, which uses the parameter
            distributions from the ObservablesClass which the observable is
            classified as.

            Parameters
            ----------
            unit_cube : array_like, shape (n_parameters, )
                The unit cube from dynesty which will be converted to physical
                parameter space

            Returns
            -------
            physical_cube : array_like, shape (n_parameters, )
                The cube in physical space.

            '''
            # Blank array to fill with new
            physical_cube = np.zeros(len(unit_cube))

            # Loop through each of the variables and convert them using the
            # distribution functions in the ObservablesClass
            for i in range(len(physical_cube)):
                #print(data_class.convert_from_unit_interval(
                    #i, unit_cube[i]))
                #print(unit_cube[i])
                physical_cube[i] = data_class.convert_from_unit_interval(
                    i, unit_cube[i])

            return physical_cube

        return prior_transform

    def _run_PCA(self, n_components):
        '''
        Runs a principal component analysis to reduce dimensionality of
        observables.

        Parameters
        ----------
        n_components : int
            The number of principal components to keep

        Returns
        -------
        pca : sklearn.decomposition.PCA
            The scikit-learn PCA object
        reduced_d_observables : array_like, shape(N, n_components)
            The observables after PCA has been applied to them
        '''
        return run_PCA(self.parameters, self.observables, n_components)

    def _run_kmeans_clustering(self, n_clusters):
        '''
        Performs K-Means clustering on the data

        Parameters
        ----------
        n_clusters : int
            The number of clusters to use.

        Returns
        -------
        clustering : sklearn.cluster.MiniBatchKMeans
            The fitted scikit learn MiniBatchKMeans object
        scaler : scklearn.preprocessing.StandardScaler
            A scikit learn scaler which can be used to endure that new points
            are scaled correctly before classification.
        '''
        scaler = StandardScaler().fit(self._pca_observables)
        X = scaler.transform(self._pca_observables)

        clusterer = MiniBatchKMeans(n_clusters=n_clusters)
        clustering = clusterer.fit(X)

        return clustering, scaler

    def _create_observables_classes(self, n_bins):
        '''
        Creates an ObservablesClass for each of the classes.

        Parameters
        ----------
        n_bins : int
            The number of bins to use in the marginalising.
        '''
        # Loop over each cluster and make the ObservablesClass
        unique_labels = set(self._clustering.labels_)

        classes = []

        for li, label in enumerate(unique_labels):
            # Mask to only use relevant class
            mask = (self._clustering.labels_ == label)
            class_parameters = self.parameters[mask]

            center = self._clustering.cluster_centers_[li]

            classes.append(ObservablesClass(class_parameters, center, n_bins))

        return classes

    def plot_classes(self, axis_labels=None, scale_axes=True, save=True,
                     save_folder=os.getcwd()):
        '''
        Generates a corner plot for each class showing the parameter distributions

        Parameters
        ----------
        axis_labels : None or array_like, optional
            Option to provide axis labels. Should be variable names in a list,
            with names for all variables. If None, will default to using P_n
            where n is in range (0, n_variables - 1). Default is None.
        scale_axes : bool, optional
            If True, the axes of each plot will be scaled to show the full range
            of each parameter, rather than just being limited by the range of a
            parameter within the class. Default is True.
        save : bool, optional:
            If True, will save the output figures as pdf images. Default is True.
        save_folder : str, optional
            If save is True, this is the folder that the figures will
            be saved to. Default is the current working directory.

        Returns
        -------
        fig_list : array_like, shape (n_classes, )
            A list of the figures generated.
        '''

        if scale_axes:
            axis_lims = [[self.parameters[:, i].min(), self.parameters[:, i].max()] for i in range(self.n_variables)]
        else:
            axis_lims = None

        fig_list = []
        for c in self.classes:
            fig_list.append(c.plot_distribution(axis_labels, axis_lims))

        if save:
            # Saving figures
            for i in range(self.n_classes):
                path = os.path.join(save_folder, 'class_plot_{}.pdf'.format(i))
                fig_list[i].savefig(path)

        return fig_list

    def plot_pca_variance(self, n_components=None,
                          save_path='PCA_plot.pdf'):
        '''
        Produces a plot of the explained variance of the first n_components
        pricipal components, along with cumulative variance.

        Parameters
        ----------
        n_components : int, optional
            The number of principal components to keep. Default is
            self.n_components
        save_path : str, optional
            The path that the figure will be saved to. Default is
            'PCA_plot.pdf'.

        Returns
        -------
        fig : matplotlib.Figure
            The plot
        '''
        if n_components is None:
            n_components = self.n_components

        return pca_plot(self.parameters, self.observables, n_components,
                        save_path=save_path)

    def test_classification_accuracy(self, test_parameters, test_observables,
                                     return_accuracy_array=False):
        '''
        Runs classification on multiple obseravbles with known parameters and
        finds the accuracy of classification through comparing the parameters
        of the assigned class with the known values

        Parameters
        ----------
        test_parameters : array_like, shape (K, n_parameters)
            K sets of parameters associated with each of the K observables
        test_observables : array_like, shape (K, X)
            The observables associated with each of the parameters. We assume
            that the observables are 1D arrays where each entry is directly
            comparable. For example, it could be F(t), but where each entry is
            at the same value of t.
        return_accuracy_array : bool, optional
            If True, will return boolean array of which test_parameters were
            accurately classified. Default is False

        Returns
        -------
        accuracy : float
            The fraction of observables which were accurately classified
        accurately_classified : np.array, shape (K, ), optional
            Array showing which test_parameters were accurately classified.
            Returned if return_accuracy_array is True.

        Notes
        -----
        An accurate classification is defined as one which results in an
        assigned class with a parameter distribution which contains the true
        set of parameters
        '''

        accurately_classified = np.zeros(len(test_parameters), bool)

        # Classify each observable and find if the true parameters fall within
        # the assigned class parameter distribution
        for i, obs in enumerate(test_observables):
            assigned_class = self.classify_observable(obs)

            if self.classes[assigned_class].are_parameters_in_class(test_parameters[i]):
                accurately_classified[i] = True

        accuracy = sum(accurately_classified) / len(accurately_classified)

        if return_accuracy_array:
            return accuracy, accurately_classified
        return accuracy
