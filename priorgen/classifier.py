'''
classifier.py

Module containing the PriorGen Classifier. This classifier deals with
generating the informed priors
'''

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

from .observablesclass import ObservablesClass


class Classifier:
    def __init__(self, parameters, observables, n_classes=50, n_components=15,
                 n_bins=20):
        '''
        The Classifier is the main PriorGen interface, which generates the
        classes of observables, as well as classifies a new observable and
        provides an informed prior which can be used in retrieval codes.

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
        n_classes : int, optional
            The number of different classes to use when training the classifier
            Default is 50.
        n_components : int, optional
            The number of principal components to use when conducting PCA on
            the observables. Default is 15.
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
        # Run PCA
        pca_obs = self._pca.transform(observable)

        # Scale the pbc_observable
        scaled_obs = self._scaler.transform(pca_obs)

        # Now run the prediction of the cluster
        label = self._clustering.predict(scaled_obs)
        return label

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

        data_class = self.classify_observable(observable)

        def prior_transform(unit_cube):
            '''
            A unit cube conversion function, which uses the parameter
            distributions from the ObservablesClass which the observable is
            classified as.

            Parameters
            ----------
            unit_cube : array_like, shape (N, )
                The unit cube from dynesty which will be converted to physical
                parameter space

            Returns
            -------
            physical_cube : array_like, shape (N, )
                The cube in physical space.

            '''
            # Blank array to fill with new
            physical_cube = np.zeros(len(unit_cube))

            # Loop through each of the variables and convert them using the
            # distribution functions in the ObservablesClass
            for i in range(len(physical_cube)):
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
        pca = PCA(n_components=n_components)
        fitted_pca = pca.fit(self.observables)
        reduced_d_observables = fitted_pca.transform(self.observables)

        return pca, reduced_d_observables

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

    def _generate_distributions(self, n_bins):
        '''
        Generates the marginalised distribution for each of the variables
        for each of the classes.

        Parameters
        ----------
        n_bins : int
            The number of bins to use in the marginalising.

        Returns
        -------
        binned_data : array_like,
            The d
        '''
        # Find all the different class labels.
        unique_labels = set(self._clustering.labels_)

        binned_data = []

        # Now loop through each class and bin each variable using np.histogram
        for li, label in enumerate(unique_labels):
            # Mask to only use relevant class
            mask = (self._clustering.labels_ == label)
            class_parameters = self.parameters[mask]

            binned_label_data = []

            # Now loop through each variable and bin to a histogram
            for vi in range(self.n_variables):
                vals = class_parameters[:, vi]
                h = np.histogram(vals, n_bins)
                binned_label_data.append(h)

            binned_data.append(binned_label_data)

        return binned_data

    def _create_observables_classes(self, n_bins):
        '''
        Creates an ObservablesClass for each of the classes.

        Parameters
        ----------
        n_bins : int
            The number of bins to use in the marginalising.
        '''
        # Bin to generate marginalised distributions
        binned_data = self._generate_distributions(n_bins)

        classes = []

        # Loop over each cluster and make the ObservablesClass
        for i in range(self.n_classes):
            center = self._clustering.cluster_centers_[i]

            # Make the ObservablesClass
            classes.append(ObservablesClass(center, binned_data[i]))

        return classes
