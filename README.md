# PriorGen
`PriorGen` is a Python 3 package designed to use the Hayes et al. (2019) method of using machine learning classification to generate informed priors which can be used in Bayesian retrieval. `PriorGen` users can either use the `ClassifiedRetriever` to run Baysian retrieval on their data, or create a `Classifier` and incorporate it into their own retrieval algorithms.

In order to use `PriorGen` effectively, you will require a trusted forward model which takes continuous parameter values as inputs and produces some single dimensioned output F(x).

## Contents

- [Requirements](#requirements)

- [The `Classifier`](#classifier)

 - [The `ClassifiedRetriever`](#retriever)


<a name="requirements"></a>
## Requirements
Along with an installation of Python 3 (with the standard Conda distribution packages), `priorgen` requires the following packages to be installed:

- [scikit-learn](https://scikit-learn.org/stable/index.html) - For handling the machine learning classification.

- [dynesty](https://dynesty.readthedocs.io/en/latest/index.html) - For nested sampling retrieval.


<a name="classifier"></a>
## The `Classifer`
In `PriorGen`, the `Classifer` is the object which deals with classification of a set of training data into different `ObservablesClass` objects, and handles classification of a previously unseen set of data.

In order to create a `Classifer`, you must provide it with two arrays - the parameters and the observables.

The observables are what the ML classifier is trained on, and are a single-dimensioned function F(x). The parameters should be provided as M-dimensional sets of values which were used to generate each observable using a trusted forward model.

The set of observables are then run through a principal component analysis (PCA) routine to perform a dimensionality reduction, before a k-means clustering routine is run to divide the observables into classes. The physical parameters for each of these classes are binned into histograms, and these distributions then act as a marginalised prior for each of the parameters in Bayesian retrieval.

The `ObservablesClass` is a Python object containing information on the parameter distributions of each of the classes found by the k-means clustering. It also contains a function which can convert a unitless value in range [0, 1] to a physical value, which is of use in nested sampling routines.

<a name="retriever"></a>
## The `ClassifiedRetriever`
Whilst a user can construct a bespoke retrieval algorithm which uses a `Classifier` to give informed priors, `PriorGen` provides the `ClassifiedRetriever` object which can run basic retrieval on a set of data.

The `ClassifiedRetriever` must be initialised in the same way as the `Classifier`, and generates a `Classifier` itself. The user must then create a function `lnprob` which generates a log probability of a set of parameters fitting the data which is being fitted. This can then be passed to the retrieval functions within the `ClassifiedRetriever` and retrieval will then be run using the member class parameter distributions as informed priors.

Currently only nested sampling retrieval is implemented using [dynesty](https://dynesty.readthedocs.io/en/latest/index.html), which can be accessed through `ClassifiedRetriever.run_dynesty()`. MCMC fitting may be provided in later versions but is not currently available.
