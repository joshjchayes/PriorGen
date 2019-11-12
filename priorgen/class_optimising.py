'''
optimising

Module containing functions to help with optimising classification schemes.
Functionality includes ability to calculate largest number of classes which
can be used in a Classifier to give a certain level of accuracy.

This is particularly useful if you are going to be running a trained classifier
on data which is partially degraded by noise.

'''

from priorgen import Classifier
import numpy as np

def find_classification_accuracies(training_parameters, training_observables,
                                   test_parameters, test_observables,
                                   min_classes=2, max_classes=30, step=1,
                                   variance=0.999):
    '''
    Generates a series of Classifiers using different numbers of classes and
    calculates the classification accuracy of each.

    Parameters
    ----------
    training_parameters : array_like, shape (N, n_parameters)
        The physical parameter values for each point we are training the
        ML classifier on. N is the number of points, whilst n_parameters is
        the number of parameters. These are all assumed to be in the same
        order. We assume that there are M variables in the model, and that
        none of them are constants.
    training_observables : array_like, shape (N, X)
        The observables associated with each of the training_parameters. We
        assume that the observables are 1D arrays where each entry is directly
        comparable. For example, it could be F(t), but where each entry is
        at the same value of t.
    test_parameters : array_like, shape (K, n_parameters)
        K sets of parameters associated with each of the K observables
    test_observables : array_like, shape (K, X)
        The observables associated with each of the test_parameters. We assume
        that the observables are 1D arrays where each entry is directly
        comparable. For example, it could be F(t), but where each entry is
        at the same value of t.
    min_classes : int, optional
        The minimum number of classes to try. Default is 2.
    max_classes : int, optional
        The maximum number of classes to try. Default is 30.
    step : int, optional
        The step size to sample number of classes over. Default is 1.
    variance : float, optional
        The fraction of explained variance to keep in the principal
        components. Default is 0.999

    Returns
    -------
    classes : np.array, shape (n_test_classes, )
        The number of classes being tested
    accuracy : np.array, shape (n_test_classes, )
        The fractional accuracy associated with each class
    '''
    accuracy = []
    classes = np.arange(min_classes, max_classes, step)
    for i in range(min_classes, max_classes, step):
        print('Testing {} classes...'.format(i))
        classifier = Classifier(training_parameters, training_observables,
                                i, variance)
        accuracy.append(classifier.test_classification_accuracy(test_parameters, test_observables))

    return classes, np.array(accuracy)


def find_maximum_acceptable_classes(training_parameters, training_observables,
                                    test_parameters, test_observables,
                                    required_accuracy=0.99, min_classes=2,
                                    max_classes=30, step=1, variance=0.999):
    '''
    Calculates the maximum number of classes which can be used on a set of
    test data to produce a given accuracy of classification.

    Parameters
    ----------
    training_parameters : array_like, shape (N, n_parameters)
        The physical parameter values for each point we are training the
        ML classifier on. N is the number of points, whilst n_parameters is
        the number of parameters. These are all assumed to be in the same
        order. We assume that there are M variables in the model, and that
        none of them are constants.
    training_observables : array_like, shape (N, X)
        The observables associated with each of the training_parameters. We
        assume that the observables are 1D arrays where each entry is directly
        comparable. For example, it could be F(t), but where each entry is
        at the same value of t.
    test_parameters : array_like, shape (K, n_parameters)
        K sets of parameters associated with each of the K observables
    test_observables : array_like, shape (K, X)
        The observables associated with each of the test_parameters. We assume
        that the observables are 1D arrays where each entry is directly
        comparable. For example, it could be F(t), but where each entry is
        at the same value of t.
    required_accuracy : float, optional
        The minimum acceptable classification accuracy allowed. Default is 0.99
    min_classes : int, optional
        The minimum number of classes to try. Default is 2.
    max_classes : int, optional
        The maximum number of classes to try. Default is 30.
    step : int, optional
        The step size to sample number of classes over. Default is 1.
    variance : float, optional
    variance : float, optional
        The fraction of explained variance to keep in the principal
        components. Default is 0.999

    '''
    n_classes, accuracy = find_classification_accuracies(training_parameters,
                                              training_observables,
                                              test_parameters,
                                              test_observables,
                                              min_classes, max_classes, step,
                                              variance)

    if not np.any(accuracy >= required_accuracy):
        return 1

    idx = np.where(accuracy >= required_accuracy)[0].max()

    return n_classes[int(idx)]
