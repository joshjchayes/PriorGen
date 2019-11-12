'''
_scaler

Scaler object to convert points with dimensions into dimensionless space with
variables in the range [0,1)
'''

import numpy as np

class Scaler:
    def __init__(self, parameter_limits):
        '''
        Scaler object to convert a point in parameter space to a dimensionless
        coordinate with all points between 0 and 1

        Parameters
        ----------
        parameter_limits : array_like, shape (n_variables, 2)
            The limits on each parameter, provided in (lower, upper) pairs
        '''
        parameter_limits = np.array(parameter_limits)

        if not parameter_limits.shape[1] == 2 and len(parameter_limits.shape) == 2:
            raise ValueError('Parameter limits must be provided in shape (N, 2)')


        self.parameter_limits = parameter_limits
        self.n_variables = len(parameter_limits)

        self.lower_limits = self.parameter_limits[:, 0]
        self.upper_limits = self.parameter_limits[:, 1]

    def point_to_dimensionless(self, point):
        '''
        Transforms a point into dimensionless space. Inverse of
        point_from_dimensionless

        Parameters
        ----------
        point : array_like, shape (n_variables,)
            The point to be converted to dimensionless space

        Returns
        -------
        dimensionless_point : array_like, shape (n_variables,)
            The point, but with all co-ordinates transformed to range [0, 1)
        '''
        point = np.asarray(point)

        return (point - self.lower_limits)/(self.upper_limits - self.lower_limits)

    def point_from_dimensionless(self, point):
        '''
        Inverse of point_to_dimensionless
        '''
        point = np.asarray(point)
        return (self.upper_limits - self.lower_limits) * point + self.lower_limits

    def errors_to_dimensionless(self, errors):
        '''
        Converts uncertainties into dimensionless space.

        Parameters
        ----------
        errors : array_like, shape (n_variables, ) or (n_variables, 2)
            The errors. If 1D array is provided, assumes uniform upper and
            lower errors. If 2D array provided, assumes errors are provided as
            (lower, upper) pairs.

        Returns
        -------
        dimensionless_errors : np.array, shape (n_variables, 2)
            The errors in dimensionless space. Provided as (lower, upper) pairs
            for each variable. Calculated as a length in the dimensionless
            space
        '''
        errors = np.array(errors)

        if len(errors.shape) == 1:
            # Sort out uniform upper and lower errors
            errors = np.vstack((errors, errors)).T

        if not errors.shape[0] == self.n_variables:
            raise ValueError('Incorrect number of uncertainties supplied!')
        if not errors.shape[1] == 2:
            raise ValueError('Lower and upper errors not supplied as (lower, upper)')

        # Find midpoint in the dimensionfull space, then create point 1 sigma
        # away so that we can work out the length of the vector between them in
        # dimensionless space
        x0 = np.array([(a[1]+a[0])/2 for a in self.parameter_limits])
        y_upper = x0 + errors[:, 1]
        y_lower = x0 - errors[:, 0]

        e_lower = self.point_to_dimensionless(x0) - self.point_to_dimensionless(y_lower)
        e_upper = self.point_to_dimensionless(y_upper) - self.point_to_dimensionless(x0)

        return np.vstack((e_lower, e_upper)).T
