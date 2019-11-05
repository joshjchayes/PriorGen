'''
retrieval_utils.py

Module with some useful functions to make using retrievals easier
'''

def make_nested_lnprob(y, yerr, model):
    '''
    Makes a function which can be passed to ClassifiedRetriever.run_dynesty
    to calculate ln likelihoods

    Parameters
    ----------
    y : array_like, shape (X, )
        The y co-ordinates of the data you will be fitting, assuming y = f(x)
    yerr : array_like, shape (X, )
        The 1-sigma uncertainty on each y value
    model : function
        A function which takes a set of parameters and returns model y values
        evaluated at each x value. Must have signature `model(params)` where
        `params` is an array with shape (n_variables, )

    Returns
    -------
    lnprob : function
        The ln likelihood function, which has signature `lnprob(params)` where
        `params` is an array with shape (n_variables, ).

    '''

    def lnprob(params):
        '''
        Calculates the ln likelihood of a set of parameters. Created using
        `make_nested_lnprob`

        Parameters
        ----------
        params : array_like, shape (n_variables)
            The parameters to evaluate the ln likelihood of

        Returns
        -------
        lnlike : float
            The ln likelihood of the parameters.
        '''
        model_y = model(params)

        chi2 = sum((model_y - y)**2 / yerr**2)

        return - chi2

    return lnprob
