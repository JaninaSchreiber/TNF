import numpy as np

def six_hump_camel_back(x1, x2):
    '''
    computes the six-hump camelback problem.
    
    Arguments:
    x1: (float)
    x2: (float)
    
    Returns: 
    f(x1, x2): (float)
    '''
    f1 = (4 - 2.1 * x1**2 + (x1**4)/3.) * x1**2
    f2 = x1 * x2 + (-4 + 4 *  x2**2) *x2**2
    return f1 + f2


def griewank_problem(parameter_list):
    '''
    computes the griewank problem.
    
    Arguments:
    parameter_list: (list of floats)
    
    Returns: 
    f(parameter_list): (float)
    '''
    f1 = 1 + 1/4000 * np.sum(np.square(parameter_list))
    f2 = 1
    for i in range(len(parameter_list)):
        f2 = f2 * np.cos(parameter_list[i]/(i+1))
    return f1 - f2
