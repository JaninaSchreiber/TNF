import threading
import time
import math
import csv
import json
import sys
import time
import pickle
import time
import os
import pudb

import stats as sts


import pandas as pd
import loader
import testfunction as tf
from random import randrange,choices, triangular

import random
import eqpy
import metrics

PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/EMEWS-scripts/swift/"

def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start

class random_search:
    def __init__(self, data,m_max):
        self.m_max = m_max # maximum number of iterations
        self.data = data  
        self.f_x = 99999999999.
        self.f_x_last = 99999999999.
        self.dfloat = data[data['step_size'].apply(lambda x: isinstance(x, float))]
        self.dint = data[data['step_size'].apply(lambda x: isinstance(x, int))]

        
    def update_float_int(self):
        # subdivides data in two dataframes: for integer and float
        self.dfloat = self.data[self.data['step_size'].apply(lambda x: isinstance(x, float))]
        self.dint = self.data[self.data['step_size'].apply(lambda x: isinstance(x, int))]

        
    def set_x(self):
        # initialize x
        # for floating point
        dfloat = self.dfloat.apply(lambda
                                   row:randrange_float(row["lower_bound"],
                                                       row["upper_bound"],
                                                       row["step_size"]),
                                   axis=1)
        # for integer

        dint = self.dint.apply(lambda
                               row:random.randrange(row["lower_bound"],
                                                    row["upper_bound"]),
                               axis=1)
        # update xcurrent
        if dfloat.empty:
            self.data['x_current'] = dint
        elif dint.empty:
            self.data['x_current'] = dfloat
        else:
            self.data['x_current'] = dfloat.append(dint)
        self.update_float_int()

        
def transform_param(o_param):
    columns = ["parameter", "lower_bound","upper_bound", "step_size"]
    df = pd.DataFrame(columns=columns)
    for param in range(len(o_param)):
        row = pd.DataFrame({
            "parameter": [o_param[param].name],
            "lower_bound": [o_param[param].lower],
            "upper_bound": [o_param[param].upper],
            "step_size": [o_param[param].sigma]})
        df = df.append(row)
    df = df.set_index("parameter")
    return df

def obj_func():
    return 0

def printf(val):
    print(val)
    sys.stdout.flush()

        
def queue_map(obj_func, pops):
    # Note that the obj_func is not used
    # sending data that looks like:
    # [[a,b,c,d],[e,f,g,h],...]
    #print("We are in queuemap {}".format(pops))
    #if not pops:
    #    return []
    eqpy.OUT_put(str(pops).replace("\'", "\""))
    result = eqpy.IN_get()
    split_result = result.split(',')
    # returns the objective function value
    return [(float(x),) for x in split_result]


if __name__=="__main__":
    """
    Use Random Search to explore the search space

    Arguments:
    problem : (string) either 'shcb' or 'griewank' 
    num_iter: (int) maximum number of iterations
    """
    try:
        problem = str(sys.argv[1])
        num_iterations = int(sys.argv[2])  # integer
    except IndexError:
        print('missing argument')
        print('input must look like >python bas_box_int.py shcb 5 hide-and-seek geometric<')
    print("problem: {}".format(problem))
    ts_all = time.time()
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_full_info()

    replications = 1
    if(problem=='griewank'):
        rs_params = pd.read_csv("../data/griewank_input.csv")
    elif(problem=='shcb'):
        rs_params = pd.read_csv("../data/shcb_input.csv")
    else:
        sys.exit("This problem does not exits.\n" +
              "Please choose 'shcb' or 'griewank'")
        
    rs_params = rs_params.set_index("parameter")
    
    rs = random_search(rs_params, num_iterations)
    rs.set_x()
    cols = list(rs.data.x_current.to_dict().keys())+["score"]
    met = metrics.metrics(cols, replications, mem)
    first_hit = "a"
    number_of_hits = 0
    m = 0
    x_best = 100
    f_x_best = 9999999999.
    while(m < num_iterations):
        rs.set_x()
        t_send_data = time.time()
        if(problem=='griewank'):
            f_x = tf.griewank_problem(rs.data.x_current.to_list())
        else:
            f_x = tf.six_hump_camel_back(rs.data.x_current["x1"], rs.data.x_current["x2"])
        t_receive_data = time.time()
        met.calculate_simulation_time(t_send_data, t_receive_data)
        met.append_score(f_x, rs.data.x_current.to_dict(), cols)
        if(f_x < f_x_best):
            f_x_best = f_x
            x_best =  rs.data.x_current.to_dict()
            number_of_hits = 1
        elif(f_x == f_x_best):
            number_of_hits += 1
        m += 1
    te_all = time.time()
    pathname = "randomsearch{}".format(num_iterations)
    metrics.summarize_results(met, ts_all, te_all, pathname)



    
