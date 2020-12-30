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
import random

import pandas as pd
from random import randrange,choices, triangular

import ga_utils
import eqpy
import metrics

PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/EMEWS-scripts/swift/"

def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start


class random_search:
    
    def __init__(self, m_max, data):
        self.m_max = m_max
        self.data = data
        self.f_x = 99999999999.0
        self.f_x_last = 99999999999.0
        self.dint = data[data['lower_bound'].apply(lambda x: isinstance(x, int))]
        self.dfloat = data[data['lower_bound'].apply(lambda x: isinstance(x, float))]
        self.tf = 0
        self.tb = 0

    def update_float_int(self):
        self.dfloat = self.data[self.data['lower_bound'].apply(lambda x: isinstance(x, float))]
        self.dint = self.data[self.data['lower_bound'].apply(lambda x: isinstance(x, int))]

    def set_x(self):
        self.update_float_int()
        dfloat = self.dfloat.apply(lambda row:randrange_float(row["lower_bound"],
                                                              row["upper_bound"],
                                                              row["step_size"]),
                                   axis=1)
        dint = self.dint.apply(lambda row:random.randrange(row["lower_bound"],
                                                           row["upper_bound"]),
                               axis=1)
        self.data['x_current'] = dfloat.append(dint)


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
    print("We are in queuemap {}".format(pops))
    #if not pops:
    #    return []
    eqpy.OUT_put(str(pops).replace("\'", "\""))
    result = eqpy.IN_get()
    split_result = result.split(',')
    # returns the objective function value
    return [(float(x),) for x in split_result]



def run():
    ts_all = time.time()
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_full_info()
    
    eqpy.OUT_put("Params")
    params = eqpy.IN_get()
    print("Parameters: {}".format(params))
    (num_iterations, rs_parameters_file) = eval('{}'.format(params))

    # TODO: code a replacement here!
    rs_params = ga_utils.create_parameters(rs_parameters_file.replace('\'', '\"'))
    rs_params = transform_param(rs_params)

    replications = 3
    rs = random_search( num_iterations,rs_params)
    rs.set_x()
    cols = list(rs.data.x_current.to_dict().keys()) + ["score"]
    met = metrics.metrics(cols, replications, mem)
    first_hit = "a"
    number_of_hits = 0
    m = 0
    x_best = 100
    f_x_best = 9999999999.
    while(m < num_iterations):
        rs.set_x()
        t_send_data = time.time()
        f_x = queue_map(0, rs.data.x_current.to_dict())[0][0]
        t_receive_data = time.time()
        print("success_x: {}".format(f_x))
        #f_x = abs(success_x[0][0])
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
    metrics.summarize_results(met, ts_all, te_all)



    
