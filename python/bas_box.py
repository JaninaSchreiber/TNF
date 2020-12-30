import threading
import time
import math
import csv
import json
import sys
import time
import pickle
import metrics
import psutil
import os
import random

from scipy import stats

import pandas as pd
import numpy as np
from random import randrange,choices, triangular
import ga_utils
import eqpy

def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start

class bas_algorithm:
    def __init__(self, m_max, data):
        self.m_max = m_max
        self.data = data
        self.f_x = 99999999999.
        self.f_x_last = 99999999999.
        self.dfloat = data[data['lower_bound'].apply(lambda x: isinstance(x, float))]
        self.dint = data[data['lower_bound'].apply(lambda x: isinstance(x, int))]
        self.tf = 0
        self.tb = 0

    def update_float_int(self):
        self.dfloat = self.data[self.data['lower_bound'].apply(lambda x: isinstance(x, float))]
        self.dint = self.data[self.data['lower_bound'].apply(lambda x: isinstance(x, int))]

    def set_x(self):
        # set a random parameter in the domain.
        dfloat = self.dfloat.apply(lambda row:randrange_float(row["lower_bound"],
                                                              row["upper_bound"],
                                                              row["step_size"]),
                                   axis=1)
        dint = self.dint.apply(lambda row:random.randrange(row["lower_bound"],
                                                           row["upper_bound"]),
                               axis=1)
        self.data['x_current'] = dfloat.append(dint)
        self.update_float_int()
                

        
    def generate_path_length(self):
        # float
        self.update_float_int()
        self.data["muhf"] = np.where(self.data["direction"] > 0,
                                     (self.data["upper_bound"] - self.data["x_current"])/
                                     self.data["step_size"],
                                     (self.data["x_current"]- self.data["lower_bound"])/
                                     self.data["step_size"])
        
        self.data["muhb"] = np.where(self.data["direction"] > 0,
                                    (self.data["x_current"] - self.data["lower_bound"])/
                                     self.data["step_size"],
                                     (self.data["upper_bound"] - self.data["x_current"])/
                                      self.data["step_size"])
        muhf =  self.data["muhf"].min()                                     
        muhb =  self.data["muhb"].min()
        i_f = self.data[self.data.muhf == muhf]
        i_f = np.where(self.data.index==i_f.index[0])[0]
        i_b = self.data[self.data.muhb == muhb]
        i_b = np.where(self.data.index==i_b.index[0])[0]
        self.tf = len(self.data.x_current) * muhf + i_f - 1
        self.tb = len(self.data.x_current) * muhb + i_b - 1
        self.update_float_int()


    def generate_z(self, permutation):
        p = 0
        self.update_float_int()
        dfloat_step = self.dfloat['step_size'].apply(lambda elem: randrange_float(0, elem, 10**(-10)) ) 
        #[randrange_float(0, elem, 10**(-10)) for elem in self.dfloat['step_size']]
        dint_step = self.dint['step_size'].apply(lambda elem: random.randrange(0, elem)) 
        #[random.randrange(0, elem) for elem in self.dint['step_size']]
        self.data["this_step_size"] = dfloat_step.append(dint_step)
        while(p==0):
            p = random.randint(-int(self.tb[0]), int(self.tf[0]))
        print("p: {}".format(p))
        # determine number of cycles
        if p > 0:
            cycles = np.floor(p/len(self.data["x_current"]))
            i_f = p - cycles * len(self.data["x_current"])
            self.data["x_proposal"] = self.data["x_current"] + self.data["direction"] * (self.data["this_step_size"] * cycles)
            for i in range(int(i_f)):
                self.data["x_proposal"][permutation[i]] = (self.data["x_proposal"][permutation[i]] +
                                           self.data["direction"][permutation[i]] *
                                           self.data["this_step_size"][permutation[i]])
        elif p <= 0:
            cycles = np.floor(-p/len(self.data["x_current"]))
            i_b = -p - cycles * len(self.data["x_current"])
            permutation.reverse()
            self.data["x_proposal"] = self.data["x_current"] - self.data["direction"] * self.data["this_step_size"] * cycles
            for i in range(int(i_b)-1):
                self.data["x_proposal"][permutation[i]]  = (self.data["x_proposal"][permutation[i]] -
                                           self.data["direction"][permutation[i]] *
                                           self.data["this_step_size"][permutation[i]])
        self.update_float_int()
        #print("x_proposal {}".format(self.data.x_proposal))
    
    
    def hit_and_run(self):
        # step 1 - generate D
        self.data["direction"] = choices([-1,1], k=len(self.data))

        # step 2 - generate random permutation
        permutation = random.sample(range(len(self.data)), len(self.data))

        # step 3.0 - BoxWalk 
        self.generate_path_length()
        self.generate_z(permutation)
        

        
    def set_x_current(self):
        self.data.x_current = self.data.x_proposal

        
    def set_f_x(self, f_x):
        self.f_x = f_x

        
    def set_f_x_last(self, f_x_last):
        self.f_x_last = f_x_last

    def set_radius(self, a=0.5):
        self.data["radius"] = self.data["radius"] * a
        self.data["radius"] = self.data["radius"].abs()
        self.radius = True
        
    def set_x_current_to_min(self, met):
        # print("x_current {}".format(self.data["x_current"]))
        self.data["x_current"] = pd.DataFrame(met.results[met.results.score==met.results.score.min()]).transpose()
        
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
    #print(pops)
    if not pops:
        return []
    eqpy.OUT_put(str(pops).replace("\'", "\""))
    result = eqpy.IN_get()
    split_result = result.split(',')
    # returns the objective function value
    return [(float(x),) for x in split_result]


def prepare_x(bas_params, x_param):
    x = dict(zip(bas_params.index, eval(x_param)))
    x = str(x).replace("'", "\"")
    return x



def improving_hit_and_run(sim):
    """
    move to the proposed point 
    if the obj. function value improved
    
    Arguments:
    sim: object of class bas_algorithm
    """
    if(sim.f_x < sim.f_x_last):
        sim.set_x_current()
        sim.f_x_last = sim.f_x
    elif(sim.f_x >= sim.f_x_last):
        pass
        #print("The point did not improve")
    return sim


def hide_and_seek(sim, m, t, met, cooling_schedule='geometric'):
    """
    decide whether or not to move to the proposed 
    point depending on the state
    
    Arguments:
    sim: object of class bas_algorithm
    m: (int) iteration number
    t: (float) value of the current cooling
    met: object of class metrics
    cooling_schedule: (string)
    """
    print("xproposal {}".format(sim.data["x_proposal"]))
    print("xcurrent {}".format(sim.data["x_current"]))
    if(sim.f_x < sim.f_x_last):
        sim.set_x_current()
        sim.set_f_x_last(sim.f_x)
        if(cooling_schedule == 'adaptive'):
            degree_of_freedom = sim.data.x_current.size
            t = 2*(sim.f_x_last - 0)/stats.chi2.ppf(0.95, degree_of_freedom)
    else:
        p_accept = min(1, float(np.exp((sim.f_x_last - sim.f_x)/t)))
        #print("p_accept: {}".format(p_accept))
        set_x = int(np.random.choice([0, 1], 1, p=[1-p_accept, p_accept])) #x , x_proposal
        if set_x == 0:
            pass
        elif set_x == 1:
            sim.set_x_current()
            sim.set_f_x_last(sim.f_x)
    if(cooling_schedule == 'geometric'):
        t = 0.99**m
    return sim, t

def randomsearch(sim):
    sim.set_x()
    return sim, 

        
def run():

    # collect time and memory metrics 
    ts_all = time.time()
    import psutil
    process = psutil.Process(os.getpid())
    cpu_time = process.cpu_times()[0] / float(2 ** 20)
    mem = process.memory_full_info()

    ### set input parameters                                                   
    # number of iterations  
    eqpy.OUT_put("Params")
    params = eqpy.IN_get()
    printf("Parameters: {}".format(params))
    (num_iter, bas_parameters_file, strategy, cooling_schedule)  = eval('{}'.format(params))

    print("bas file {}".format(bas_parameters_file))        
    print(bas_parameters_file)
    bas_params = ga_utils.create_parameters(bas_parameters_file)
    bas_params = transform_param(bas_params)
    # load possible tuples                                                    
    # instantiate                           
    sim = bas_algorithm(m_max=num_iter, data = bas_params)
    # set initial x
    sim.set_x()

    replications = 1

    # columns of parameters and results
    cols = list(sim.data.x_current.to_dict().keys()) + ["score"]

    # initialize a script "metrics" in which all metrics are calculated
    met = metrics.metrics(cols, replications, mem)
    
    ### select a cooling schedule for hide-and-seek
    # can be "adaptive" or "geometric"
    if strategy=="hide-and-seek":
        cooling_schedule = "adaptive"
    else:
        cooling_schedule = ""
    m = 0
    t = 1
    while(m < num_iter):
        #print("--------- this is m {}-----------------------".format(m))
        # generate x_proposal in hit-and-run
        sim.hit_and_run()
        # start time of the simulation
        t_send_data = time.time()
        # evaluate the objective function by running the simulation
        f_x = queue_map(0, sim.data.x_proposal.to_dict())[0][0]
        sim.set_f_x(f_x)
        # end time of the simulation
        t_receive_data = time.time()
        # update the metrics with the simulation time and results
        met.calculate_simulation_time(t_send_data, t_receive_data)
        met.append_score(sim.f_x, sim.data.x_proposal.to_dict(), cols)
        # choose strategy for neighbour acceptance
        if(strategy == "improving_hit-and-run"):
            sim = improving_hit_and_run(sim)
        elif(strategy == "hide-and-seek"):
            sim, t = hide_and_seek(sim, m, t, met,cooling_schedule)
        elif(strategy == "randomsearch"):
            print('going to rs')
            sim = randomsearch(sim)
        else:
            print("going here")
            break
        # increment m
        m += 1
    # calculate total time of the simulation
    te_all = time.time()
    # name of file where to store the results
    pathname = strategy + str(num_iter) + cooling_schedule
    # calculate and store results
    metrics.summarize_results(met, ts_all, te_all)

    
