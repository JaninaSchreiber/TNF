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
import pudb

from scipy import stats

import pandas as pd
import numpy as np
import testfunction as tf
from random import randrange,choices, triangular

#import eqpy, ga_utils

PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/EMEWS-scripts/experiments/experiment2"

def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start

class bas_algorithm:
    def __init__(self, m_max, data):
        self.m_max = m_max
        self.data = data
        self.f_x = 99999999999.
        self.f_x_last = 99999999999.
        self.radius = False
        self.tf = 0
        self.tb = 0
        
    def set_x(self):            
        self.data["x_current"] = self.data.apply(lambda
                                             row:randrange_float(row["lower_bound"],
                                                            row["upper_bound"]+
                                                            row["step_size"],
                                                            row["step_size"]),
                                       axis=1)
        self.data["radius"] = (self.data.upper_bound - self.data.lower_bound)/self.data.step_size

        
    def generate_path_length(self):
        self.data["muhf"] = np.where(self.data["direction"] > 0,
                                     np.round((self.data["upper_bound"] - self.data["x_current"])/
                                     self.data["step_size"]),
                                     np.round((self.data["x_current"]- self.data["lower_bound"])/
                                     self.data["step_size"]))
        
        self.data["muhb"] = np.where(self.data["direction"] > 0,
                                    np.round((self.data["x_current"] - self.data["lower_bound"])/
                                     self.data["step_size"]),
                                     np.round((self.data["upper_bound"] - self.data["x_current"])/
                                      self.data["step_size"]))
        #print("muhf {}".format(self.data["muhf"]))
        #print("muhb {}".format(self.data["muhb"]))
        muhf =  self.data["muhf"].min()                                     
        muhb =  self.data["muhb"].min()
        i_f = self.data[self.data.muhf == muhf]
        i_f = np.where(self.data.index==i_f.index[0])[0]
        i_b = self.data[self.data.muhb == muhb]
        i_b = np.where(self.data.index==i_b.index[0])[0]
        self.tf = len(self.data.x_current) * muhf + i_f - 1
        self.tb = len(self.data.x_current) * muhb + i_b - 1
        #print("-tb {}".format(-self.tb))
        #print("tf {}".format(self.tf))


    def generate_z(self, permutation):
        p = 0
        self.data["this_step_size"] = [randrange_float(0, elem, 10**(-10)) for elem in self.data["step_size"]]
        while(p==0):
            p = random.randint(-self.tb[0], self.tf[0])
        #print("p: {}".format(p))
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
            
        print("x_proposal {}".format(self.data.x_proposal))
    
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
        print("x_current {}".format(self.data["x_current"]))
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
    print(pops)
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
    print("f(x) {}".format(sim.f_x))
    print("f(xlast) {}".format(sim.f_x_last))
    if(sim.f_x < sim.f_x_last):
        sim.set_x_current()
        sim.f_x_last = sim.f_x
    elif(sim.f_x >= sim.f_x_last):
        print("The point did not improve")
    return sim


def hide_and_seek(sim, m, t, met, n_t=50, cooling_schedule='geometric'):
    print("f(x) {}".format(sim.f_x))
    print("f(xlast) {}".format(sim.f_x_last))
    print("x_current {}".format(sim.data.x_current))
    print("x_proposal {}".format(sim.data.x_proposal))
    if(sim.f_x < sim.f_x_last):
        print("going in fx < fxlast")
        sim.set_x_current()
        sim.set_f_x_last(sim.f_x)        
    else:
        p_accept = min(1, float(np.exp((sim.f_x_last - sim.f_x)/t)))
        print("p_accept: {}".format(p_accept))
        set_x = int(np.random.choice([0, 1], 1, p=[1-p_accept, p_accept])) #x , x_proposal
        if set_x == 0:
            pass
        elif set_x == 1:
            sim.set_x_current()
            sim.set_f_x_last(sim.f_x)
    if(cooling_schedule == 'geometric'):
        t = 0.99**m
    elif(cooling_schedule == 'adaptive'):
        degree_of_freedom = sim.data.x_current.size
        #print("degree of freedom {}".format(degree_of_freedom))
        t = 2*(sim.f_x - 0)/stats.chi2.ppf(0.95, degree_of_freedom)
        print("t {}".format(t))
    if (m % n_t == 0):
        print("results \n {}".format(met.results[met.results.score==met.results.score.min()]))
        sim.set_x_current_to_min(met)
        print("x_changed {}".format(sim.data["x_current"]))
    return sim, t


        
if __name__ == "__main__":
    
    ts_all = time.time()
    import psutil
    process = psutil.Process(os.getpid())
    #mem = process.memory_info()[0] / float(2 ** 20)
    cpu_time = process.cpu_times()[0] / float(2 ** 20)
    mem = process.memory_full_info()
    print("mem: \n {}".format(mem))

    num_iter = 1000
    bas_params = pd.read_csv("sixhumpcamelback.csv")
    bas_params = bas_params.set_index("parameter") 
    
    # load possible tuples
    # instantiate 
    sim = bas_algorithm(m_max=num_iter, data = bas_params)
    # set initial x
    sim.set_x()
    print("set x_current {}".format(sim.data.x_current))
    replications = 1
    cols = list(sim.data.x_current.to_dict().keys()) + ["score"]
    met = metrics.metrics(cols, replications, mem)

    strategy="hitandrun"
    cooling_schedule = ""
    number_of_hits = 0
    m = 0
    x_best = 100
    t = 10
    f_x_last = 9999999999.
    while(m < num_iter):
        print("--------- this is m {}-----------------------".format(m))
        # generate x_proposal in hit-and-run
        sim.hit_and_run()
        #sim.set_x_current()
        t_send_data = time.time()
        #met.calculate_simulation_time(t_send_data, t_receive_data)
        met.append_score(sim.f_x, sim.data.x_proposal.to_dict(), cols)
        m += 1
    te_all = time.time()
    pathname = strategy + str(num_iter) + cooling_schedule
    metrics.summarize_results(met, ts_all, te_all, pathname)
 
