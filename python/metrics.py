import numpy as np
import pandas as pd
import os



class metrics:

    def __init__(self, cols, replications, mem):
        self.scores = []
        self.replications = replications
        self.results = pd.DataFrame(columns = cols)
        self.mem = mem
        self.hall_of_fame = pd.DataFrame()
        self.len_simulation = 0


    def append_score(self, score, x_current, cols):
        # saves the values of f_x
        self.scores.append(score)
        # saves the values with
        mylist = list(x_current.values())#[item for sublist in x_current.values() for item in sublist]
        self.results = self.results.append(pd.DataFrame([mylist + [score]],
                                                        columns = cols))
        instance_directory = os.environ["TURBINE_OUTPUT"]
        directory = instance_directory + "/mymetrics/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.results.to_csv(directory + "myresults.csv")

    def get_nr_of_evaluations(self):
        return self.replications * len(self.scores)

    def get_nr_of_hits(self):
        return len(self.hall_of_fame)

    def get_optimum(self):
        opt = self.results[self.results["score"] == self.results["score"][0].min()]
        return opt

    def get_memory_use(self):
        return self.mem

    def get_accuracy():
        import metrics
        my_dict = {"first":[2], "second": [3], "third":[3]}
        cols = list(my_dict.keys())+["score"]
        met = metrics.metrics(cols)
        met.append_score(4, my_dict, cols)
        pass

    def calculate_simulation_time(self, t_send_data, t_receive_data):
        self.len_simulation = self.len_simulation + (t_receive_data - t_send_data)

    def calculate_alg_time(self, ts_all, te_all):
        """ return the runtime of the algorithm without the simulation"""
        return te_all - (self.len_simulation + ts_all)

    def calculate_total_time(self, ts_all, te_all):
        """ return the runtime of the algorithm with the simulation"""
        return te_all - ts_all

    def calculate_robustness(self, feasible_point, instance_directory):
        # calculate how many points lie in the feasible region
        self.hall_of_fame = self.results[self.results["score"] <= feasible_point]
        # write hall of fame in <experiment>/mymetrics
        directory = instance_directory + "/mymetrics/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.hall_of_fame.to_csv(directory + "hall_of_fame.csv")
        #list(filter(lambda x: (x < feasible_point), self.scores))
        
    def calculate_reproducibility():
        pass

    def write_results(self, directory):
        self.results.to_csv(directory + "results.csv")
                    

    
def summarize_results(met, ts_all, te_all):
    instance_dir = os.environ["TURBINE_OUTPUT"]
    optimum = met.get_optimum().to_dict()
    total_time = met.calculate_total_time(ts_all, te_all)
    alg_time = met.calculate_alg_time(ts_all, te_all)
    nr_of_hits = met.get_nr_of_hits()
    nr_of_evaluations = met.get_nr_of_evaluations()
    mem = met.get_memory_use()

    # create result dict
    result_dict = {"experiment": [instance_dir.split("/")[-1]],
                   "total_time": [total_time],
                   "alg_time": [alg_time],
                   "nr_of_hits": [nr_of_hits],
                   "nr_of_evaluations": [nr_of_evaluations],
                   "rss": [mem.rss],
                   "uss": [mem.uss]}

    df = pd.DataFrame.from_dict(result_dict)
    df = pd.concat([df, pd.DataFrame(optimum)], axis=1)
    print(df)

    directory = instance_dir + "/mymetrics/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(directory + "summary.csv")
    met.write_results(directory)
    met.calculate_robustness(90, instance_dir)

