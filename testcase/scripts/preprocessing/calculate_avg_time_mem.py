import os
import pandas as pd
import matplotlib.pyplot as plt

"""
Each algorithm is run several times on a problem. 
For the testcases each algorithm is run N_r=30 times.
For each algorithm a folder with the results is created. 
Select testcase in PATH
"""

#### get a summary of the memory
PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/EMEWS-scripts/testcase/sixhumpcamelbacktime/"

# choose the algorithm: "hide-and-seek", "randomseach", "improving hit-and-run"
algorithm = "randomsearch"

#choose the number of evaluation
evaluations = "100"

# choose the cooling schedule "geometric" or "adaptive" of HNS,
# "" otherwise
cooling_schedule = ""

# path to result folder 
file = "{}{}{}/".format(algorithm, evaluations, cooling_schedule)

# open the subfolders in "file"
subdirs = [x[0] for x in os.walk(PATH + file)]
print(subdirs)

# create empty dataframe
columns = ["test", "best result", "evaluations"]
data = pd.DataFrame()

# iterate over every directory in subdirs and extract the f_best value
for dir in subdirs:
    if dir == PATH + file:
        pass
    else:
        df = pd.read_csv(dir + "/summary.csv")
        test = dir.split("/")[-1]
        timedata = pd.DataFrame({"test": [test],
                                 "rss": df["rss"],
                                 "uss": df["uss"]})
        data = data.append(timedata, ignore_index=True)
        
data.to_csv(PATH + file + "summary_of_memory.csv")




#### get a summary of the time
# returns the immediate subdirectories of PATH
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# ask for immediate subdirectories
a = get_immediate_subdirectories(PATH)


data_time = pd.DataFrame()

# calculate the mean result of each algorithm
for directory in a:
    df = pd.read_csv(PATH +  directory + "/summary_of_memory.csv")
    uss = df["uss"].mean()
    uss_var = df["uss"].var()
    rss = df["rss"].mean()
    rss_var = df["rss"].var()
    evaluations =  int("".join(filter(str.isdigit, directory)))
    algorithm = directory.split(str(evaluations))
    algorithm = " ".join(algorithm)
    data_time = data_time.append({"algorithm": algorithm,
                                  "evaluations": evaluations,
                                  "uss": uss,
                                  "rss":rss,
                                  "uss_var": uss_var,
                                  "rss_var":rss_var}, ignore_index=True)



# iterate over every directory in subdirs and extract the f_best value
for dir in subdirs:
    if dir == PATH + file:
        pass
    else:
        df = pd.read_csv(dir + "/summary.csv")
        test = dir.split("/")[-1]
        timedata = pd.DataFrame({"test": [test],
                                 "total_time": df["total_time"],
                                 "alg_time": df["alg_time"]})
        data = data.append(timedata, ignore_index=True)
        
data.to_csv(PATH + file + "summary_of_time.csv")

# create empty dataframe
data_time = pd.DataFrame()

# calculate the mean result of each algorithm
for directory in a:
    df = pd.read_csv(PATH +  directory + "/summary_of_memory.csv")
    totaltime = df["total_time"].mean()
    totaltime = str(datetime.timedelta(minutes=totaltime))
    algtime = df["alg_time"].mean()
    algtime = str(datetime.timedelta(minutes=algtime))
    evaluations =  int("".join(filter(str.isdigit, directory)))
    algorithm = directory.split(str(evaluations))
    algorithm = " ".join(algorithm)
    data_time = data_time.append({"algorithm": algorithm,
                                  "evalutations": evaluations,
                                  "totaltime": totaltime,
                                  "algtime": algtime}, ignore_index=True)

# set index
data_time = data_mean.set_index('algorithm', drop=True)

# store dataframe
data_time.to_csv(PATH + "mean_results.csv") 


#calculate the variance of results of each algorithm
data_var = pd.DataFrame()
for directory in a:
    df = pd.read_csv(PATH +  directory + "/summary_of_results.csv")
    df = df.groupby("evaluations").var()
    df = df.drop("Unnamed: 0", axis=1)x
    df = df.transpose()
    df["algorithm"] = directory
    data_var = data_var.append(df)

# set index
data_var  = data_var.set_index('algorithm', drop=True)
    
# store dataframe
data_var.to_csv(PATH + "var_results.csv")    
