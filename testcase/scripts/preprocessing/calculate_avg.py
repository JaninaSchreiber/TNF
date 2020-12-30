import os
import pandas as pd
import matplotlib.pyplot as plt

"""
Each algorithm is run several times on a problem. 
For the testcases each algorithm is replicated N_r=30 times.
For each replicate a folder with name "test1", "test2",..., "test30" 
is created. 

In PART1 of this script a summary of the replicates is created 
and stored in "summary_of_results.csv"
- stored in folder of experiment i.e. PATH + "hide-and-seek100geomtric"
- contains a summary of the replicates for i.e. HNS with a geometric 
  cooling schedule and a maximum of 100 iterations 


In PART2 the "summary_of_results.csv" files are summarized again
- stored in OUTPUT_PATH
- contains summary of the mean results of all algorithms
"""

PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/EMEWS-scripts/testcase/sixhumpcamelbacktime/"
OUTPUT_PATH =  "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/EMEWS-scripts/testcase/scripts/data/"

########## PART1 ##########
# choose the algorithm: "hide-and-seek", "randomseach", "improving hit-and-run"
algorithm = "hide-and-seek"

#choose the number of evaluation
evaluations = "100"

# choose the cooling schedule "geometric" or "adaptive" of HNS,
# "" otherwise
cooling_schedule = "geometric"

# path to result folder 
file = "{}{}{}/".format(algorithm, evaluations, cooling_schedule)

# open the subfolders in "file"
subdirs = [x[0] for x in os.walk(PATH + file)]
print(subdirs)

# create empty dataframe
columns = ["test", "best result", "viable result", "evaluations"]
data = pd.DataFrame()

# iterate over every directory in subdirs and extract the f_best value
for dir in subdirs:
    if dir == PATH + file:
        pass
    else:
        df = pd.read_csv(dir + "/results.csv")
        result_list= [df.iloc[:25]['score'].min(),
                      df.iloc[:50]['score'].min(),
                      df.iloc[:100]['score'].min()]
                      #df.iloc[:2500]['score'].min(),
                      #df.iloc[:5000]['score'].min(),
                      #df.iloc[:10000]['score'].min()]
        viable_list= [len(df.iloc[:25][df.iloc[:25]['score']<0]),
                      len(df.iloc[:50][df.iloc[:50]['score']<0]),
                      len(df.iloc[:100][df.iloc[:100]['score']<0])]
                      #len(df[df.iloc[:2500]['score']<0]),
                      #len(df[df.iloc[:5000]['score']<0]),
                      #len(df[df.iloc[:10000]['score']<0])]
        #evaluations = [300, 625, 1250, 2500, 5000, 10000]
        evaluations = [25, 50, 100]
        test = dir.split("/")[-1]
        curr_data = pd.DataFrame({"test":[test] * 3,
                                  "best result": result_list,
                                  "viable result": viable_list,
                                  "evaluations": evaluations})
        data = data.append(curr_data, ignore_index=True)

# create a summary file for the specified algorithm        
data.to_csv(PATH + file + "summary_of_results.csv")

# for the curious: see the results of the algorithms ;D
mean = data.groupby("evaluations").mean()
var = data.groupby('evaluation').var()


########## PART1 ##########

#Read the summary files of each algorithm and summarize once more

# returns the immediate subdirectories of PATH
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# ask for immediate subdirectories
a = get_immediate_subdirectories(PATH)


# create empty dataframe
data_mean = pd.DataFrame()

# calculate the mean result of each algorithm
for directory in a:
    df = pd.read_csv(PATH +  directory + "/summary_of_results.csv")
    df = df.groupby("evaluations").mean()
    df = df.drop("Unnamed: 0", axis=1)
    df = df.transpose()
    df["algorithm"] = directory
    data_mean = data_mean.append(df)

# set index
data_mean = data_mean.set_index('algorithm', drop=True)

# store dataframe
data_mean.to_csv(PATH + "grie_mean.csv")


#calculate the variance of results of each algorithm
data_var = pd.DataFrame()
for directory in a:
    df = pd.read_csv(PATH +  directory + "/summary_of_results.csv")
    df = df.groupby("evaluations").std()
    df = df.drop("Unnamed: 0", axis=1)
    df = df.transpose()
    df["algorithm"] = directory
    data_var = data_var.append(df)

# set index
data_var  = data_var.set_index('algorithm', drop=True)
    
# store dataframe
data_var.to_csv(PATH + "grie_var.csv")    


