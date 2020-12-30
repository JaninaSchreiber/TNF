import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

# set PATH of data location
PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/EMEWS-scripts/testcase/griewank/"

# read mean and variance
grie = pd.read_csv(PATH + "grie_mean.csv")
grie2 = pd.read_csv(PATH + "grie_var.csv")

# set algorithm to index
grie2 = grie2.set_index('algorithm')
grie = grie.set_index('algorithm')

# transform string to numeric
grie.columns = [ast.literal_eval(i) for i in grie.columns]
grie2.columns = [ast.literal_eval(i) for i in grie2.columns]

# set font size
plt.rcParams.update({'font.size': 16})

# set distance between markers and marker size
n = 5
markersize=10

# generate column list
column_list = np.array([100, 200, 300, 400, 500, 600])

# generate figure
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1)

# plot RS
plt.errorbar(column_list + 2.5*n,
             grie.iloc[2],
             grie2.iloc[2],
             label="PRS", linestyle='None',
             marker='^', markersize=markersize)

# plot HNS adaptive
plt.errorbar(column_list + 0.75*n,
             grie.iloc[0],
             grie2.iloc[0],
             label="HNS adaptive", linestyle='None',
             marker='.', markersize=markersize, c='g')

# plot IHR
plt.errorbar(column_list - 0.75*n,
             grie.iloc[1],
             grie2.iloc[1],
             label="IHR", linestyle='None',
             marker='<', markersize=markersize, c='orange')

# plot HNS geometric
plt.errorbar(column_list - 2.5*n,
             grie.iloc[3],
             grie2.iloc[3],
             label="HNS geometric", linestyle='None',
             marker='*', markersize=markersize ,c='crimson')

# logarithmiize y axis
ax.set_yscale("log")

#generate legend
plt.legend()

# set tick labels
ax.set_xticklabels([0,300,625,1250,2500,5000,10000])

# set label for axis
plt.xlabel("Objective function evaluations " + r'$F_e$')
plt.ylabel(r'$\overline{f}_{best})$')

# set title
plt.title("Griewank problem")

# enable grid
plt.grid(axis='y')

plt.savefig(PATH +"griewank_grid_c.png")
plt.show()


