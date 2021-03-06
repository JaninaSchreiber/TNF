import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/EMEWS-scripts/testcase/sixhumpcamelback2/"

# read data
df = pd.read_csv(PATH + "sixhumpcamelback_summary.csv")

# create a dataframe for each algorithm
random = df[df["algorithm"]=="randomsearch"]
hasa =  df[df["algorithm"]=="Hide-and-seek adaptive"]
hasg =  df[df["algorithm"]=="Hide-and-seek geometric"]
ihr = df[df["algorithm"]=="improving hit-and-run"]

# set distance between markers and marker size
n = 0.3
markersize=7

# set font size
plt.rcParams.update({'font.size': 16})

# generate column list
column_list = np.array([10, 20, 30])

# generate figure
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1)

# plot RS
random = random.sort_values('nr. evaluations')
plt.errorbar(column_list - n,
             random["minimum mean"],
             random["minimum var"],
             label="Pure Random Search", linestyle='None',
             marker='^', markersize=markersize)

# plot HNS adaptive
hasa = hasa.sort_values('nr. evaluations')
plt.errorbar(column_list + n,
             hasa["minimum mean"],
             hasa["minimum var"],
             label="Hide-and-Seek adaptive", linestyle='None',
             marker='.', markersize=markersize, c='g')

# plot HNS geonetric
hasg = hasg.sort_values('nr. evaluations')
plt.errorbar(column_list - 2*n,
             hasg["minimum mean"],
             hasg["minimum var"],
             label="Hide-and-Seek geometric", linestyle='None',
             marker='*', markersize=markersize, c='crimson')

# plot IHR
ihr = ihr.sort_values('nr. evaluations')
plt.errorbar(column_list + 2*n,
             ihr["minimum mean"],
             ihr["minimum var"],
             label="Improving Hit-and-Run", linestyle='None',
             marker='<', markersize=markersize, c='orange')

# plot legend
plt.legend()

# set label for axis
plt.xlabel("Objective function evaluations " + r'$F_e$')
plt.ylabel(r'$\overline{f}_{best}$')

# set title
plt.title("Six-hump Camelback problem")

# set tick labels
ax.set_xticklabels([0,25,'',50,'',100])

# enable grid
plt.grid(axis="y")

# save and show figure
plt.savefig(PATH +"sixhumpcamelback_grid.png")
plt.show()
