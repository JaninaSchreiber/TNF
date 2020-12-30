import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

'''
Generates the distribution plots for the shcb problem
'''

PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/EMEWS-scripts/testcase/sixhumpcamelback2/"

plt.rcParams.update({'font.size': 20})

max_eval = 100
evaluations = "100"
min = [[-0.0898, 0.0898],[0.7126, -0.7126]]

algorithm = "randomsearch"
testnr = "3"

file = "{}{}/test{}/".format(algorithm, evaluations, testnr)
rs = pd.read_csv(PATH + file + "results.csv")

rs = rs[:max_eval]
rs["score"]= (rs["score"]--1.03162)/(163 --1.03162)

algorithm = "improving hit-and-run"
testnr = "3"

file = "{}{}/test{}/".format(algorithm, evaluations, testnr)
ihr = pd.read_csv(PATH + file + "results.csv")

ihr = ihr[:max_eval]
ihr["score"]= (ihr["score"]--1.03162)/(163 --1.03162)

algorithm = "hide-and-seek"
testnr = "3"

file = "{}{}adaptive/test{}/".format(algorithm, evaluations, testnr)
hnsa = pd.read_csv(PATH + file + "results.csv")

hnsa = hnsa[:max_eval]
hnsa["score"]= (hnsa["score"]--1.03162)/(163 --1.03162)

algorithm = "hide-and-seek"
testnr = "3"

file = "{}{}geometric/test{}/".format(algorithm, evaluations, testnr)
hnsg = pd.read_csv(PATH + file + "results.csv")

hnsg = hnsg[:max_eval]
hnsg["score"]= (hnsg["score"]--1.03162)/(163 --1.03162)

alg_list = [rs, ihr, hnsa, hnsg]
namelist = ["rs", "ihr", "hnsadaptive", "hnsgeometric"]

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(18, 10))

lw = 7.5
alpha = 0.9
x = 0
y = 0
evaluations= max_eval
for i in range(len(alg_list)):
    df = alg_list[i]
    im =  axs[x][y].scatter(df.x1, df.x2, c=df['score'], norm=colors.LogNorm(),
                            cmap='Spectral', alpha=alpha, marker='o', linewidth=lw,
                            s=10)
    axs[x][y].scatter(x=min[0], y=min[1], c='black', marker='^', linewidth=lw,)
    if namelist[i]=="rs":
        title = "PRS with $F_e = {}$".format(evaluations)
    elif namelist[i]=="ihr":
        title = "IHR with $F_e = {}$".format(evaluations)
    elif namelist[i]=="hnsadaptive":
        title = "HNS adaptive with $F_e = {}$".format(evaluations)
    else:
        title = "HNS geometric with $F_e = {}$".format(evaluations)
    axs[x][y].set_title(title)
    x += 1
    plt.xlim(-3, 3)
    plt.ylim(-2,2)
    if(x==2):
        y += 1
        x = 0


fig.text(0.45, 0.04, r'$x_1$', ha='center')
fig.text(0.04, 0.5, r'$x_2$', va='center', rotation='vertical')
cbar = plt.colorbar(im, ax=axs, pad=0.1)#ravel().tolist()
cbar.set_label(r'$g_z$')
        
plt.savefig(PATH + str(max_eval) + "distributionplot.png")



fig, axs = plt.subplots(figsize=(10,10))
im =  axs.scatter(df.x1, df.x2, c=df['score'], norm=colors.LogNorm(),
                  cmap='Spectral', alpha=alpha, marker='o', linewidth=lw,
                  s=10)
axs.scatter(x=min[0], y=min[1], c='black', marker='^', linewidth=lw)
plt.xlim(-3, 3)
plt.ylim(-2,2)
axs.set_title(title)
fig.text(0.45, 0.04, 'x(1)', ha='center')
fig.text(0.04, 0.5, 'x(2)', va='center', rotation='vertical')
cbar = plt.colorbar(im, ax=axs, pad=0.1)#ravel().tolist()
cbar.set_label(r'$g_z$')
plt.autoscale()
plt.savefig(PATH + file + "sampling_dist.png")



