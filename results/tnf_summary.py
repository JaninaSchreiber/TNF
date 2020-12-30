import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

PATH = '/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/spheroid-tnf-v2-emews/experiments/'

dirs = os.listdir(PATH)

plt.rcParams['font.size']=28

fig = plt.figure(figsize=(15,10))

j=0
color = ['tab:blue', 'g', 'orange', 'crimson']
x = np.arange(100)
for dir in dirs:
    df = pd.read_csv(PATH + dir + '/mymetrics/results.csv')
    name = dir[:-5].replace('_', ' ')
    name = name.replace(name[:3],name[:3].upper())
    y = []
    for i in x:
        y.append(df['score'][:i].min())
    print(y)
    plt.plot(x, y, label=name, c=color[j], linewidth=2)
    j += 1
 
plt.legend()
plt.xlabel(r'$F_e$')
plt.ylabel(r'$f_{best}$')
plt.autoscale()
plt.savefig(PATH + 'results_tnf_c.png')
