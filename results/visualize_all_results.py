import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker

import pudb
import pandas as pd
import sys
import os

IMG_PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/spheroid-tnf-v2-emews/experiments/"


def plot_scatter_3d():
    # create axis
    # ax = fig.add_subplot(subpltnr, projection='3d')
    mpl.rcParams['font.size'] = 26
    mpl.rcParams['axes.titlesize'] = 'large'
    #mpl.rcParams['xtick.labelsize'] = 'small'
    mpl.rcParams['axes.formatter.use_mathtext'] = True

    fig, axes = plt.subplots(2,2, figsize=(20, 10),
                             subplot_kw=dict(projection='3d'),
                             constrained_layout=True)
    #constrained_layout=True)
    i = 0
    j = 0
    for dir in os.listdir(IMG_PATH):
        #fig.text(0.5, 0.975, 'TNF results',
        # horizontalalignment='center',
        # verticalalignment='top')
        name = dir[:-5].replace('_', ' ')
        name = name.replace(name[:3],name[:3].upper())
        data = pd.read_csv(IMG_PATH + dir + '/mymetrics/myresults.csv')
        x = data['user_parameters.concentration_tnf']
        y = data['user_parameters.time_add_tnf']
        z = data['user_parameters.duration_add_tnf']
        alpha = 0.9
        im = axes[i][j].scatter(x, y, z,  c=data['score'], vmin=0, vmax=1000, s=100,marker='o', cmap='Spectral', linewidths=4)
        axes[i][j].set_ylim(0,900)
        axes[i][j].set_yticks([0,200,400,600, 800])
        axes[i][j].set_xlim(0,0.4)
        axes[i][j].set_xticks([0.0,0.1,0.2,0.3])
        axes[i][j].title.set_text(name)
        i += 1
        if(i==2):
            i = 0
            j = 1
            #axes[i][j].set_zlabel('duration add TNF')
    cbar = fig.colorbar(im, ax=axes,pad=0.1, shrink=0.55)
    cbar.set_label(r'$f(\mathbf{x})$')
    fig.text(x=0.83, y=0.37, s='duration add TNF [min]', rotation=90, fontsize=30)
    fig.text(x=0.78, y=0, s='time add TNF [min]', rotation=55, fontsize=30)
    fig.text(x=0.28, y=0, s='TNF concentration '+ r'$[\mu m^3]$', fontsize=30)
    plt.autoscale()
    fig.set_size_inches(16, 18)
    plt.savefig(IMG_PATH + 'dist.png')


        
if __name__ == "__main__":
    # name of experiment
    plot_scatter_3d()
                      
