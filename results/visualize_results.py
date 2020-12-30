import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker

import pandas as pd
import sys
import pudb

IMG_PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/spheroid-tnf-v2-emews/experiments/"

class pretty_plot:
    def __init__(self, data, exp_name):
        self.data = data
        self.exp_name = exp_name
            
        
    def plot_scatter_3d(self):
        # create axis
        # ax = fig.add_subplot(subpltnr, projection='3d')
        mpl.rcParams['font.size'] = 16
        mpl.rcParams['axes.labelsize'] = 'small'
        mpl.rcParams['xtick.labelsize'] = 'small'
        mpl.rcParams['axes.formatter.use_mathtext'] = True
        #mpl.rcParams['xtick.major.pad'] = 2
        #mpl.rcParams['axes.labelpad'] = 1
        
        #mpl.rcParams['axes.formatter.offset_threshold'] = 2
        fig = plt.figure(figsize=(15, 15))
        fig, ax = plt.subplots( subplot_kw=dict(projection='3d'))
                                #constrained_layout=True)
        
        #fig.text(0.5, 0.975, 'TNF results',
        # horizontalalignment='center',
        # verticalalignment='top')
        x = self.data['user_parameters.concentration_tnf']
        y = self.data['user_parameters.time_add_tnf'] 
        z = self.data['user_parameters.duration_add_tnf']
        
        #ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.xaxis.set_ticks_position('top')
        # create label
        ax.set_xlabel('TNF concentration')
        ax.set_ylabel('time add TNF')
        ax.set_ylim(0,900)
        ax.set_zlabel('duration add TNF')

        #pu.db
        # set markersize
        alpha = 0.4
        im = ax.scatter(x, y, z,  c=self.data['score'], vmin=0, vmax=1000, s=50,cmap='Spectral')
        cbar = fig.colorbar(im, ax=ax,pad=0.1, shrink=0.75)#.ravel().tolist()
        cbar.set_label(r'$f(\mathbf{x})$')
        cbar.mappable.set_clim(1000,0)
        ax.xaxis.set_ticks_position('top')
        plt.autoscale()
        plt.savefig(IMG_PATH + "/dist.png")
        plt.show()

        
if __name__ == "__main__":
    # name of experiment
    exp_name = sys.argv[0]
    exp_name = sys.argv[1]
    print("......................")
    print(exp_name)
    # read data
    name = IMG_PATH + exp_name + "/mymetrics/results.csv"
    data = pd.read_csv(name)

    data = data.drop("Unnamed: 0", axis=1)
    # create instance of pretty_plot
    pp = pretty_plot(data, exp_name)
    pp.plot_scatter_3d()
                      
