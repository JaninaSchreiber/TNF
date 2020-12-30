import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as mticker
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter

import sys
import ga_utils
import pudb



IMG_PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/spheroid-tnf-v2-emews/experiments/"

class pretty_plot:
    def __init__(self, data, bounds_df, mypath):
        self.data = data
        self.mypath = mypath
        self.bounds_df = bounds_df

    def iterate_params(self):
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.labelsize'] = 'small'
        mpl.rcParams['xtick.labelsize'] = 'small'
        mpl.rcParams['axes.formatter.use_mathtext'] = True
        #mpl.rcParams['figure.constrained_layout.use'] = True
        
        fig = plt.figure(figsize=(14, 10))
        fig, axes = plt.subplots(nrows=3, ncols=3,
                                 sharex=True, sharey=True)
        #fig.suptitle('Results')
        #print("data min score {}".format(self.data['score'].min()))
        k,l = 0, 0
        mybool = True
        for j in np.arange(0, 900, 100):
            j_lower = j
            j_upper = j + 100
            j_max = self.bounds_df['upper_bound'].loc['user_parameters.time_add_tnf'] 
            if(l >= 3):
                l = 0 
                k += 1
                mybool = False
            j_upper = min(j_upper, j_max)
            df = self.data[self.data['user_parameters.time_add_tnf'] <= j_upper][self.data['user_parameters.time_add_tnf'] >= j_lower]
            ax = axes[k][l]
            fig, ax, im =  self.plot_scatter_3d(df, ax, fig, l, j_upper)
            if(mybool == True):
                ax.set_title(round(j_upper, 3))
            l += 1
 
                
        fig.text(0.45, 0.04, 'TNF concentration in ' + r'$\mu m^³$', ha='center')
        fig.text(0.04, 0.5, 'duration add TNF in min.', va='center', rotation='vertical')
        fig.text(0.45, 0.92, 'time add TNF in min.', ha='center')
        #fig.text(0.78, 0.5, 'immune kill rate', va='center', rotation='vertical')
        plt.subplots_adjust(left=0.13, right=0.95, bottom=0.15, top=0.85)
        plt.locator_params(axis="y", nbins=3)
        
        cbar = fig.colorbar(im, ax=axes,pad=0.1, location='right')#.ravel().tolist()
        cbar.set_label(r'$f(\vec{x})$')
        cbar.mappable.set_clim(1000,0)
        plt.autoscale()

        #mpl.cm.ScalarMappable.set_clim(0., 1000.)
        plt.savefig(mypath + "results_d1d4.png")
        plt.show()
            
        
    def plot_scatter_3d(self, data, ax, fig, l, i_upper):
        # create axis
        mpl.rcParams['font.size'] = 8

        x = data['user_parameters.concentration_tnf']
        y = data['user_parameters.duration_add_tnf']

        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.set_xlim(5,30)
        ax.set_ylim(0.005,0.4)
        #ax.locator_params(axis="y", nbins=3)
        #ax.set_yticks([0,0.5,1])
  
        ax.set_title(str(round(i_upper,3)))
        ax.yaxis.set_label_position("right")
        
        # set markersize
        lw = 2
        alpha = 0.4

        im1 =  ax.scatter(x, y, c=data['score'],vmin=0, vmax=1000,
                    cmap='Spectral', alpha=alpha, marker='.', linewidth=lw)
        return fig, ax, im1

    
def transform_param(o_param):
    """
    transforms parameters into a dataframe
    """
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
    
def retrieve_bounds(bounds_file):
    """
    returns a DataFrame with bounds and the step size
    """
    bounds_file = bounds_file.replace('\'', '\"')
    bounds_params = ga_utils.create_parameters(bounds_file)
    bounds_params = transform_param(bounds_params)
    return bounds_params


if __name__ == "__main__":
    # name of experiment
    exp_name = sys.argv[0]
    exp_name = sys.argv[1]
    # bounds file in .json format
    bounds_file  = sys.argv[2]

    print("......................")
    print(exp_name)
    # read data
    mypath = IMG_PATH + exp_name + "/mymetrics/"
    name = mypath + "results.csv" 
    data = pd.read_csv(name)
    data["score"][data['score']>=900]=1000

    data = data.drop("Unnamed: 0", axis=1)

    # get bounds of the domain and step size 
    bounds_df = retrieve_bounds(bounds_file)
    
    # create instance of pretty_plot
    pp = pretty_plot(data, bounds_df, mypath)
    pp.iterate_params()
