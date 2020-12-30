import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime

PATH = "/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/EMEWS-scripts/testcase/scripts/data/"

# read time summary as pandas DataFrame
df = pd.read_csv(PATH + "time_summary.csv")

# format the time
format = '%H:%M:%S.%f'
df["totaltime"] = pd.to_datetime(df.totaltime, format=format).dt.time
df["algtime"] = pd.to_datetime(df.algtime, format=format).dt.time

for i in range(len(df)):
    t =  df.totaltime[i]
    df.totaltime[i] = int(datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds())

df = df.sort_values(by=['algorithm', 'evalutations']) 
    #df.totaltime[i] =  str(df.totaltime[i])[:-7]

# create subframe for each algorithm
random = df[df["algorithm"]=="randomsearch "]
hasa =  df[df["algorithm"]=="hide-and-seek adaptive"]
hasg =  df[df["algorithm"]=="hide-and-seek geometric"]
ihr = df[df["algorithm"]=="improving hit-and-run "]

# space between labels
n = 1

#generate figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

linestyle = 'dotted'
x_param = 'algtime'

# plot PRS 
plt.plot(random["evalutations"],
              random[x_param],
              linestyle=linestyle, marker='^',c='tab:blue', label='PRS')

# plot HNS adaptive
hasa = hasa.sort_values(by=['evalutations']) 
plt.plot(hasa["evalutations"],
              hasa[x_param],
              linestyle=linestyle, marker='.',c='g', label='HNS adaptive')

# plot HNS geometric
hasg = hasg.sort_values(by=['evalutations']) 
plt.plot(hasg["evalutations"],
              hasg[x_param],
              linestyle=linestyle, marker='*',c='crimson', label='HNS geometric')

# plot IHR
ihr = ihr.sort_values(by=['evalutations']) 
plt.plot(ihr["evalutations"],
              ihr[x_param],
              linestyle=linestyle, marker='<',c='orange', label='IHR')

# create legend
plt.legend()

# set labels
plt.xlabel("Objective function evaluations")
plt.ylabel("Time in seconds")
plt.title("Six-Hump Camelback problem")

# save plot
plt.savefig(PATH + "sixhumptime_time_c.png")
plt.show()
