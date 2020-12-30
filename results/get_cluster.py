import pandas as pd
import math

df = pd.read_csv("../experiments/ihr_1440/mymetrics/results.csv")

df = df.drop('Unnamed: 0',axis=1)

df1['user_parameters.time_add_tnf']
df1['user_parameters.duration_add_tnf']
df1['user_parameters.concentration_tnf']

df = pd.read_csv("../experiments/hns_adaptive_1440/mymetrics/results.csv")
df3 = df[df['score']<350]

mean = df1.mean()

intv_pos = z90 * df1.std()/math.sqrt(len(df1))
z90 = 1.645
z99 = 2.576


    
