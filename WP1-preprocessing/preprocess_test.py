import pandas as pd
import numpy as np
import Mrch2k20.preprocess_helper as pss
from scipy.interpolate import BSpline

df_delft = pd.read_csv('../2018_01/'+ 'ADSB_DECODED_20180112.csv.gz',compression= 'gzip')  #'test' + '.csv') # test-1 smaller test dataset
df_open = pd.read_csv('../open_sky_full/'+ '2018-01-12.png',compression= 'gzip')  # test-1 smaller test dataset

# prepare data for next steps
df_delft['onground'] = np.full((df_delft.shape[0]), np.nan)
df_open = pss.opensky_manual(df_open)

# concatenate e remove duplicates
df = pd.concat([df_delft, df_open], ignore_index=True)
df = df.drop_duplicates(subset=['ts','icao'])

# remove operations out of scope
df = df.drop(df[ (df['alt'] > 10000) | (df['lon'] < 3.3) | (df['lon'] > 5.7) | (df['lat'] < 51.6)
                 | (df['lat'] > 53) | (df['alt'] < -0.1)].index)  # also removes negative altitudes

df = pss.remove_gveh(df)
df = pss.fix_ground(df)

df = df.set_index(['icao', 'ts'])

df = pss.dt_column(df)

df_point = pss.remove_spoint(df, 60.0)  # 60 seconds
df_point = pss.remove_traj(df_point, 60.0)  # 60 seconds, might be too short
df_point = pss.divide_flights2(df_point)  #todo fix divide flights
#df_use = df_point.drop(df_point[(df_point['alt']<-1)|(df_point['alt']>100)].index)
#pss.density_map(df_use)
#df_point[df_point['phase'].str.contains('C')]
df_point = df_point.reset_index().set_index('traj')
#np.array(df_point.reset_index().set_index('icao').index.unique())
df_point = pss.remove_outlier(df_point)
new_df = pss.smoother(df_point)  # compress information as well by storing info about spline
a=0
#new_df.to_csv("./Mrch2k20/pss_data.csv.gz", compression= 'gzip') #save file

# Example code
'''
example = new_df.set_index('icao').loc['484F6D']
speed = example.loc[:,'gs']
start = example.loc[:,'t_start']
end = example.loc[:,'t_end']
t = np.r_[(start[0],) * (3 + 1), speed[0][1],(end[0],) * (3 + 1)]
spline = BSpline(t, speed[0][0], 3)
t_steps = np.linspace(start[0], end[0])
import matplotlib.pyplot as plt
df_point = df_point.reset_index().set_index('icao')
interesting = df_point.loc['484F6D',:].reset_index().set_index('ts')
interesting = interesting.loc[start[0]:end[0]]
plt.plot(t_steps, spline(t_steps), '--', label='spline')
plt.plot(interesting.index.values, interesting.loc[:,'gs'].values,label='original')
plt.legend()
plt.show()
'''
