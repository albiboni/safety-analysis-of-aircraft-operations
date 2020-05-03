import pandas as pd
import numpy as np
import Mrch2k20.preprocess_helper as pss
import tools.latable as lt
from scipy.interpolate import BSpline
import Mrch2k20.analysis_pointsxalt as tabviz
import os
altitude_10000 = 0
altitude_8000 = 0
altitude_6000 = 0
altitude_4000 = 0
altitude_2000 = 0
altitude_1000 = 0
altitude_500 =0
altitude_100 =0
icao_10000 =0
icao_8000 = 0
icao_6000 = 0
icao_4000 = 0
icao_2000 = 0
icao_1000 = 0
icao_500 = 0
icao_100 = 0

#bello = os.listdir("./data")[-1]
content = os.listdir("../open_sky_full")[1:]
content1= os.listdir("../2018_01")[3:24]


for i in range(8,12): #(0,22)
    df_delft = pd.read_csv('../2018_01/' + content1[i],compression= 'gzip')  # 'test' + '.csv') # test-1 smaller test dataset
    df_open = pd.read_csv('../open_sky_full/' + content[i], compression= 'gzip' )  # test-1 smaller test dataset
    #df_open = pd.read_csv('../2018_01/' + content1[i], compression='gzip')
    #df_delft = pd.read_csv('../2018_01/'+ 'test.csv')  #'test' + '.csv') # test-1 smaller test dataset
    #df_open = pd.read_csv('../opensky/'+ 'maybe_works' + '.csv')  # test-1 smaller test dataset
    year = '2018'# TODO: set it automatically from string of file
    month = '01'
    day = '01'
    # prepare data for next steps
    df_delft['onground'] = np.full((df_delft.shape[0]), np.nan)
    df_open = pss.opensky_manual(df_open)
    #df_open['onground'] = np.full((df_delft.shape[0]), np.nan) #general not needed
    # concatenate e remove duplicates
    df = pd.concat([df_delft, df_open], ignore_index=True)
    df = df.drop_duplicates(subset=['ts','icao'])

    # remove operations out of scope
    df = df.drop(df[ (df['alt'] > 10000) | (df['lon'] < 3.3) | (df['lon'] > 5.7) | (df['lat'] < 51.6)
                     | (df['lat'] > 53) | (df['alt'] < -0.1)].index)  # also removes negative altitudes

    df =pss.remove_gveh(df)
    df = pss.fix_ground(df)

    df = df.set_index(['icao', 'ts'])

    df = pss.dt_column(df)

    df_point = pss.remove_spoint(df, 60.0)  # 60 seconds
    df_point = pss.remove_traj(df_point, 60.0)  # 60 seconds, might be too short
    #df_point = pss.divide_flight(df_point,year,month,day)


    df_point = df_point.reset_index().set_index('traj')

    # print latex table with analysis of ADS-B data for different altitudes

    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p = tabviz.single_df_analysis(df_point)
    altitude_10000+= a
    altitude_8000 +=b
    altitude_6000 +=c
    altitude_4000 +=d
    altitude_2000 +=e
    altitude_1000 +=f
    altitude_500 +=g
    altitude_100 +=h
    icao_10000+=i
    icao_8000 +=j
    icao_6000 +=k
    icao_4000 +=l
    icao_2000 +=m
    icao_1000 +=n
    icao_500 +=o
    icao_100 +=p

days = 4
altitude_10000 = altitude_10000/ days
altitude_8000 = altitude_8000         /days
altitude_6000 = altitude_6000         /days
altitude_4000 = altitude_4000         /days
altitude_2000 = altitude_2000         /days
altitude_1000 = altitude_1000         /days
altitude_500 = altitude_500          /days
altitude_100 = altitude_100 / days
icao_10000 = icao_10000 / days
icao_8000 = icao_8000 / days
icao_6000 = icao_6000 / days
icao_4000 = icao_4000 / days
icao_2000 = icao_2000 / days
icao_1000 = icao_1000 / days
icao_500 = icao_500 / days
icao_100 = icao_100 / days
header, row, footer = lt.prepare('lccc', table=True, caption='A4 paper size',
                                pos='ht', label='a4paper')
print(header)
print(row('Altitude range [ft]', '\# fid', '\# of points', 'ratio'))
print(row('10000 - 8000',str(round(icao_10000,2)),str(round(altitude_10000,2)),str(round(altitude_10000/icao_10000,2))))
print(row('8000 - 6000',str(round(icao_8000,2) ) ,str(round(altitude_8000,2) ) ,str(round(altitude_8000/icao_8000,2) )))
print(row('6000 - 4000',str(round(icao_6000,2) ) ,str(round(altitude_6000,2) ) ,str(round(altitude_6000/icao_6000,2) )))
print(row('4000 - 2000',str(round(icao_4000,2) ) ,str(round(altitude_4000,2) ) ,str(round(altitude_4000/icao_4000,2) )))
print(row('2000 - 1000',str(round(icao_2000,2) ) ,str(round(altitude_2000,2) ) ,str(round(altitude_2000/icao_2000,2) )))
print(row('1000 - 500',str(round(icao_1000,2) ),str(round(altitude_1000,2) ) ,str(round(altitude_1000/icao_1000 ,2))))
print(row('500 - 100', str(round(icao_500 ,2)) ,str(round(altitude_500 ,2)) ,str(round(altitude_500/icao_500,2)) ))
print(row('100 - 0',   str(round(icao_100 ,2)) ,str(round(altitude_100 ,2)) ,str(round(altitude_100/icao_100,2)) ))
print(footer)




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
