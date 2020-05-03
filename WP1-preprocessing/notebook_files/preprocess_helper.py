import pandas as pd
import numpy as np
from numba import jit
import math as m
from scipy.interpolate import make_lsq_spline, LSQUnivariateSpline
from sklearn import preprocessing
from datetime import datetime

def make_upper(icao):
    return icao.upper()


def opensky_manual(df_open):
    #manual preprocessing of opensky database
    # remove duplicates and columns
    df_open = df_open.drop_duplicates(['lastposupdate','icao24'])
    df_open = df_open.drop(columns=['Unnamed: 0', 'time', 'spi', 'alert', 'geoaltitude', 'lastcontact', 'hour'])

    # reorder and rename columns
    df_open = df_open[['lastposupdate','icao24','lat', 'lon', 'baroaltitude', 'velocity', 'heading',
                       'vertrate','callsign', 'squawk','onground']]
    df_open.columns = ['ts', 'icao', 'lat', 'lon', 'alt', 'gs', 'trk', 'roc', 'callsign', 'fid', 'onground']

    # make same unit
    df_open['alt'] = df_open['alt']*3.28084   # from meter to feet
    df_open['gs'] = df_open['gs']*1.94384     # from m/s to knot
    df_open['roc'] = df_open['roc']*196.8504  # from m/s to ft/minute
    df_open['icao'] = df_open['icao'].apply(make_upper)
    return df_open

def remove_gveh(df):
    # remove ground vehicles
    alt_more_0=df[df['alt'] > 50]
    df = df[df['icao'].isin(alt_more_0['icao'].unique())]
    return df

def fix_ground(df):
    # set altitude =0 if onground is 1
    index_1s = df.drop(df[df['onground'].isnull()].index)
    index_1s = index_1s.drop(index_1s[index_1s['onground']==0].index)
    df.loc[index_1s.index, 'alt'] = float(0)
    return df

def dt_column(df):
    df = df.sort_index()
    df = df.assign(dt=pd.Series(df.index.get_level_values(1), index=df.index).diff())
    df.iloc[df.reset_index().drop_duplicates(subset=['icao']).index, 9] = np.nan  # todo:very important line
    return df

def remove_spoint(df, t):
    # t in seconds
    # spoint, single point = a point that to its right and left doesn't have neighbour
    # remove single point maybe make shorter interval, for now 60 sec and looks before and after point
    df_point = df.drop(df[(df['dt'] > t) & (df['dt'].shift(-1) > t)].index)
    df_point = df_point.drop(df[(df['dt'].isnull()) & (df['dt'].shift(-1) > t)].index)
    df_point = df_point.drop(df[(df['dt'] > t) & (df['dt'].shift(-1).isnull())].index)
    return df_point

def remove_traj(df_point, t1, t2):
    # remove trajectories with less of a minute of data and separate them based on the time distance between them.
    # the time distance for division is set to 60 sec
    df_point.loc[:,'dt'] = pd.Series(df_point.index.get_level_values(1), index=df_point.index).diff()
    list_nan = df_point.reset_index().drop_duplicates(subset=['icao']).index
    df_point.iloc[list_nan, 9] = np.nan  # todo:very important line

        # point to considerwhen to divide trajectories
    list_occurence = df_point.loc[df_point.loc[:,'dt'] > t1].index
    list_nan = df_point.loc[df_point.loc[:,'dt'].isnull()].index
    total_list = list_nan.union(list_occurence, sort=None)
    list_before= df_point.index.searchsorted(total_list)
    total_list_before = df_point.iloc[(list_before[1:]-1)].index  # todo:very important line
    last_index = df_point.index[-1]
    traj_times = np.append(total_list_before.get_level_values(1).values,last_index[1])\
                 - total_list.get_level_values(1).values
    traj_remove = np.where(traj_times < t2)[0]

        # add label for division of trajectory
    count = 1
    count_array = np.full((df_point.shape[0]), np.nan)
    for i in range(list_before.shape[0] - 1):
        count_array[list_before[i]:list_before[i+1]]=count
        count+=1
    count_array[list_before[-1]:]=count
    df_point['traj']= count_array

        # remove if trajectory is shorter than 60 seconds
    indexes = np.array([], dtype=int)
    for i in range(traj_remove.shape[0]):  # -1
        if traj_remove[i] == list_before.shape[0]-1:
            idx1 = list_before[traj_remove[i]]
            idx2 = df_point.shape[0]
        else:
            idx1 = list_before[traj_remove[i]]
            idx2 = list_before[traj_remove[i] + 1]

        index = np.arange(idx1, idx2, 1,dtype=int)
        indexes = np.append(indexes, index)
    df_point = df_point.drop(df_point.index[indexes])
    return df_point

@jit(nopython=True)
def hampel_filter_forloop(input_series, window_size, n_sigmas=3):
    '''
        Based on the implementation of Eryk Lewinson.
        https://github.com/erykml/medium_articles/blob/master/Machine%20Learning/outlier_detection_hampel_filter.ipynb
        Pearson, R. K. (1999). “Data cleaning for dynamic modeling and control”. European Control
        Conference, ETH Zurich, Switzerland.
        Function for outlier detection using the Hampel filter.
        Based on `pracma` implementation in R.

        Parameters
        ------------
        input_series : np.ndarray
            The series on which outlier detection will be performed
        window_size : int
            The size of the window (one-side). Total window size is 2*window_size+1
        n_sigmas : int
            The number of median standard deviations used for identifying outliers

        Returns
        -----------
        new_series : np.ndarray
            The array in which outliers were replaced with respective window medians
        indices : np.ndarray
            The array containing the indices of detected outliers
        '''

    n = input_series.shape[0]
    new_series = input_series.copy()
    k = 1 #1.4826  # scale factor for Gaussian distribution

    indices = []

    # possibly use np.nanmedian

    for i in range((window_size), (n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size+1)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size+1)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0): #or np.isnan(input_series[i]):  # make nan median
            new_series[i] = x0
            indices.append(i)

    #x_init = np.nanmedian(input_series[: window_size+1])
    #for i in range(0, (window_size) + 1):
    #    if np.isnan(input_series[i]):
    #        new_series[i] = x_init
#
    #x_end = np.nanmedian(input_series[(- window_size-1):])
    #for i in range((n- window_size-1), n):
    #    if np.isnan(input_series[i]):
    #        new_series[i] = x_end

    return new_series , indices

def remove_outlier(df_point):

    for i in df_point.index.unique():
        spec_traj = df_point.loc[i]  # subset of pandas select specific trajectory
        lat = spec_traj.loc[:, 'lat'].values  # array with specific property
        new_lat, indices = hampel_filter_forloop(lat, 5)  # indices contains index of array of outlier
        df_point.loc[i, 'lat'] = pd.Series(new_lat, index=spec_traj.index)  # insert back in main dataframe
        lon = spec_traj.loc[:, 'lon'].values  # array with specific property
        new_lon, indices = hampel_filter_forloop(lon, 5)  # indices contains index of array of outlier
        df_point.loc[i, 'lon'] = pd.Series(new_lon, index=spec_traj.index)  # insert back in main dataframe
        alt = spec_traj.loc[:, 'alt'].values  # array with specific property
        new_alt, indices = hampel_filter_forloop(alt, 5)  # indices contains index of array of outlier
        df_point.loc[i, 'alt'] = pd.Series(new_alt, index=spec_traj.index)  # insert back in main dataframe
        roc = spec_traj.loc[:, 'roc'].values  # array with specific property
        new_roc, indices = hampel_filter_forloop(roc, 5)  # indices contains index of array of outlier
        df_point.loc[i, 'roc'] = pd.Series(new_roc, index=spec_traj.index)  # insert back in main dataframe
        gs = spec_traj.loc[:, 'gs'].values  # array with specific property
        new_gs, indices = hampel_filter_forloop(gs, 5)  # indices contains index of array of outlier
        df_point.loc[i, 'gs'] = pd.Series(new_gs, index=spec_traj.index)  # insert back in main dataframe
        trk = spec_traj.loc[:, 'trk'].values  # array with specific property
        trk[~np.isnan(trk)] = np.rad2deg(np.unwrap(np.deg2rad(trk)[~np.isnan(trk)]))  #, discont=2. * np.pi adjust angle
        new_trk, indices = hampel_filter_forloop(trk, 5)  # indices contains index of array of outlier
        df_point.loc[i, 'trk'] = pd.Series(new_trk, index=spec_traj.index)  # insert back in main dataframe

    return df_point

#Spline method follows the implementation proposed in the paper below
#A direct method to solve optimal knots of B-spline curves: An application for non-uniform B-spline curves fitting
#Van Than Dung ,Tegoeh Tjahjowidodo

def compute_spline(data, time, idx1, idx2, k):
    '''
    :param data: data
    :param time: time
    :param idx1: current index start
    :param idx2: current index end
    :param k: order of spline
    :return: t: knots
    '''
    idx2 = idx2-1
    variable = data[idx1:idx2]
    time = time[idx1:idx2]
    t = np.r_[(time[0],)*(k+1),  (time[-1],)*(k+1)]
    spl_lsq = make_lsq_spline(time, variable, t, k=k)
    residuals = np.power((variable - spl_lsq(time)),2)
    avg_sum_residuals = np.sum(residuals) #/ variable.shape[0] if uncomment do average per point
    return spl_lsq, avg_sum_residuals

def optimal_knots(speed_normal, time_speed):

    speed = preprocessing.normalize(speed_normal.reshape(-1, 1))
    start_idx = 0
    end_idx = speed.shape[0]
    degree = 3
    error = 1e-29  # 0.000000000000000000000000000001  # maybe do average mse per point
    # Knots= np.zeros(2+2*degree)
    save_data = np.zeros(speed.shape[0] * 1000, dtype=int)
    store_idx = 0
    while end_idx > start_idx + 1:
        save_index1 = start_idx
        save_index2 = end_idx
        left_idx = start_idx
        right_idx = end_idx

        while (right_idx - left_idx) > 1:

            if start_idx + degree >= speed.shape[0]:
                break

            if end_idx - start_idx < degree + 2:
                left_idx = end_idx
                save_index1 = start_idx
                save_index2 = end_idx
                break


            spl_local, mse_local = compute_spline(speed, time_speed, start_idx, end_idx, degree)

            success = 0

            if mse_local <= error:
                success = 1
                # save_local = [start_idx, end_idx, mse_local, spl_local]
                save_index1 = start_idx
                save_index2 = end_idx

            if success == 1:
                left_idx = end_idx
            else:
                right_idx = end_idx

            end_idx = m.floor((left_idx + right_idx) / 2)

        start_idx = left_idx + 1
        save_data[store_idx] = save_index1
        store_idx += 1
        save_data[store_idx] = save_index2
        store_idx += 1
        # save_data.append(save_local)
        end_idx = speed.shape[0]

    return np.unique(save_data)


def spline_coeff(lat, time):

    time = time[~np.isnan(lat)]
    time_diff = time[1:]-time[:-1]
    counter = 0
    indeces_remotion = np.array([])
    for i in range(0, time_diff.shape[0]):

        if time_diff[i] > 60:
            if counter < 5:
                slack = np.arange(i - counter, i + 1)
                indeces_remotion = np.append(indeces_remotion, slack)

            counter = 0

        else:
            counter += 1
    time = np.delete(time, indeces_remotion)
    lat = lat[~np.isnan(lat)]
    lat = np.delete(lat, indeces_remotion)
    if lat.shape[0] > 5:
        save_data = optimal_knots(lat, time)
        t2 = time[save_data[1:-1]]
        try2 = LSQUnivariateSpline(time, lat, t2, k=3)
        spl_coeff = try2.get_coeffs()
        knots = np.r_[(time[save_data[0]],) * (3 + 1), t2, (time[save_data[-1]-1],) * (3 + 1)]
        return [spl_coeff, knots]  # in this case returns spline coeff and knots
    else:
        return ["not enough data", lat]  # in this case returns original data

def smoother(df_point):

    n_rows = df_point.index.unique().shape[0]
    init_columns = ['icao', 'phase', 'n_points', 'duration', 't_start', 't_end', 'lat', 'lon', 'alt', 'gs', 'trk', 'roc']
    init_array = np.empty([n_rows, len(init_columns)])  # might specify dtype
    new_df = pd.DataFrame(init_array, columns=init_columns)
    counter_rows = 0
    new_df['alt'] = new_df['alt'].astype('object')
    new_df['lon'] = new_df['lon'].astype('object')
    new_df['lat'] = new_df['lat'].astype('object')
    new_df['trk'] = new_df['trk'].astype('object')
    new_df['gs'] = new_df['gs'].astype('object')
    new_df['roc'] = new_df['roc'].astype('object')

    for i in df_point.index.unique():
        spec_traj = df_point.loc[i]  # subset of pandas select specific trajectory
        new_df.loc[counter_rows, 'n_points'] = spec_traj.shape[0]
        new_df.loc[counter_rows, 'icao'] = spec_traj.loc[:,'icao'].values[0]
        t_start = spec_traj.loc[:,'ts'].values[0]
        t_end = spec_traj.loc[:, 'ts'].values[-1]
        new_df.loc[counter_rows, 't_start'] = t_start
        new_df.loc[counter_rows, 't_end'] = t_end
        new_df.loc[counter_rows, 'duration'] = t_end - t_start
        if spec_traj.loc[:, 'phase'].values[0][9:10]=='G':
            new_df.loc[counter_rows, 'phase'] = spec_traj.loc[:, 'phase'].values[-1]  # changes results of divide flight
        else:
            new_df.loc[counter_rows, 'phase'] = spec_traj.loc[:, 'phase'].values[0]

        time = spec_traj.loc[:, 'ts'].values
        lat = spec_traj.loc[:, 'lat'].values  # array with specific property
        #lat[~np.isnan(lat)]
        #time[~np.isnan(lat)]
        #latitude = preprocessing.normalize(lat[~np.isnan(lat)].reshape(-1, 1))
        new_df.at[counter_rows, 'lat'] =spline_coeff(lat,time)

        lon = spec_traj.loc[:, 'lon'].values
        new_df.at[counter_rows, 'lon'] = spline_coeff(lon, time)

        alt = spec_traj.loc[:, 'alt'].values
        new_df.at[counter_rows, 'alt'] = spline_coeff(alt, time)

        gs = spec_traj.loc[:, 'gs'].values
        new_df.at[counter_rows, 'gs'] = spline_coeff(gs, time)

        trk = spec_traj.loc[:, 'trk'].values
        new_df.at[counter_rows, 'trk'] = spline_coeff(trk, time)

        roc = spec_traj.loc[:, 'roc'].values
        new_df.at[counter_rows, 'roc'] = spline_coeff(roc, time)
        counter_rows += 1

    return new_df

def divide_flight(df, year, month, day):
    array_init = np.empty(df.shape[0], dtype='U21')
    df = df.assign(phase=pd.Series(array_init, index=df.index))

    for icao in df.index.get_level_values(0).unique():
        #icao = '40666B'
        primary_df = df.loc[icao]
        flight_df = primary_df.reset_index()
        # flight = df_count.loc[icao].loc[:, ['ts', 'lat', 'lon', 'alt', 'dt']].values

        cluster_g = -1
        cluster_r = 1
        clusters_r = np.zeros((flight_df.shape[0]))  # 0 means not filled
        clusters_g = np.zeros((flight_df.shape[0]))  # 0 means not filled
        awareness = np.full((flight_df.shape[0]), -1, dtype='U21')  # -1 means not filled

        # Look for ground movements
        ground = flight_df.loc[flight_df.loc[:, 'alt'] < 50]
        ground_index = ground.index.values
        if ground_index.shape[0] > 0:
            clusters_g[ground_index[0]] = cluster_g
            cluster_g = cluster_g - 1

            # First case, a/c lands at Schiphol
            if ground_index[0] > 0:
                clusters_g[0:ground_index[0]] = cluster_r
                cluster_r = cluster_r + 1

            # Multiple landings/take-off in the same day
            if ground_index.shape[0] > 1:
                diff = ground_index[1::] - ground_index[0:-1]
                for i in range(0, diff.shape[0]):
                    if diff[i] == 1:
                        clusters_g[ground_index[i + 1]] = clusters_g[ground_index[i]]

                    elif diff[i] > 1:
                        clusters_g[ground_index[i + 1]] = cluster_g
                        cluster_g = cluster_g - 1
                        # Todo: maybe add time division
                        clusters_g[ground_index[i] + 1:ground_index[i + 1]] = cluster_r
                        cluster_r = cluster_r + 1
                    else:
                        print('there is a problem')

            # filling last part of array in case ops doesn't finish on the ground
            if ground_index[-1] != clusters_g.shape[0] - 1:
                clusters_g[(ground_index[i + 1] + 1)::] = cluster_r
                cluster_r = cluster_r + 1
                # ground_slices = np.where(ground_index[1::] - ground_index[0:-1] != 1)[0] todo:delete

        # awareness[ground_index] = 0 # g:0, l:1, t:2 todo:delete
        # flight_df = flight_df.loc[flight_df.loc[:, 'alt'] > 50]  # remove elements just added todo:delete

        # flight_df = flight_df.assign(cluster=pd.Series(np.zeros((flight_df.shape[0])), index=flight_df.index))

        # Look for time intervals where it is sure there has been a change
        time_df = flight_df.loc[flight_df.loc[:, 'dt'] > 20. * 60.]  # 20 minutes todo: important line
        time_df_index = time_df.index.values  # here things happen
        start = 0

        for i in range(0, time_df_index.shape[0]):
            clusters_r[start:time_df_index[i]] = cluster_r
            cluster_r = cluster_r + 1
            start = time_df_index[i]

        # last step, which include also case of a single trajectory

        clusters_r[start::] = cluster_r
        cluster_r = cluster_r + 1

        clusters_g_only = np.zeros((flight_df.shape[0]))
        clusters_g_only[ground_index] = clusters_g[ground_index]
        clusters_total = clusters_r + clusters_g
        clusters_total[ground_index] = clusters_g_only[ground_index]

        # small test
        if np.where(clusters_total == 0)[0].shape[0] > 0:
            print('Albi, there is an error, double check your code')

        # Awareness of situation + NEW FID
        uniques, index_cluster = np.unique(clusters_total, return_index=True)
        index_cluster = np.sort(index_cluster)

        g = 0
        l = 0
        t = 0
        fid = day + '.' + month + '.' + year[-2:] + '.'  # + icao + '.'

        for i in range(0, (index_cluster.shape[0] - 1)):
            if set(range(index_cluster[i], index_cluster[i + 1])).issubset(ground_index):
                awareness[index_cluster[i]:index_cluster[i + 1]] = fid + 'G' + str(g)  # ground movements
                g += 1

            elif flight_df.loc[index_cluster[i]:index_cluster[i + 1], 'alt'].diff().sum() > 0:
                awareness[index_cluster[i]:index_cluster[i + 1]] = fid + 'T' + str(t)  # take-off
                t += 1

            elif flight_df.loc[index_cluster[i]:index_cluster[i + 1], 'alt'].diff().sum() < 0:
                awareness[index_cluster[i]:index_cluster[i + 1]] = fid + 'L' + str(l)  # landing
                l += 1

        # final step + case of single trajectory
        final_fid = index_cluster[-1]

        if flight_df.loc[final_fid::, 'alt'].diff().sum() > 0:
            awareness[final_fid::] = fid + 'T' + str(t)  # take-off

        elif flight_df.loc[final_fid::, 'alt'].diff().sum() < 0:
            awareness[final_fid::] = fid + 'L' + str(l)  # landing

        elif flight_df.loc[final_fid::, 'alt'].diff().sum() == 0:
            awareness[final_fid::] = fid + 'G' + str(l)  # ground

        # small test
        if np.where(awareness == str(-1))[0].shape[0] > 0:
            print('Albi2, there is an error, double check your code')

        df.loc[icao, 'phase'] = awareness #pd.Series(awareness, index=primary_df.index)
        #df_point['traj'] = count_array
    return df

from cartopy.feature import NaturalEarthFeature
from cartopy.crs import Geodetic, EuroPP,PlateCarree, Mercator
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from cartopy.crs import Geodetic, EuroPP, PlateCarree, Mercator
import tools.plot_schiphol as plotschiphol
#df_point = df_point.drop(df_point[df_point['alt']>1].index) #only ground
def plot_netherlands():
    
    fig = plt.figure(figsize=(12,9)) # open matplotlib figure
    ax = plt.axes(projection=EuroPP(), zorder=1) #

    land = NaturalEarthFeature(
                category='cultural',
                name='admin_0_countries',
                scale='10m',
                facecolor='none',
                edgecolor='#524c50', alpha= 0.3)
    ax.add_feature(land, zorder=2)
    ax.set_extent((3, 7.5, 51, 54))
    plotschiphol.plot_schiphol(ax)
    return fig, ax

def density_map(df_delft):
    #df_delft = df_delft.drop(df_delft[(df_delft['alt']<-1)|(df_delft['alt']>500)].index)#loc[df_delft.loc[:, 'alt'] < 10000]
    lon = df_delft.loc[:,'lon'].values
    lat = df_delft.loc[:,'lat'].values
    dxlon = 0.005#0.00005
    dylat = 0.005#0.00005
    gridxlon =  np.arange(4, 5.4+dxlon,dxlon)
    gridylat =  np.arange(51.5, 52.9 + dylat, dylat)
    cmap = mpl.cm.Blues
    grid,gridx, gridy = np.histogram2d(lon, lat, bins=[gridxlon, gridylat])
    fig, ax = plot_netherlands()
    barra = ax.pcolormesh(gridx, gridy, np.transpose(grid), cmap=cmap, zorder=2, transform= PlateCarree())
    #ax.scatter(lon, lat, s= 0.01, marker='o', color='r', zorder=4, transform= PlateCarree())

    plt.colorbar(barra)
    plt.show()

    return fig, ax


def unix2utc(ts):
    utc_time = datetime.utcfromtimestamp(ts)
    return utc_time

def divide_flights2(df_point):
    df_point.loc[:,'dt'] = pd.Series(df_point.index.get_level_values(1), index=df_point.index).diff()
    list_nan = df_point.reset_index().drop_duplicates(subset=['icao']).index
    df_point.iloc[list_nan, 9] = np.nan  # todo:very important line ricontrolla dopo aver aggiunto colonna traj
    array_init = np.empty(df_point.shape[0], dtype='U21')
    df_point = df_point.assign(phase=pd.Series(array_init, index=df_point.index))
    year_month_day_utc = df_point.index.get_level_values(1)[-1]
    year_month_day = str(unix2utc(year_month_day_utc))[2:10]

    for icao in df_point.index.get_level_values(0).unique():
        flight_df = df_point.loc[icao]
        flight_df = flight_df.reset_index()
        time_division = 20. * 60.  # 20 minutes
        clusters = np.zeros((flight_df.shape[0]), dtype='U21')

        # know when a change of phase happen
        list_occurence = flight_df.loc[flight_df.loc[:,'dt'] > time_division].index.values
        l = 0  # land
        t = 0  # take-off
        c = 0  # not known exactly and not ground
        if list_occurence.shape[0] >0:

            #begin
            begin = flight_df.loc[0:list_occurence[0], 'alt'].diff().sum()
            if begin > 100: #goes up
                clusters[0:list_occurence[0]] = year_month_day + '-T' + str(t)
                t +=1
            elif begin < -100: #goes up
                clusters[0:list_occurence[0]] = year_month_day + '-L' + str(l)
                l +=1
            else:
                clusters[0:list_occurence[0]] = year_month_day + '-C' + str(c)
                c += 1

            if list_occurence.shape[0] > 1:
                #intra cases
                for i in range(0, list_occurence.shape[0]-1):
                    between = flight_df.loc[list_occurence[i]:list_occurence[i + 1], 'alt'].diff().sum()
                    if between > 100:  # goes up
                        clusters[list_occurence[i]:list_occurence[i + 1]] = year_month_day + '-T' + str(t)
                        t += 1
                    elif between < -100:  # goes up
                        clusters[list_occurence[i]:list_occurence[i + 1]] = year_month_day + '-L' + str(l)
                        l += 1
                    else:
                        clusters[list_occurence[i]:list_occurence[i + 1]] = year_month_day + '-C' + str(c)
                        c += 1

            #end
            if list_occurence[-1] < flight_df.shape[0]-1:
                end = flight_df.loc[list_occurence[-1]:, 'alt'].diff().sum()
                if end > 100:  # goes up
                    clusters[list_occurence[-1]:] = year_month_day + '-T' + str(t)
                    t += 1
                elif end < -100:  # goes up
                    clusters[list_occurence[-1]:]  = year_month_day + '-L' + str(l)
                    l += 1
                else:
                    clusters[list_occurence[-1]:]  = year_month_day + '-C' + str(c)
                    c += 1



        # know when it is on ground
        list_ground = flight_df.loc[flight_df.loc[:, 'alt'] < 100].index.values
        if list_ground.shape[0] > 0:
            if list_occurence.shape[0] > 0:
                for i in list_ground:
                    g = np.searchsorted(list_occurence, i, side='right')
                    clusters[i] = year_month_day + '-G' + str(g)

            elif list_occurence.shape[0] == 0:
                clusters[list_ground] = year_month_day + '-G0'

                if list_ground[0] > 0:
                    begin = flight_df.loc[0:list_ground[0], 'alt'].diff().sum()
                    if begin > 100:  # goes up
                        clusters[0:list_ground[0]] = year_month_day + '-T' + str(t)
                        t += 1
                    elif begin < -100:  # goes up
                        clusters[0:list_ground[0]] = year_month_day + '-L' + str(l)
                        l += 1
                    else:
                        clusters[0:list_ground[0]] = year_month_day + '-C' + str(c)
                        c += 1

                if list_ground[-1] < flight_df.shape[0]-1:
                    end = flight_df.loc[list_ground[-1]:, 'alt'].diff().sum()
                    if end > 100:  # goes up
                        clusters[(list_ground[-1]+1):] = year_month_day + '-T' + str(t)
                        t += 1
                    elif end < -100:  # goes up
                        clusters[(list_ground[-1]+1):] = year_month_day + '-L' + str(l)
                        l += 1
                    else:
                        clusters[(list_ground[-1]+1):] = year_month_day + '-C' + str(c)
                        c += 1

            # final case list_ground and list_occurence are 0, meaning it is a single trajectory
        if list_ground.shape[0] == 0 and list_occurence.shape[0] == 0:
            s_traj = flight_df.loc[:, 'alt'].diff().sum()
            if s_traj > 100:  # goes up
                clusters[:] = year_month_day + '-T' + str(t)
                t += 1
            elif s_traj < -100:  # goes down
                clusters[:] = year_month_day + '-L' + str(l)
                l += 1
            else:
                clusters[:] = year_month_day + '-C' + str(c)
                c += 1
        df_point.loc[icao, 'phase'] = clusters

    return df_point