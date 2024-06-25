import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import matplotlib.dates as mdates
from datetime import timedelta
import h5py
import os
import numpy as np
#PURPOSE:
# This function has a routine for obtaining interpolations.  
#INPUT:
# xl= Geographic Longitude
# order= order by interpalation
# il= A integer from 1 to 8
#OUTPUT:
# bspl4_long

def getRangeIndices(h_min, h_max, range_array):
    min_ind = list(np.abs(h_min-range_array)).index(min(np.abs(h_min-range_array)))
    max_ind = list(np.abs(h_max-range_array)).index(min(np.abs(h_max-range_array)))
    #print(min_ind, max_ind)
    #print(range_array[min_ind], range_array[max_ind])
    return min_ind, max_ind

def SumArrays(a,b):
    fils,cols = a.shape
    c = np.ones((fils,cols))*np.nan
    for i in range(fils):
        for j in range(cols):
            if np.isnan(a[i,j]):
                if np.isnan(b[i,j]):
                    c[i,j]=np.nan
                else:
                    c[i,j] = b[i,j]
            else: 
                if np.isnan(b[i,j]):
                    c[i,j] =  a[i,j]
                else:
                    #print(a[i,j], b[i,j])
                    c[i,j] = a[i,j] + b[i,j]
    return c

def ShowArrays(directory, filename, PlotFlag):
    file_hf5 = directory + os.sep +filename
    hf = h5py.File(file_hf5, 'r')
    rango = hf['Data/Table Layout/']['gdalt']
    timestamps = hf['Data/Array Layout/']['timestamps']
    vipe1 = hf['Data/Array Layout/2D Parameters/vipe']
    vipn1 = hf['Data/Array Layout/2D Parameters/vipn']
    v_zonal = np.array(vipe1)
    v_vertical = np.array(vipn1)
    delta_range = np.diff(rango)[0]
    range2D = rango.reshape(v_vertical.T.shape)

    MinRange, MaxRange = np.min(rango), np.max(rango)
    DataMatrixRows = int((MaxRange-MinRange)/delta_range)
    #range_array = np.linspace(MinRange, MaxRange, DataMatrixRows+1)
    range_array = range2D[0]
    datetime_objects = []
    for ts in timestamps:
        date_time_obj = datetime.datetime.fromtimestamp(ts)
        datetime_objects.append(date_time_obj)
    datetime_objects = np.array(datetime_objects)
    #year  = datetime_objects[0].astype('datetime[ns]').year
    #month = datetime_objects[0].astype('datetime[ns]').month
    #day   = datetime_objects[0].astype('datetime[ns]').day
    year  = datetime_objects[0].year
    month = datetime_objects[0].month
    day   = datetime_objects[0].day
    str_month = GetMonth(month)
    dir_plots = 'Plots-%s-%d' % (str_month, year)
    plot_format = 'png'
    v_vertical = RemoveIQR(v_vertical)
    v_vertical[v_vertical<-45] = np.nan
    v_vertical[v_vertical> 45] = np.nan
    print("Shapes inside: ", rango.shape, datetime_objects.shape, v_vertical.shape)
    #'''
    if PlotFlag:
        fig, ax = plt.subplots(figsize=(12, 6))
    #plt.rcParams['xtick.labelsize']=14
        #plt.style.use('dark_background')
        print("Shapes: ", datetime_objects.shape, range_array.shape, v_vertical.shape)
        print(range_array)
        clrs= ax.pcolormesh(mdates.date2num(datetime_objects), range_array, v_vertical, cmap=plt.cm.RdBu_r)
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_xlabel("Hora Local (h)", fontsize=16)
        ax.set_ylabel("Rango (km)", fontsize=17)
        ax.set_ylim(300,400)
        fig_title = r'Derivas Verticales ISR (%d-%02d-%02d)' % (year, month, day)
        plt.title(fig_title, fontsize=15)
        str_date = '(%d-%02d-%02d)' % (year, month, day)
        #ax.set_ylim(300, 1200)
        # This simply sets the x-axis data to diagonal so it fits better.
        fig.autofmt_xdate()
        box=ax.get_position()
        cbarax=fig.add_axes([box.x0+box.width+0.01, box.y0, 0.025, box.height])
        cb=plt.colorbar(clrs,cax=cbarax)
        cb.set_label(r'Derivas verticales (m/s)')
        plt.savefig(r'%s/Drifts-ISR-%d-%02d-%02d-v-vertical.%s' % (dir_plots,year, month, day, plot_format),bbox_inches='tight')
        plt.show()
        plt.close(fig)
    #'''

    return    range_array,datetime_objects,v_vertical
#range_array,datetime_objects[0:154],v_vertical[:,0:154]

def interp_mat(v_vertical):
    A = v_vertical.copy()
    ok = ~np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~np.isnan(A)]
    x  = np.isnan(A).ravel().nonzero()[0]

    A[np.isnan(A)] = np.interp(x, xp, fp)
    return A

def GetMatrix(directory, filename, PlotFlag, plot_format):
    ##########################################################
    ## 2020-06-16: Se verificó que la función trabaja
    ## correctamente. Se creara una nueva para hacer pruebas 
    ## con las dimensiones
    ##########################################################
    file_hf5 = directory + os.sep +filename
    hf = h5py.File(file_hf5, 'r')
    #with h5py.File(file_hf5, 'r') as f:
    #    g = f.visit(print)
    '''
    days = np.array(hf['Data/Table Layout/']['day'],dtype=int)
    year = np.array(hf['Data/Table Layout/']['year'],dtype=int)
    month = np.array(hf['Data/Table Layout/']['month'],dtype=int)
    hour = np.array(hf['Data/Table Layout/']['hour'],dtype=int)
    minutes = np.array( hf['Data/Table Layout/']['min'],dtype=int)
    seconds = np.array(hf['Data/Table Layout/']['sec'],dtype=int)
    '''
    rango = hf['Data/Table Layout/']['gdalt']
    #rango2D = hf['Data/Array Layout/']['range']
    #Data/Array Layout/timestamps
    timestamps = hf['Data/Array Layout/']['timestamps']
    #snl =  hf['Data/Table Layout/']['snl']
    #snl2 = hf['Data/Array Layout/2D Parameters/snl']
      
    #vipe1 = hf['Data/Array Layout/2D Parameters/vipe1'] 
    #vipn1 = hf['Data/Array Layout/2D Parameters/vipn2']
    vipe1 = hf['Data/Array Layout/2D Parameters/vipe'] 
    vipn1 = hf['Data/Array Layout/2D Parameters/vipn']
    v_zonal = np.array(vipe1)
    v_vertical = np.array(vipn1)
    #snl2 = np.array(snl2)
    time_vector = []
    date_list = [] # list for datetime objects
    rango = getattr(rango, "tolist", lambda: rango)()
    ###########################################################
    ran_max = max(rango)
    ran_min = min(rango)
    #rang_list = list(rango)
    max_index = rango.index(ran_max)
    min_index = rango.index(ran_min)
    range_diff = np.diff(rango)
    delta_range = range_diff[0]
    delta_ran = delta_range
   #valor constante para todo el arreglo
    MinRange, MaxRange = np.min(rango), ran_max#np.max(rango)
    DataMatrixRows = int((MaxRange-MinRange)/delta_range) + 1
    #DataMatrix = np.ones((DataMatrixRows+1, snl2.shape[0]))*np.nan
    #RowInMatrix = np.array((rango-MinRange)/delta_range+1, dtype=int)
    range_array = np.linspace(MinRange, MaxRange, DataMatrixRows)
    #RangeMatrix = np.ones((DataMatrixRows+1, snl2.shape[0]))*np.nan
    #DataMatrix_v_zonal = np.ones((DataMatrixRows+1, snl2.shape[0]))*np.nan
    #DataMatrix_v_vertical = np.ones((DataMatrixRows+1, snl2.shape[0]))*np.nan
    
    #string_date = timestamps[0]#.strftime('%B %d, %Y, %r')
    #date_time_str = '2018-06-29 08:15:27.243860'
    
    datetime_objects = []
    for ts in timestamps:
        date_time_obj = datetime.datetime.fromtimestamp(ts)
        datetime_objects.append(date_time_obj)
    index = pd.DatetimeIndex(datetime_objects) #- timedelta(hours=5)
    datetime_objects = np.array(datetime_objects)
    h_min_aux=300
    h_max_aux=400
    datetime_objects = np.array(datetime_objects)
    min_ind_aux, max_ind_aux = getRangeIndices(h_min_aux, h_max_aux, range_array)
    #v_vert_avg = np.nanmean(v_vertical[:,min_ind:max_ind],axis=1)
    #v_vert_std = np.nanstd(v_vertical[:,min_ind:max_ind],axis=1)
    #v_vertical[v_vertical>35]=np.nan
    #v_vertical[v_vertical<-35]=np.nan
    v_zonal_avg = np.nanmean(v_zonal[:,min_ind_aux:max_ind_aux],axis=1)
    v_zonal_std = np.nanstd(v_zonal[:,min_ind_aux:max_ind_aux],axis=1)
    #.groupby(lambda x: (x.year, x.month, x.hour, x.minute)).vals.mean()
    vvert_interp = interp_mat(v_vertical)
    #line.split()[0]
    #mes = #string_date.split()[0]
    mes = date_time_obj.month
    month_str = GetMonth(mes)
    dia = date_time_obj.day
    anio = date_time_obj.year
    v_vertical[v_vertical<-25] = np.nan
    v_vertical[v_vertical>25] = np.nan
    p95, p5 = np.percentile(v_vertical, [95, 5])
    #v_vertical[v_vertical>p95] = np.nan
    #v_vertical[v_vertical<p5] = np.nan
    dir_plots = 'Plots-%s-%d' % (month_str, anio)
    print(dir_plots)
    df_zonal = pd.DataFrame({'vz':v_zonal_avg,'std_vz':v_zonal_std})
    #df_zonal.set_index(index,inplace=True)
    #df_zonal_final = df_zonal.resample('30Min').mean()#.interpolate()#["vz"].plot()#interpolate()

    #str_zonal_csv = 'derivas-zonales-30min-promedio-%02d-%02d-%02d.csv' % (year, month,dia)
    #df_zonal_final.to_csv(str_zonal_csv)
    masked_data = np.ma.masked_where(np.isnan(v_vertical),v_vertical)#np.ma.#fill_gaps(datetime_objects, range_array,v_vertical)
    #######################################################################################################
    if (PlotFlag):
        fig, ax = plt.subplots(figsize=(18, 6))
        #plt.rcParams['xtick.labelsize']=14
        
        #plt.style.use('dark_background')
        x_min = mdates.date2num(np.min(index))
        x_max = mdates.date2num(np.max(index))
        extent=[x_min, x_max,150,800]#ran_min,ran_max]
        print("Shapes: ", v_vertical.T.shape, range_array.shape, len(datetime_objects))
        clrs = plt.imshow(v_vertical, cmap='jet',aspect='auto',interpolation='nearest',origin="lower", extent=extent)
        #clrs= ax.pcolormesh(mdates.date2num(datetime_objects), range_array, masked_data, cmap='jet')
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_xlabel("Hora Local (h)", fontsize=16)
        ax.set_ylabel("Rango (km)", fontsize=17)
        fig_title = r'Derivas Verticales ISR JULIA (%d-%02d-%02d)' % (anio, mes, dia) 
        plt.title(fig_title, fontsize=20)
        str_date = '(%d-%02d-%02d)' % (anio, mes, dia)
        ax.set_ylim(250, 700)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
        dt0 = datetime.datetime(index[0].year, index[0].month, index[0].day, 0)
        dtf = datetime.datetime(index[0].year, index[0].month, index[0].day, 0) + pd.Timedelta(hours=24)
        #string_t0 = dt0.strftime("%Y:%m:%d","%H:%M:%S")#'%d-%02d-%02d %02d:00:00' % (year, index[0].month, index[0].day, 0)
        #string_tf = dtf.strftime("%Y:%m:%d","%H:%M:%S")#'%d-%02d-%02d %02d:00:00' % (year, index[0].month, index[0].day+1, 0)
        ax.set_xlim(mdates.date2num(dt0),mdates.date2num(dtf))
        clrs.cmap.set_under('white')
        # This simply sets the x-axis data to diagonal so it fits better.
        fig.autofmt_xdate()
        box=ax.get_position()
        cbarax=fig.add_axes([box.x0+box.width+0.01, box.y0, 0.025, box.height])
        cb=plt.colorbar(clrs,cax=cbarax)
        cb.ax.tick_params(labelsize=17)
        cb.set_label(r'm/s', fontsize=17)
        #cb.set_label(r'Derivas verticales (m/s)')
        #cb2 = fig.colorbar(im2)
       # cb2.set_label(r'$lo      '''y,xt
        year = index[0].year
        mes = index[0].month
        dia = index[0].day
        plt.rcParams['xtick.labelsize']=14
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=0 ,ha='center')
        plt.savefig(r'%s/v-vertical-isr-%d-%02d-%02d.%s' % (dir_plots,year, mes, dia, plot_format),bbox_inches='tight')
        #plt.savefig(r'%s/ecos-150km-%d-%02d-%02d-SNR.png' % (dir_plots,anio, month_prime, dia))
        plt.show()
        plt.close(fig)   
        
    #####################################################################################
  
    return range_array, datetime_objects[:154], v_vertical.T[:154,:] 
#index, range_array, rango, dir_plots, v_zonal, v_vertical.T, timestamps, str_date


def GetDate(year,doy):
    d = datetime.datetime.strptime('{} {}'.format(doy, year),'%j %Y')
    month = d.month
    day = d.day
    return month, day

def RemoveIQR(vx):
    vz = vx.copy()
    q1 = np.nanpercentile(vz,25)
    q3 = np.nanpercentile(vz,75)
    IQR = q3-q1
    #print(q1, q3, IQR)

    upper=q3+1.5*IQR
    lower=q1-1.5*IQR

    vz[vz<lower] = np.nan
    vz[vz>upper] = np.nan    
    return vz

def GetStandardDriftMatrix(directory,filename, plot_format, PlotFlag):
    file_hf5 = directory +os.sep +filename
    hf = h5py.File(file_hf5, 'r')
    alturas = hf['Data/Table Layout/']['gdalt']
    range_file = np.array(hf['Data/Array Layout/']['range'])
    timestamps = hf['Data/Array Layout/']['timestamps']
    vipe1 = hf['Data/Array Layout/2D Parameters/vipe'] 
    vipn1 = hf['Data/Array Layout/2D Parameters/vipn']
    #rango2D = np.array(hf['Data/Array Layout/2D Parameters/range'])
    v_zonal = np.array(vipe1)
    v_vertical = np.array(vipn1)#.T
    v_vertical[v_vertical>120]=np.nan
    alturas = np.array(getattr(alturas, "tolist", lambda: alturas)())
    ###########################################################
    ran_max = np.max(range_file)
    ran_min = np.min(range_file)
    rango = list(range_file)
    range_diff = np.diff(range_file)
    delta_range = range_diff[0]
    ranNum = int((ran_max-ran_min)/delta_range) + 1
    range_file = np.arange(ran_min,ran_max+delta_range,delta_range)#np.linspace(ran_min, ran_max, ranNum)
    datetime_objects = []
    for ts in timestamps:
        date_time_obj = datetime.datetime.fromtimestamp(ts)
        datetime_objects.append(date_time_obj)
    index = pd.DatetimeIndex(datetime_objects) #- timedelta(hours=5)
    ###Creando arreglo de datetime objetcs:
    dt0 = datetime.datetime(index[0].year, index[0].month, index[0].day,0,0,0)
    dtf = dt0+pd.Timedelta(hours=24)
    ntimes = 288
    dt_list = []
    dt_array = np.arange(dt0,dtf,timedelta(seconds=300))#np.array(dt_list)
    delta_time_array = np.diff(index)[0]/ np.timedelta64(1, 's')#(dtf-dt0).total_seconds()/ntimes#/np.timedelta64(1, 's')#np.linspace(dt0,dtf,timedelta(seconds=300))
    print("delta_time_array: ", delta_time_array)
    #dt_indices = (datetime_objects-dt_array)/delta_time_array
    ###################################################################
    ## Arreglos Estandarizados##########################
    MinRange, MaxRange = np.min(rango),np.max(rango)
    DataMatrixRows = int((MaxRange-MinRange)/delta_range) #+ 1 #
    DataMatrix = np.ones((DataMatrixRows, ntimes))*np.nan
    RowInMatrix = np.array((range_file-MinRange)/delta_range, dtype=int)
    range_array = np.linspace(MinRange,MaxRange,68)#np.arange(MinRange,MaxRange+delta_range,delta_range)
    ###################################################################
    col = 0 #counter for current columns
    PastRow = 0 #saving past row index
    month = date_time_obj.month
    #month_prime = 12
    mes = GetMonth(month)
    dia = date_time_obj.day
    anio = date_time_obj.year

    dir_plots = 'Plots-%s-%d' % (GetMonth(month),anio)#'Plots-150km-%s-%d' % (mes, anio)
    print("Shapes: ", v_zonal.T.shape, range_file.shape)
    print("Shapes: ", v_vertical.T.shape, range_array.shape, len(datetime_objects))
    print("Shapes: ", DataMatrix.shape, range_file.shape, dt_array.shape)
    print("ran_min, ran_max: ", ran_min, ran_max)
    
    #######################################################################################################
    print(range_array[-1])
    print(range_file[-1])
    #print(rango2D[-1])
    #print(rango[-1])
    diff = list(np.diff(range_file).flatten())
    #print(diff.count(14), diff.count(15), len(diff))
    
    datetime_objects = np.array(datetime_objects)
    RowInMatrixTime = np.array((datetime_objects-dt0)/timedelta(seconds=300),dtype=int)
    #DataMat = np.ones((ntimes,DataMatrixRows))*np.nan
    ncols = 64#68#100#58#64
    DataMat = np.ones((ntimes,ncols))*np.nan
    col = 0 #counter for current columns
    #'''
    range_indices = np.array((range_file - MinRange)/delta_range,dtype=int)
    #print(range_indices)
    for i in range(0,datetime_objects.shape[0]):
        row = RowInMatrixTime[i]
        for j in range_indices:
            DataMat[row,j] = v_vertical[i,j]
    #DataMat = v_vertical.copy()
    #'''
    DataMat = RemoveIQR(DataMat)
    if (PlotFlag):
        fig, ax = plt.subplots(figsize=(12, 6))
    #plt.rcParams['xtick.labelsize']=14
        #plt.style.use('dark_background')
        clrs= ax.pcolormesh(mdates.date2num(dt_array), range_array, DataMat, cmap=plt.cm.RdBu_r)
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_xlabel("Hora Local (h)", fontsize=16)
        ax.set_ylabel("Rango (km)", fontsize=17)
        fig_title = r'Derivas Verticales ISR (%d-%02d-%02d)' % (anio, month, dia) 
        plt.title(fig_title, fontsize=15)
        str_date = '(%d-%02d-%02d)' % (anio, month, dia)
        #ax.set_ylim(300, 1200)
        # This simply sets the x-axis data to diagonal so it fits better.
        fig.autofmt_xdate()
        box=ax.get_position()
        cbarax=fig.add_axes([box.x0+box.width+0.01, box.y0, 0.025, box.height])
        cb=plt.colorbar(clrs,cax=cbarax)
        cb.set_label(r'Derivas verticales (m/s)')
        plt.savefig(r'%s/Drifts-ISR-%d-%02d-%02d-v-zonal.%s' % (dir_plots,anio, month, dia, plot_format))
        plt.show()
        plt.close(fig)   
    #####################################################################################
    
    
    return DataMat, range_array, dt_array

def AddNaN(a,b):
    result = np.add(a,b)

    a_is_nan = np.isnan(a)
    b_is_nan = np.isnan(b)

    result_is_nan = np.isnan(result)

    mask_a = np.logical_and(result_is_nan, np.logical_not(a_is_nan))
    result[mask_a] = a[mask_a]

    mask_b = np.logical_and(result_is_nan, np.logical_not(b_is_nan))
    result[mask_b] = b[mask_b]

    return result

def Diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

def GetTimeRangeArrays(year, month, day, hour_i, hour_f, h_min, h_max, delta_ran, intFactor):
    range_fixed = np.arange(h_min,h_max+delta_ran,delta_ran)
    string_t0 = '%d-%02d-%02d %02d:00:00' % (year, month, day, hour_i)
    #string_tf = '%d-%02d-%02d %02d:00:00' % (year, month, day, hour_f)
    dt0 = datetime.datetime.strptime(string_t0, '%Y-%m-%d %H:%M:%S')
    dtf = dt0+timedelta(hours=24)#datetime.datetime.strptime(string_tf, '%Y-%m-%d %H:%M:%S')# + timedelta(days=1)
#ax.set_xlim(mdates.date2num(dt0),mdates.date2num(dtf))
    time_range = np.arange(dt0,dtf,timedelta(seconds=intFactor*60)).astype(datetime.datetime)
    print(time_range[1]-time_range[0])
    print("time_range.shape: ", time_range.shape)
    #print("Type: ", type(time_range[0]))
    #time_range = np.array(time_range).astype(datetime.datetime)#,dtype='datetime64[s]')
    return time_range, range_fixed


def GetMonth(month):
    
    str_month = ['Enero','Febrero','Marzo','Abril','Mayo', 'Junio','Julio','Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre','Noviembre-Diciembre']
    return str_month[month-1]


def GetIndices(Data):
    YYYY = Data['Year']
    DD = Data['DayOfYear']
    hh = Data['Hour']
    fechas = []
    for Y, D, h in zip(YYYY, DD, hh):
        str_temp = '%d-%02d' % (Y,D) 
        fecha = datetime.datetime.strptime(str_temp,'%Y-%j')
        M = fecha.month
        D = fecha.day
        fecha = datetime.datetime(Y, M, D, h, 0, 0)
        fechas.append(fecha)
    dates = pd.to_datetime(fechas)

    return dates

def bspl4_long(il,xl):
    order=4
    b=np.zeros((20,20))#FLTARR(20,20)
#----------------------------------------------------------------------------  
    t_l=[0,10,100,190,200,250,280,310,360,370,460,550,560,610,640,670,\
         720,730,820,910,920,970,1000,1030,1080]
#----------------------------------------------------------------------------      
    i=il     
    x=xl
#---------------------------------------------------------------------------- 
#B-Spline
    if(i-1 >= 0):
        if (x < t_l[i-1]):
            x=x+360
    for j in range(i,i+order-1 + 1):
        if (x >= t_l[j]) and x < t_l[j+1]:
            b[j,1] = 1
        else:
            b[j,1] = 0
    for j in range(2,order + 1):
        for k in range(i,i+order-j + 1):
            b[k,j]=(x-t_l[k])/(t_l[k+j-1]-t_l[k])*b[k,j-1]
            b[k,j]=b[k,j]+(t_l[k+j]-x)/(t_l[k+j]-t_l[k+1])*b[k+1,j-1]
    bspl4_long=b[i,order]
    return bspl4_long
################################################################################################

'''
;  NAME:bspl4_time
;  PURPOSE:
;    This function has a routine for obtaining interpolations.
;  INPUT:
;    xt= Solar Local Time
;    order= order by interpalation
;    i = A integer from 1 to 13
;  OUTPUT:
;    bspl4_time
'''
def bspl4_time(i,xt):
    order=4
    b = np.zeros((20,20))
    t_t=[ 0.00,2.75,4.75,5.50,6.25,\
    7.25,10.00,14.00,17.25,18.00,\
    18.75,19.75,21.00,24.00,26.75,\
    28.75,29.50,30.25,31.25,34.00,\
    38.00,41.25,42.00,42.75,43.75,\
    45.00,48.00,50.75,52.75,53.50,\
    54.25,55.25,58.00,62.00,65.25,\
    66.00,66.75,67.75,69.00,72.00]
    x=xt
    #'''
    if (i-1 >= 0):
        if (x < t_t[i-1]):
            x=x+24
    for j in range(i,i+order-1 + 1):
        #print("Valor de j: ", j)
        if (x >= t_t[j] and x < t_t[j+1]):
            #print("Condicional 1")
            b[j,1] = 1
        else:
            b[j,1] = 0
            #print("Condicional 2")
    for j in range(2,order + 1):
        for k in range(i,i+order-j + 1):
            b[k,j]=(x-t_t[k])/(t_t[k+j-1]-t_t[k])*b[k,j-1]
            b[k,j]=b[k,j]+(t_t[k+j]-x)/(t_t[k+j]-t_t[k+1])*b[k+1,j-1]
            #print("Time splines: ", b[k,j])
    bspl4_time=b[i,order]

    return bspl4_time
    #'''
#############################################################################

#;  INPUTS:
#;    xt = Solar Local Time
#;    xl = Geographic Longitude
#;
#;  OUTPUTS
#;    vdrift = vertical drifts
#;
#;  FUNCTION CALLED
#;    bspl4_time,bspl4_long
#;

def vdrift(xt,xl,doy,f107cm):

    coeff=[-10.80592, -9.63722,-11.52666, -0.05716, -0.06288,  0.03564,\
    -5.80962, -7.86988, -8.50888, -0.05194, -0.05798, -0.00138,\
    2.09876,-19.99896, -5.11393, -0.05370, -0.06585,  0.03171,\
    -10.22653, -3.62499,-14.85924, -0.04023, -0.01190, -0.09656,\
    -4.85180,-26.26264, -6.20501, -0.05342, -0.05174,  0.02419,\
    -13.98936,-18.10416, -9.30503, -0.01969, -0.03132, -0.01984,\
    -18.36633,-24.44898,-16.69001,  0.02033, -0.03414, -0.02062,\
    -20.27621,-16.95623,-36.58234,  0.01445, -0.02044, -0.08297,\
    1.44450,  5.53004,  4.55166, -0.02356, -0.04267,  0.05023,\
    5.50589,  7.05381,  1.94387, -0.03147, -0.03548,  0.01166,\
    3.24165, 10.05002,  4.26218, -0.03419, -0.02651,  0.07456,\
    7.02218,  0.06708,-11.31012, -0.03252, -0.01021, -0.09008,\
    -3.47588, -2.82534, -4.17668, -0.03719, -0.01519,  0.06507,\
    -4.02607,-11.19563,-10.52923, -0.00592, -0.01286, -0.00477,\
    -11.47478, -9.57758,-10.36887,  0.04555, -0.02249,  0.00528,\
    -14.19283,  7.86422, -8.76821,  0.05758, -0.02398, -0.04075,\
    14.58890, 36.63322, 27.57497,  0.01358, -0.02316,  0.04723,\
    12.53122, 29.38367, 21.40356, -0.00071, -0.00553,  0.01484,\
    18.64421, 26.27327, 18.32704,  0.00578,  0.03349,  0.11249,\
    4.53014,  6.15099,  7.41935, -0.02860, -0.00395, -0.08394,\
    14.29422,  9.77569,  2.85689, -0.00107,  0.04263,  0.10739,\
    7.17246,  4.40242, -1.00794,  0.00089,  0.01436,  0.00626,\
    7.75487,  5.01928,  4.36908,  0.03952, -0.00614,  0.03039,\
    10.25556,  8.82631, 24.21745,  0.05492, -0.02968,  0.00177,\
    21.86648, 24.03218, 39.82008,  0.00490, -0.01281, -0.01715,\
    19.18547, 23.97403, 34.44242,  0.01978,  0.01564, -0.02434,\
    26.30614, 14.22662, 31.16844,  0.06495,  0.19590,  0.05631,\
    21.09354, 25.56253, 29.91629, -0.04397, -0.08079, -0.07903,\
    28.30202, 16.80567, 38.63945,  0.05864,  0.16407,  0.07622,\
    22.68528, 25.91119, 40.45979, -0.03185, -0.01039, -0.01206,\
    31.98703, 24.46271, 38.13028, -0.08738, -0.00280,  0.01322,\
    46.67387, 16.80171, 22.77190, -0.13643, -0.05277, -0.01982,\
    13.87476, 20.52521,  5.22899,  0.00485, -0.04357,  0.09970,\
    21.46928, 13.55871, 10.23772, -0.04457,  0.01307,  0.06589,\
    16.18181, 16.02960,  9.28661, -0.01225,  0.14623, -0.01570,\
    18.16289, -1.58230, 14.54986, -0.00375, -0.00087,  0.04991,\
    10.00292, 11.82653,  0.44417, -0.00768,  0.15940, -0.01775,\
    12.15362,  5.65843, -1.94855, -0.00689,  0.03851,  0.04851,\
    -1.25167,  9.05439,  0.74164,  0.01065,  0.03153,  0.02433,\
    -15.46799, 18.23132, 27.45320,  0.00899, -0.00017,  0.03385,\
    2.70396, -0.87077,  6.11476, -0.00081,  0.05167, -0.08932,\
    3.21321, -1.06622,  5.43623,  0.01942,  0.05449, -0.03084,\
    17.79267, -3.44694,  7.10702,  0.04734, -0.00945,  0.11516,\
    0.46435,  6.78467,  4.27231, -0.02122,  0.10922, -0.03331,\
    15.31708,  1.70927,  7.99584,  0.07462,  0.07515,  0.08934,\
    4.19893,  6.01231,  8.04861,  0.04023,  0.14767, -0.04308,\
    9.97541,  5.99412,  5.93588,  0.06611,  0.12144, -0.02124,\
    13.02837, 10.29950, -4.86200,  0.04521,  0.10715, -0.05465,\
    5.26779,  7.09019,  1.76617,  0.09339,  0.22256,  0.09222,\
    9.17810,  5.27558,  5.45022,  0.14749,  0.11616,  0.10418,\
    9.26391,  4.19982, 12.66250,  0.11334,  0.02532,  0.18919,\
    13.18695,  6.06564, 11.87835,  0.26347,  0.02858,  0.14801,\
    10.08476,  6.14899, 17.62618,  0.09331,  0.08832,  0.28208,\
    10.75302,  7.09244, 13.90643,  0.09556,  0.16652,  0.22751,\
    6.70338, 11.97698, 18.51413,  0.15873,  0.18936,  0.15705,\
    5.68102, 23.81606, 20.65174,  0.19930,  0.15645,  0.08151,\
    29.61644,  5.49433, 48.90934,  0.70710,  0.40791,  0.26325,\
    17.11994, 19.65380, 44.88810,  0.45510,  0.41689,  0.22398,\
    8.45700, 34.54442, 27.25364,  0.40867,  0.37223,  0.22374,\
    -2.30305, 32.00660, 47.75799,  0.02178,  0.43626,  0.30187,\
    8.98134, 33.01820, 33.09674,  0.33703,  0.33242,  0.41156,\
    14.27619, 20.70858, 50.10005,  0.30115,  0.32570,  0.45061,\
    14.44685, 16.14272, 45.40065,  0.37552,  0.31419,  0.30129,\
    6.19718, 18.89559, 28.24927,  0.08864,  0.41627,  0.19993,\
    7.70847, -2.36281,-21.41381,  0.13766,  0.05113, -0.11631,\
    -9.07236,  3.76797,-20.49962,  0.03343,  0.08630,  0.00188,\
    -8.58113,  5.06009, -6.23262,  0.04967,  0.03334,  0.24214,\
    -27.85742,  8.34615,-27.72532, -0.08935,  0.15905, -0.03655,\
    2.77234,  0.14626, -4.01786,  0.22338, -0.04478,  0.18650,\
    5.61364, -3.82235,-16.72282,  0.26456, -0.03119, -0.08376,\
    13.35847, -6.11518,-16.50327,  0.28957, -0.01345, -0.19223,\
    -5.37290, -0.09562,-27.27889,  0.00266,  0.22823, -0.35585,\
    -15.29676,-18.36622,-24.62948, -0.31299, -0.23832, -0.08463,\
    -23.37099,-13.69954,-26.71177, -0.19654, -0.18522, -0.20679,\
    -26.33762,-15.96657,-42.51953, -0.13575, -0.00329, -0.28355,\
    -25.42140,-14.14291,-21.91748, -0.20960, -0.19176, -0.32593,\
    -23.36042,-23.89895,-46.05270, -0.10336,  0.03030, -0.21839,\
    -19.46259,-21.27918,-32.38143, -0.17673, -0.15484, -0.11226,\
    -19.06169,-21.13240,-34.01677, -0.25497, -0.16878, -0.11004,\
    -18.39463,-16.11516,-19.55804, -0.19834, -0.23271, -0.25699,\
    -19.93482,-17.56433,-18.58818,  0.06508, -0.18075,  0.02796,\
    -23.64078,-18.77269,-22.77715, -0.02456, -0.12238,  0.02959,\
    -12.44508,-21.06941,-19.36011,  0.02746, -0.16329,  0.19792,\
    -26.34187,-19.78854,-24.06651, -0.07299, -0.03082, -0.03535,\
    -10.71667,-26.04401,-16.59048,  0.02850, -0.09680,  0.15143,\
    -18.40481,-23.37770,-16.31450, -0.03989, -0.00729, -0.01688,\
    -9.68886,-20.59304,-18.46657,  0.01092, -0.07901,  0.03422,\
    -0.06685,-19.24590,-29.35494,  0.12265, -0.24792,  0.05978,\
    -15.32341, -9.07320,-13.76101, -0.17018, -0.15122, -0.06144,\
    -14.68939,-14.82251,-13.65846, -0.11173, -0.14410, -0.07133,\
    -18.38628,-18.94631,-19.00893, -0.08062, -0.14481, -0.12949,\
    -16.15328,-17.40999,-14.08705, -0.08485, -0.06896, -0.11583,\
    -14.50295,-16.91671,-25.25793, -0.06814, -0.13727, -0.12213,\
    -10.92188,-14.10852,-24.43877, -0.09375, -0.11638, -0.09053,\
    -11.64716,-14.92020,-19.99063, -0.14792, -0.08681, -0.12085,\
    -24.09766,-16.14519, -8.05683, -0.24065, -0.05877, -0.23726,\
    -25.18396,-15.02034,-15.50531, -0.12236, -0.09610, -0.00529,\
    -15.27905,-19.36708,-12.94046, -0.08571, -0.09560, -0.03544,\
    -7.48927,-16.00753,-13.02842, -0.07862, -0.10110, -0.05807,\
    -13.06383,-27.98698,-18.80004, -0.05875, -0.03737, -0.11214,\
    -13.67370,-16.44925,-16.12632, -0.07228, -0.09322, -0.05652,\
    -22.61245,-21.24717,-18.09933, -0.05197, -0.07477, -0.05235,\
    -27.09189,-21.85181,-20.34676, -0.05123, -0.05683, -0.07214,\
    -27.09561,-22.76383,-25.41151, -0.10272, -0.02058, -0.16720]
#;--------------------------------------------------------------------
    param = np.zeros(2)#FLTARR(2)
    funct = np.zeros(6)#FLTARR(6)
    gauss = 0
    cflux = 0

#;----------------------------------------------------------------------
    param[0]=doy
    param[1]=f107cm
    flux=param[1]
    #IF(param(1) LE 75) THEN flux=75.
    if (param[1] <= 75):
        flux = 75
    #IF(param(1) GE 230) THEN flux=230.
    if param[1] >= 230:
        flux = 230
    cflux = flux
    a=0.
    if (param[0] >= 120 and param[0]<= 240):
        a = 170
        sigma = 60
    if (param[0] <= 60 or param[0]>= 300):
        a = 170
        sigma = 40

    if (flux <= 95 and a!=0):
        gauss=np.exp(-0.5*((xl-a)**2)/sigma**2)
        cflux=gauss*95.+(1-gauss)*flux

    if param[0] >= 135 and param[0] <= 230:
        funct[0]=1
    if param[0] <= 45 or param[0] >= 320:
        funct[1]=1
    if param[0] > 75 and param[0] < 105:
        funct[2]=1
    if param[0] > 260 and param[0] < 290:
        funct[2]=1

    if param[0] >= 45 and param[0] <= 75: #W-E
        funct[1] = 1 - (param[0]-45.)/30.
        funct[2] = 1 - funct[1]

    if param[0] >= 105 and param[0] <= 135: #E-S
        funct[2] = 1 - (param[0]-105.)/30.
        funct[0] = 1 - funct[2]

    if param[0] >= 230 and param[0] <= 260: #S-E
        funct[0] = 1 - (param[0]-230.)/30.
        funct[2] = 1 - funct[0]

    if param[0] >= 290 and param[0] <= 320: # E-W
        funct[0] = 1 - (param[0]-290.)/30.
        funct[1] = 1 - funct[2]
    funct[3] = (cflux-140.0)*funct[0]
    funct[4] = (cflux-140.0)*funct[1]
    funct[5] = (cflux-140.0)*funct[2]

    index_t=13
    index_l=8
    nfunc=6
#;--------------------------------------------------------------------
    y=0.
    for i in range(1,index_t+1):
        for il in range(1, index_l+1):
            kk = index_l*(i-1)+il
            for j in range(1,nfunc+1):
                ind = nfunc*(kk-1)+j
                bspl4 = float(bspl4_time(i,xt))*float(bspl4_long(il,xl))
                y=y+bspl4*funct[j-1]*coeff[ind-1]
                #print("i,il, kk, j: ", i,il, kk, j)
                #print(bspl4_time(i,xt),bspl4_long(il,xl),funct[j-1],coeff[ind-1],y)


    return y

def getF107(month,year):
    dir_path = "geomagnetic_indices"
    filename = "hourly-geomagnetic-indices-%02d-%d.txt" % (month, year)
    filename = './' + dir_path + '/' + filename
    #current_month = 'Noviembre-2020'
    #print(filename)
    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    current_month = '%s-%d' % (meses[month-1], year)
    dir_plots = 'Plots-%s/' % current_month
    Data = pd.read_csv(filename,delimiter=r"\s+")#
    indices = GetIndices(Data)
    Data.set_index(indices)
    #Data.head()
    return Data

def getIndexF107(year, doy, hour):
    #month=3
    d = datetime.datetime.strptime('{} {}'.format(doy, year),'%j %Y')
    month = d.month
    F107 = getF107(month,year)
    F107.tail()
    a = F107.loc[(F107['Hour'] == hour) & (F107['DayOfYear'] == doy), 'F107']
    #a.apply(lambda x: float(x))
    #print(doy,hour,a)
    #a = 150
    return a.values#[0]#float(a)

def drift_model(year, doy,longitude):
    if doy == 0:
        doy = 161
    if longitude ==0:
        longitude = 270.0
    #constant = longitude
    start = 0.0
    finish = 24.0
    step = 0.5
    #nsteps = 49
    #step = int(finish-start)/nsteps
    ###############################################################
    st = 0.0
    end = 24.0
    nsteps = 49
    a = np.linspace(start,end,nsteps, endpoint=False)
    minutos = []
    horas = []
    for t in a:
        hour, minute = divmod(t, 1)
        minute *= 60
        result = '{}:{}'.format(int(hour), int(minute))
        #print(result)
        horas.append(int(hour))
        minutos.append(int(minute))
    ###################

    d = datetime.datetime.strptime('{} {}'.format(int(doy), year),'%j %Y')
    ###################
    m = np.zeros(nsteps)
    steps = np.zeros(nsteps)
    #FOR i=start,(finish*(1+(profil EQ 0))) DO steps[i]=i*step
    j=0
    f107_list = []#getIndexF107(year, doy, hour)
    datetime_list = []
    for i in np.linspace(start,finish,nsteps,endpoint=False):
        steps[j]=i*step
        temp = i#start + i*step
        steps[j] = temp
        str_time = '%d-%02d-%02d %02d:%02d:%02d' % (year, d.month, d.day,horas[j], minutos[j],0)
        dt = datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
        datetime_list.append(dt)
        #print('i:', int(i))
        f107_list.append(getIndexF107(year, doy, int(i)))
        j = j + 1
    time = steps
    yv = []
    #Calling VDrifts to compute vertical drifts.
    for ii in range(0,(nsteps-1) + 1):
        xt=time[ii]
        xl=longitude
        y=vdrift(xt,xl,doy,f107_list[ii])

        yv.append(y)
    return yv,datetime_list
