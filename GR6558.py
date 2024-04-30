#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:53:17 2024

@author: georgegyabaah
"""

import pandas as pd
import numpy as np
from scipy import stats as st
from scipy.integrate import quad
import scipy.special as special
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib import cm
import xarray as xr
import pymannkendall as mk
import cartopy.crs as ccrs 
from cartopy import feature as cfeature
import scipy.io as sio
import os
from scipy.io import loadmat
import xarray as xr
import glob
from scipy.interpolate import griddata
from scipy import stats as sp
import matplotlib as mpl
from scipy import stats as sp
from scipy.stats import linregress


#Final project codes

#TCGI  codes : Note this code is for only historical. The other experiments are changed from the file paths
#TCGI  codes : Note this code is for only historical. The other experiments are changed from the file paths
dfhc=xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/CESM2/r3i1p1f1/mon/TCGI_CRH_Amon_CESM2_historical_r3i1p1f1_1950-2014_r180x90.nc')
dfhc=dfhc.resample(time='1AS').mean('time')
dfhc['time']=np.arange(1950,2015)
dfhmr=xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/MRI-ESM2-0/r1i1p1f1/mon/TCGI_CRH_Amon_MRI-ESM2-0_historical_r1i1p1f1_1950-2014_r180x90.nc')
pd.date_range(start='1950-12-01',periods=65, freq='AS-DEC')
dfhmr=dfhmr.resample(time='1AS').mean('time')
dfhmr['time']=np.arange(1950,2015)
dfhmi=xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/MIROC6/r1i1p1f1/mon/TCGI_CRH_Amon_MIROC6_historical_r1i1p1f1_1950-2014_r180x90.nc')
pd.date_range(start='1950-12-01',periods=65, freq='AS-DEC')
dfhmi=dfhmi.resample(time='1AS').mean('time')
dfhmi['time']=np.arange(1950,2015)
dfhma1=xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/ACCESS-CM2/r2i1p1f1/mon/TCGI_CRH_Amon_ACCESS-CM2_historical_r2i1p1f1_1950-2014_r180x90.nc')
pd.date_range(start='1950-12-01',periods=65, freq='AS-DEC')
dfhma1=dfhma1.resample(time='1AS').mean('time')
dfhma1['time']=np.arange(1950,2015)
dfhma2=xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/ACCESS-ESM1-5/r1i1p1f1/mon/TCGI_CRH_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_1950-2014_r180x90.nc')
pd.date_range(start='1950-12-01',periods=65, freq='AS-DEC')
dfhma2=dfhma2.resample(time='1AS').mean('time')
dfhma2['time']=np.arange(1950,2015)
dfhcan=xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/CanESM5/r1i1p1f1/mon/TCGI_SCRH_Amon_CanESM5_historical_r1i1p1f1_1950-2014_r180x90.nc')
pd.date_range(start='1950-12-01',periods=65, freq='AS-DEC')
dfhcan=dfhcan.resample(time='1AS').mean('time')
dfhcan['time']=np.arange(1950,2015)
dfhcnr=xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/CNRM-CM6-1/r1i1p1f2/mon/TCGI_CRH_Amon_CNRM-CM6-1_historical_r1i1p1f2_1950-2014_r180x90.nc')
pd.date_range(start='1950-12-01',periods=65, freq='AS-DEC')
dfhcnr=dfhcnr.resample(time='1AS').mean('time')
dfhcnr['time']=np.arange(1950,2015)
dfhfg=xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/FGOALS-g3/r1i1p1f1/mon/TCGI_CRH_Amon_FGOALS-g3_historical_r1i1p1f1_1950-2014_r180x90.nc')
pd.date_range(start='1950-12-01',periods=65, freq='AS-DEC')
dfhfg=dfhfg.resample(time='1AS').mean('time')
dfhfg['time']=np.arange(1950,2015)
dfhhad=xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/HadGEM3-GC31-LL/r1i1p1f3/mon/TCGI_CRH_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_1950-2014_r180x90.nc')
pd.date_range(start='1950-12-01',periods=65, freq='AS-DEC')
dfhhad=dfhhad.resample(time='1AS').mean('time')
dfhhad['time']=np.arange(1950,2015)
dfhip=xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/IPSL-CM6A-LR/r8i1p1f1/mon/TCGI_CRH_Amon_IPSL-CM6A-LR_historical_r8i1p1f1_1950-2014_r180x90.nc')
pd.date_range(start='1950-12-01',periods=65, freq='AS-DEC')
dfhip=dfhip.resample(time='1AS').mean('time')
dfhip['time']=np.arange(1950,2015)
dfhesh=xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/E3SM-2-0/r1i1p1f1/mon/TCGI_CRH_Amon_E3SM-2-0_historical_r1i1p1f1_1950-2014_r180x90.nc')
pd.date_range(start='1950-12-01',periods=65, freq='AS-DEC')
dfhesh=dfhesh.resample(time='1AS').mean('time')
dfhesh['time']=np.arange(1950,2015)
add_h=(dfhc+dfhmr+dfhma1+dfhmi+dfhcan+dfhcnr+dfhfg+dfhhad+dfhip+dfhma2+dfhesh)/11


#This part calculates the statistical significance on the TCGI plots


def ztest(a,b):
    zstat = ((a.mean('time') - b.mean('time')) - 0)/(np.sqrt((a.var('time')/len(a))+(b.var('time')/len(b))))
    pvalue = sp.norm.sf(abs(zstat))*2
    return(pvalue)
def create_xr(pval):
    ds=xr.Dataset(

        {

            "PValue": (("lat", "lon"), pval),

        },

        coords={

            "lat": ("lat", hist.lat.data),

            "lon": ("lon", hist.lon.data),

        },

    )

    ds = ds['PValue']

    return(ds)

#   This is the plotting section


cmap = plt.cm.jet
cmap.set_under('white')

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines(resolution='110m')

land_mask = cfeature.NaturalEarthFeature('physical', 'land', '110m')
ax.add_feature(land_mask, facecolor='lightgray')
just_h=add_h.TCGI.mean('time').sel(lon=slice(0,358),lat=slice(-89,89))
#gl = ax.gridlines()
vmin = 0.00
vmax = 0.013
levels = np.linspace(vmin, vmax, 20)
# Plot the TCGIs with the modified colormap and levels
tcgi_plot = ax.contourf(just_h.lon, just_h.lat, just_h, cmap='jet', levels=(levels), extend='both', transform=ccrs.PlateCarree())
#plt.contour(just_a.lon,just_a.lat,-1 * add_h-180,levels=(9))
mpl.rc('hatch',color='black',lw=1)
ax.contourf(just_h.lon,just_h.lat+5,PVal,cmap='jet',levels=[0,0.05,999],transform=ccrs.PlateCarree(),hatches=['//',''],alpha=0)

cbar = plt.colorbar(tcgi_plot,pad=0.001, shrink=0.3810)
cbar.set_label('TCGI/year', fontsize=11)
title = ax.set_title('TCGI mean HIST (1950-2014)', fontsize=10, loc='left')
title = ax.set_title('#', fontsize=7, loc='right')
title.set_y(1.02)  # Shift the title upwards
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

ax.set_xticks(range(0, 361, 30), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(plt.FixedFormatter(['0°E', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W', '0°E']))
ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
ax.yaxis.set_major_formatter(plt.FixedFormatter(['90°S', '60°S', '30°S', '0°', '30°N', '60°N', '90°N']))
land_mask = cfeature.NaturalEarthFeature('physical', 'land', '110m')
ax.add_feature(land_mask, facecolor='lightgray')
ax.coastlines()
ax.set_ylim(-30,45)
plt.show()







#Predictors codes : Note this code is for only historical. The other experiments are changed from the file paths

dvt1 = xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/MRI-ESM2-0/r1i1p1f1/mon/Vort_Amon_MRI-ESM2-0_historical_r1i1p1f1_1950-2014_r180x90.nc')
dvt2 = xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/CESM2/r3i1p1f1/mon/Vort_Amon_CESM2_historical_r3i1p1f1_1950-2014_r180x90.nc')
dvt3 = xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/CanESM5/r1i1p1f1/mon/Vort_Amon_CanESM5_historical_r1i1p1f1_1950-2014_r180x90.nc')
dvt4 = xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/FGOALS-g3/r1i1p1f1/mon/Vort_Amon_FGOALS-g3_historical_r1i1p1f1_1950-2014_r180x90.nc')
dvt5 = xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/ACCESS-CM2/r2i1p1f1/mon/Vort_Amon_ACCESS-CM2_historical_r2i1p1f1_1950-2014_r180x90.nc')
dvt6 = xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/ACCESS-ESM1-5/r1i1p1f1/mon/Vort_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_1950-2014_r180x90.nc')
dvt7 = xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/CNRM-CM6-1/r1i1p1f2/mon/Vort_Amon_CNRM-CM6-1_historical_r1i1p1f2_1950-2014_r180x90.nc')
dvt8 = xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/MIROC6/r1i1p1f1/mon/Vort_Amon_MIROC6_historical_r1i1p1f1_1950-2014_r180x90.nc')
dvt9 = xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/HadGEM3-GC31-LL/r1i1p1f3/mon/Vort_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_1950-2014_r180x90.nc')
dvt10 = xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/IPSL-CM6A-LR/r8i1p1f1/mon/Vort_Amon_IPSL-CM6A-LR_historical_r8i1p1f1_1950-2014_r180x90.nc')
dvt11 = xr.open_dataset('/CMIP_Databank/CMIP6/regrid/historical/E3SM-2-0/r1i1p1f1/mon/Vort_Amon_E3SM-2-0_historical_r1i1p1f1_1950-2014_r180x90.nc')
avt=dvt1['avort'].resample(time='YS').mean('time').mean('time')
bvt=dvt2['avort'].resample(time='YS').mean('time').mean('time')
cvt=dvt3['avort'].resample(time='YS').mean('time').mean('time')
dvt=dvt4['avort'].resample(time='YS').mean('time').mean('time')
evt=dvt5['avort'].resample(time='YS').mean('time').mean('time')
fvt=dvt6['avort'].resample(time='YS').mean('time').mean('time')
gvt=dvt7['avort'].resample(time='YS').mean('time').mean('time')
hvt=dvt8['avort'].resample(time='YS').mean('time').mean('time')
ivt=dvt9['avort'].resample(time='YS').mean('time').mean('time')
jvt=dvt10['avort'].resample(time='YS').mean('time').mean('time')
kvt=dvt11['avort'].resample(time='YS').mean('time').mean('time')
hsum=(avt+bvt+cvt+dvt+evt+fvt+gvt+hvt+ivt+jvt+kvt)/11



data = [avoth/11, avotg/11, avota/11]  
cmap = plt.cm.jet
cmap.set_under('white')
fig, axs = plt.subplots(3, 1, figsize=(15, 24), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
titles = ['aVORT mean difference (Historical - Hist-Nat)', 'aVORT mean difference (Hist-GHG - Hist-Nat)', 'aVORT mean difference (Hist-Aer - Hist-Nat)']  # Replace these with your actual titles
for i, ax in enumerate(axs):
    ax.coastlines(resolution='110m')
    ax.set_title(titles[i])
    land_mask = cfeature.NaturalEarthFeature('physical', 'land', '110m')
    just_h = data[i].sel(lon=slice(270, 358), lat=slice(0, 60))  
    vmin = -0.000001
    vmax = 0.000001
    levels = np.linspace(vmin, vmax, 20)
    extent = [270, 358, 0, 40]
    tcgi_plot = ax.contourf(just_h.lon, just_h.lat, just_h, cmap='bwr', levels=levels, extend='both', transform=ccrs.PlateCarree())
    mpl.rc('hatch', color='black', lw=1)
    cbar = plt.colorbar(tcgi_plot, ax=ax, pad=0.001, shrink=1)
    title = ax.set_title('$ms^-1$', fontsize=7, loc='right')
    title.set_y(1.02) 
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    land_mask = cfeature.NaturalEarthFeature('physical', 'land', '110m')
    ax.add_feature(land_mask, facecolor='lightgray')
    ax.set_xticks(range(270, 358, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(plt.FixedFormatter(['90°W', '60°W', '30°W']))
    ax.set_yticks(range(0, 60, 20), crs=ccrs.PlateCarree()) 
    ax.coastlines()
    ax.set_ylim(0, 60)  

plt.tight_layout()
plt.show()


#MEAN TC TRACK DENSITY: Note this code is for only historical and for only one model. The other experiments and models are changed from the file paths

densityesmh=np.zeros((65,36,72)) #year,lat,lon every 5 degree lat lon
for ie in range(10):
    for iy,yr in enumerate(range(1950,2015)):
        read_in_esmh=xr.open_dataset(f'/work/ggyabaah/CHAZ/E3SM/historical//wdir_HIST_CRH/output/E3SM-2-0_{yr}_ens{ie:03d}.nc')
        for ic in range(len(read_in_esmh.stormID)):
            ln=read_in_esmh.isel(stormID=ic)['longitude'].values
            lt=read_in_esmh.isel(stormID=ic)['latitude'].values
            ww=read_in_esmh.isel(stormID=ic)['Mwspd'].isel(ensembleNum=0).values
            ln=ln[~np.isnan(ww)]
            lt=lt[~np.isnan(ww)]
            ww=ww[~np.isnan(ww)]
            for ixx0,iyy0 in zip(ln,lt):
                ixx=np.floor(ixx0/5)
                iyy=np.floor((iyy0+90)/5)
                densityesmh[iy,int(iyy),int(ixx)]=densityesmh[iy,int(iyy),int(ixx)]+1
        read_in_esmh.close()
densityesmh=densityesmh/10. #units
esmh=np.mean(densityesmh[0:64], axis=0)
# after all experiments have been calculated:
hhh=added_h-added_n
ggg=added_g-added_n
aaa=added_a-added_n

#calculating the statistical significance: Assunibg for only historical experiment
    
def ztest(a,b):
    zstat = ((a.mean('time') - b.mean('time')) - 0)/(np.sqrt((a.var('time')/len(a))+(b.var('time')/len(b))))
    pvalue = sp.norm.sf(abs(zstat))*2
    return(pvalue)
def create_xr(pval):
    ds=xr.Dataset(

        {

            "PValue": (("lat", "lon"), pval),

        },

        coords={

            "lat": ("lat", hist.lat.data),

            "lon": ("lon", hist.lon.data),

        },

    )

    ds = ds['PValue']

    return(ds)
PVal=create_xr(ztest(nat,hist))


#plotting the mean track densities

fig, axes = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
axes.coastlines()
cmap = plt.get_cmap('bwr')  
below_0_2_color = 'white'  
cmap.set_under(below_0_2_color)
nrows, ncols = ggg.shape
longitude = np.arange(0, 360, 5)
latitude = np.arange(-90, 90, 5)
x = latitude
y = longitude
X, Y = np.meshgrid(x, y)
cb = axes.pcolormesh(Y, X, ggg.T, cmap=cmap, transform=ccrs.PlateCarree(),vmin=-2,vmax=2)
plt.contour(Y-180,X,-1 * added_g.T-180,levels=(9))
mpl.rc('hatch',color='black',lw=1)
axes.contourf(Y,X+5,PVal.T,cmap='bwr',levels=[0,0.05,999],transform=ccrs.PlateCarree(),hatches=['..',''],alpha=0)
cbar = fig.colorbar(cb, label='Track Density/year', pad=0.001, shrink=0.4380)
cbar.set_label('Track Density/year')
axes.set_xlabel('Longitude')
axes.set_ylabel('Latitude')
axes.set_title('MEAN DIFFERENCE TC TRACK DENSITY COMPOSITE 11 MODELS GHG 10 SEEDING ENSEMBLES(65 years)')
axes.set_xticks(range(0, 361, 30), crs=ccrs.PlateCarree())
axes.xaxis.set_major_formatter(plt.FixedFormatter(['0°E', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W', '0°E']))
axes.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
axes.yaxis.set_major_formatter(plt.FixedFormatter(['90°N', '60°N', '30°N', '0°', '30°S', '60°S', '90°S']))
axes.coastlines()
land_mask = cfeature.NaturalEarthFeature('physical', 'land', '110m')
axes.add_feature(land_mask, facecolor='grey')
axes.set_ylim(-60, 60,1)
plt.show()

#TC Climatology and changes

# An example of classifing CHAZ TC in WNP for 10 seeding ens, 40 intensity ens
# This is a basic code, so it take about 3 minute to generate the final product
# To improve the speed, you can do the average for all ensembes(400) to get the mean of frequency for one model
#This is for only 1 model and 1 experiment.

new_regions=xr.open_dataset('/data/ylin/share_scripts/TC_basin_180x90.nc')['__xarray_dataarray_variable__']

def is_point_in_region_xarray(lat1, lon1, data_array):
    value = data_array.sel(lat=lat1, lon=lon1, method='nearest').values
    return value


in_dir='/work/ggyabaah/CHAZ/'
model='ACCESS-CM2'
exp='historical'
hum='SD'
ens=10

freq_wnp_ac2h_CRH=np.zeros((65,10,40,6))
freq_ep_ac2h_CRH=np.zeros((65,10,40,6))
freq_na_ac2h_CRH=np.zeros((65,10,40,6))
freq_ni_ac2h_CRH=np.zeros((65,10,40,6))
freq_si_ac2h_CRH=np.zeros((65,10,40,6))
freq_sp_ac2h_CRH=np.zeros((65,10,40,6))
#freq_au_ac2h_CRH=np.zeros((65,10,40,6))#year(1950-2015),ens(10),int(40),category(0,1,2,3,4,5)

for iy,year in enumerate(range(1950,2015)):
    for ie in range(ens):
        
        
        
        
        in_file=in_dir+model+'/'+exp+'/wdir_HIST_'+hum+'/output/'+model+'_'+str(year)+'_ens00'+str(ie)+'.nc'
        readin=xr.open_dataset(in_file)
            
        maxw_id=(readin['Mwspd'].max('lifelength'))
            
        lon_id=readin['longitude'][0,:]
        lat_id=readin['latitude'][0,:]
        basin_idx=np.array([is_point_in_region_xarray(lat1, lon1, new_regions) for lat1, lon1 in zip(lat_id.values,lon_id.values)])
            
            
        wk_basin=(np.argwhere((basin_idx==0))).T[0] #basin_idx 0:WNP
            
        for inten in range(40):
            
            wk_maxw_id=maxw_id[inten,wk_basin]
            
            freq_wnp_ac2h_CRH[iy,ie,inten,5]=xr.where((wk_maxw_id>=137),1,np.nan).sum().values
            freq_wnp_ac2h_CRH[iy,ie,inten,4]=xr.where(((wk_maxw_id<137)&(wk_maxw_id>=113)),1,np.nan).sum().values
            freq_wnp_ac2h_CRH[iy,ie,inten,3]=xr.where(((wk_maxw_id<113)&(wk_maxw_id>=96)),1,np.nan).sum().values
            freq_wnp_ac2h_CRH[iy,ie,inten,2]=xr.where(((wk_maxw_id<96)&(wk_maxw_id>=83)),1,np.nan).sum().values
            freq_wnp_ac2h_CRH[iy,ie,inten,1]=xr.where(((wk_maxw_id<83)&(wk_maxw_id>=64)),1,np.nan).sum().values
            freq_wnp_ac2h_CRH[iy,ie,inten,0]=xr.where(((wk_maxw_id<64)&(wk_maxw_id>=34)),1,np.nan).sum().values
        
        wk_basin=(np.argwhere((basin_idx==1))).T[0] 
            
        for inten in range(40):
            
            wk_maxw_id=maxw_id[inten,wk_basin]
            
            freq_ep_ac2h_CRH[iy,ie,inten,5]=xr.where((wk_maxw_id>=137),1,np.nan).sum().values
            freq_ep_ac2h_CRH[iy,ie,inten,4]=xr.where(((wk_maxw_id<137)&(wk_maxw_id>=113)),1,np.nan).sum().values
            freq_ep_ac2h_CRH[iy,ie,inten,3]=xr.where(((wk_maxw_id<113)&(wk_maxw_id>=96)),1,np.nan).sum().values
            freq_ep_ac2h_CRH[iy,ie,inten,2]=xr.where(((wk_maxw_id<96)&(wk_maxw_id>=83)),1,np.nan).sum().values
            freq_ep_ac2h_CRH[iy,ie,inten,1]=xr.where(((wk_maxw_id<83)&(wk_maxw_id>=64)),1,np.nan).sum().values
            freq_ep_ac2h_CRH[iy,ie,inten,0]=xr.where(((wk_maxw_id<64)&(wk_maxw_id>=34)),1,np.nan).sum().values
        
        wk_basin=(np.argwhere((basin_idx==2))).T[0] 
            
        for inten in range(40):
            
            wk_maxw_id=maxw_id[inten,wk_basin]
            
            freq_na_ac2h_CRH[iy,ie,inten,5]=xr.where((wk_maxw_id>=137),1,np.nan).sum().values
            freq_na_ac2h_CRH[iy,ie,inten,4]=xr.where(((wk_maxw_id<137)&(wk_maxw_id>=113)),1,np.nan).sum().values
            freq_na_ac2h_CRH[iy,ie,inten,3]=xr.where(((wk_maxw_id<113)&(wk_maxw_id>=96)),1,np.nan).sum().values
            freq_na_ac2h_CRH[iy,ie,inten,2]=xr.where(((wk_maxw_id<96)&(wk_maxw_id>=83)),1,np.nan).sum().values
            freq_na_ac2h_CRH[iy,ie,inten,1]=xr.where(((wk_maxw_id<83)&(wk_maxw_id>=64)),1,np.nan).sum().values
            freq_na_ac2h_CRH[iy,ie,inten,0]=xr.where(((wk_maxw_id<64)&(wk_maxw_id>=34)),1,np.nan).sum().values
        
        wk_basin=(np.argwhere((basin_idx==4))).T[0]  
        for inten in range(40):            
            wk_maxw_id=maxw_id[inten,wk_basin]
            freq_ni_ac2h_CRH[iy,ie,inten,5]=xr.where((wk_maxw_id>=137),1,np.nan).sum().values
            freq_ni_ac2h_CRH[iy,ie,inten,4]=xr.where(((wk_maxw_id<137)&(wk_maxw_id>=113)),1,np.nan).sum().values
            freq_ni_ac2h_CRH[iy,ie,inten,3]=xr.where(((wk_maxw_id<113)&(wk_maxw_id>=96)),1,np.nan).sum().values
            freq_ni_ac2h_CRH[iy,ie,inten,2]=xr.where(((wk_maxw_id<96)&(wk_maxw_id>=83)),1,np.nan).sum().values
            freq_ni_ac2h_CRH[iy,ie,inten,1]=xr.where(((wk_maxw_id<83)&(wk_maxw_id>=64)),1,np.nan).sum().values
            freq_ni_ac2h_CRH[iy,ie,inten,0]=xr.where(((wk_maxw_id<64)&(wk_maxw_id>=34)),1,np.nan).sum().values
        
        wk_basin=(np.argwhere((basin_idx==5))).T[0]
        for inten in range(40):
            
            wk_maxw_id=maxw_id[inten,wk_basin]
            
            freq_si_ac2h_CRH[iy,ie,inten,5]=xr.where((wk_maxw_id>=137),1,np.nan).sum().values
            freq_si_ac2h_CRH[iy,ie,inten,4]=xr.where(((wk_maxw_id<137)&(wk_maxw_id>=113)),1,np.nan).sum().values
            freq_si_ac2h_CRH[iy,ie,inten,3]=xr.where(((wk_maxw_id<113)&(wk_maxw_id>=96)),1,np.nan).sum().values
            freq_si_ac2h_CRH[iy,ie,inten,2]=xr.where(((wk_maxw_id<96)&(wk_maxw_id>=83)),1,np.nan).sum().values
            freq_si_ac2h_CRH[iy,ie,inten,1]=xr.where(((wk_maxw_id<83)&(wk_maxw_id>=64)),1,np.nan).sum().values
            freq_si_ac2h_CRH[iy,ie,inten,0]=xr.where(((wk_maxw_id<64)&(wk_maxw_id>=34)),1,np.nan).sum().values
        wk_basin=(np.argwhere((basin_idx==6))).T[0] #basin_idx 0:WNP
            
        for inten in range(40):
            
            wk_maxw_id=maxw_id[inten,wk_basin]
            
            freq_sp_ac2h_CRH[iy,ie,inten,5]=xr.where((wk_maxw_id>=137),1,np.nan).sum().values
            freq_sp_ac2h_CRH[iy,ie,inten,4]=xr.where(((wk_maxw_id<137)&(wk_maxw_id>=113)),1,np.nan).sum().values
            freq_sp_ac2h_CRH[iy,ie,inten,3]=xr.where(((wk_maxw_id<113)&(wk_maxw_id>=96)),1,np.nan).sum().values
            freq_sp_ac2h_CRH[iy,ie,inten,2]=xr.where(((wk_maxw_id<96)&(wk_maxw_id>=83)),1,np.nan).sum().values
            freq_sp_ac2h_CRH[iy,ie,inten,1]=xr.where(((wk_maxw_id<83)&(wk_maxw_id>=64)),1,np.nan).sum().values
            freq_sp_ac2h_CRH[iy,ie,inten,0]=xr.where(((wk_maxw_id<64)&(wk_maxw_id>=34)),1,np.nan).sum().values
                            
        readin.close()



#Here the code is repeated for each experiment and their composite is what was used to make the plots:
    
basins = ['WNP', 'EP', 'NA', 'NI', 'SI', 'SP']
freqs = [wnph,
         eph,
         nah,
         nih,
         sih,
         sph]
freqsa = [wnpa,
         epa,
         naa,
         nia,
         sia,
         spa]
freqsg = [wnpg,
         epg,
         nag,
         nig,
         sig,
         spg]
freqsn = [wnpn,
         epn,
         nan,
         nin,
         sin,
         spn]


basins = ['WNP', 'EP', 'NA', 'NI', 'SI', 'SP']
cats = ['TS', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5']

fh = [f - fn for f, fn in zip(freqs, freqsn)]
fa = [a - fn for a, fn in zip(freqsa, freqsn)]
fg = [g - fn for g, fn in zip(freqsg, freqsn)]

bar_width = 0.2
index = np.arange(len(cats))

fig, axs = plt.subplots(6, 1, figsize=(10, 20), dpi=100)

for i, basin in enumerate(basins):
    ax = axs[i]

    ax.bar(index - bar_width, freqs[i][:len(cats)], bar_width, color='blue', label='HIST-nat')
    ax.bar(index, freqsg[i][:len(cats)], bar_width, color='red', label='GHG-nat')
    ax.bar(index + bar_width, freqsa[i][:len(cats)], bar_width, color='brown', label='AER-nat')
    ax.bar(index + bar_width * 2, freqsn[i][:len(cats)], bar_width, color='green', label='NAT')

    ax.set_xlabel('Categories')
    ax.set_ylabel('Frequency')
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(cats)
    
    ax.legend()
    
    ax.set_title(f'{basin.upper()}')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

cats = ['TS', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5']
n = (wnpn + epn + nan +nin+ sin + spn)
h = (wnph + eph + nah +nin+ sih + sph)-n
a = (wnpa + epa + naa +nin+ sia + spa)-n
g = (wnpg + epg + nag + nin+ sig + spg)-n

bar_width = 0.2
index = np.arange(len(cats))

fig, axs = plt.subplots(7, 1, figsize=(10, 24), dpi=1000)

ax = axs[0]
ax.bar(index - bar_width, h[:len(cats)], bar_width, color='blue', label='HIST')
ax.bar(index, g[:len(cats)], bar_width, color='red', label='GHG')
ax.bar(index + bar_width, a[:len(cats)], bar_width, color='brown', label='AER')
#ax.bar(index + bar_width * 2, n[:len(cats)], bar_width, color='green', label='NAT')

ax.set_xlabel('Categories')
ax.set_ylabel('Frequency')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(cats)
ax.legend()
ax.set_ylim(-1,1.5,0.1)

ax.set_title('SD Global TC frequency difference 10 models (10ens,40int) (1950-2014)', fontsize=12)
ax.text(0.5, 0.95, 'Global', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold',color='black')

for i, basin in enumerate(basins):
    ax = axs[i+1]
    

    ax.bar(index - bar_width, fh[i][:len(cats)], bar_width, color='blue', label='HIST-NAT')
    
    ax.bar(index, fg[i][:len(cats)], bar_width, color='red', label='GHG-NAT')
    
    ax.bar(index + bar_width, fa[i][:len(cats)], bar_width, color='brown', label='AER-NAT')
    #ax.bar(index + bar_width * 2, fn[i][:len(cats)], bar_width, color='green', label='NAT-NAT')

    ax.set_xlabel('Categories')
    ax.set_ylabel('Frequency')
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(cats)
    ax.set_ylim(-1,1.5,0.1)
    ax.legend()
    ax.text(0.5, 0.95, basin, ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold', color='black')

    ax.set_title(f'{basin.upper()}')

plt.tight_layout()
plt.show()

        



