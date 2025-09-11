# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:39:40 2025

@author: rg727
"""

#########################################################################Create Livneh Geophysical Datasets########################################

import xarray as xr
import geopandas as gpd
import rioxarray
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box
from shapely.geometry import MultiPolygon, Polygon
import time
import matplotlib.colors as mcolors
from math import sin, cos, acos, tan, radians
import numpy as np
import warnings
warnings.filterwarnings("ignore")


gauge_list = pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/canada_expanded_gl_gauges.csv", dtype={'STAID': str})
gauge_list["STAID"] = gauge_list["STAID"].str.strip()
gauge_ids = gauge_list["STAID"].tolist()

grip = gpd.read_file("C:/Users/rg727/Desktop/Earths_Future_Revision/taxes/MDA_ADP_02/MDA_ADP_02_DrainageBasin_BassinDeDrainage.shp")
grip["StationNum"] = grip["StationNum"].astype(str).str.strip()
shape = grip[grip["StationNum"].isin(gauge_ids)].reset_index()

for g in range(0,len(shape)):
    grip=shape
    grip=grip.iloc[g:g+1,:]
    grip=grip.to_crs("EPSG:4326")
    
    
    data_precip=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Precipitation/Livneh_Canada_Expanded/"+str(grip["StationNum"][g])+'.csv')
    data_tmin=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Precipitation/Livneh_Canada_Expanded/"+str(grip["StationNum"][g])+'.csv')
    data_tmax=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Precipitation/Livneh_Canada_Expanded/"+str(grip["StationNum"][g])+'.csv')
    
    df=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Geophysical_Datasets/Expanded/canada_geophysical/"+str(grip["StationNum"][g])+'.csv')
    
    ########################################################LONG TERM CLIMATE VARIABLES#################################################################
    p_mean=data_precip['pr'].mean()
    df['p_mean']=data_precip['pr'].mean()
    
    ##############################################################PET (Hamon Method) ######################################################################################
    if isinstance(grip['geometry'][g], Polygon):
    # For single Polygon
        coords = list(grip['geometry'][g].exterior.coords)
        average_latitude = sum(point[1] for point in coords) / len(coords)

    elif isinstance(grip['geometry'][g], MultiPolygon):
    # For MultiPolygon
        all_coords = []
        for polygon in grip['geometry'][g].geoms:  # Access individual polygons in MultiPolygon
            all_coords.extend(list(polygon.exterior.coords))
            average_latitude = sum(point[1] for point in all_coords) / len(all_coords)

   
    #Calculate the day of the year
    
    dti = pd.date_range('01-01-1950', '12-31-2013')
    doy=dti.dayofyear
    
    
    # Function to calculate day length based on latitude and day of year
    def calculate_day_length(lat, doy):
    # Declination of the sun (in radians)
        declination = 23.45 * sin(radians(360 / 365 * (doy - 81)))
        declination_rad = radians(declination)
    
        # Latitude in radians
        lat_rad = radians(average_latitude)
        
        # Calculate day length in hours
        sunset_angle = acos(-tan(lat_rad) * tan(declination_rad))  # Sunset angle
        day_length = (24 / np.pi) * 2 * sunset_angle
        return day_length

    # Calculate day length for each row
    data_tmax['latitude']=average_latitude
    data_tmax['doy']=doy
    data_tmax['day_length'] = data_tmax.apply(lambda row: calculate_day_length(row['latitude'], row['doy']), axis=1)
    
    # Calculate saturation vapor pressure (e_s) in mb
    data_tmax['saturation_vapor_pressure'] = 6.108 * np.exp((17.27 * ((data_tmax['tmax']+data_tmin['tmin'])/2)) / (((data_tmax['tmax']+data_tmin['tmin'])/2) + 237.3))
    
    # Calculate PET using the Hamon equation
    data_tmax['PET'] = 0.1651 * (data_tmax['day_length'] * data_tmax['saturation_vapor_pressure']) / 100
    
    pet_mean=data_tmax['PET'].mean()
    df['pet_mean']=pet_mean
    
    ###############################################################################################################################################################
        
    aridity= pet_mean/p_mean
    df['aridity']=aridity
    
    t_mean=((data_tmin['tmin']+data_tmax['tmax'])/2).mean()
    df['t_mean']=t_mean
    
    ###############################################################Snow Fraction ###############################################
    
    #calculate total precipitation 
    p_tot=data_precip['pr'].sum()
    
    #keep only days where daily temp is below zero and there is precipitation 
    
    data_precip['average_T']=((data_tmin['tmin']+data_tmax['tmax'])/2)
    
    data_precip_subset= data_precip[(data_precip['pr'] > 0) & (data_precip['average_T'] < 0)]
    frac_snow=data_precip_subset['pr'].sum()/p_tot
    
    df['frac_snow']=frac_snow
    
    #############################################################################################################################

    high_prec_freq= (data_precip['pr'] > 5 * p_mean).mean()
    df['high_prec_freq']=high_prec_freq
    
    ############################################################################################################################################
    high_days = (data_precip['pr'] > 5 * p_mean)
    high_periods = (high_days != high_days.shift()).cumsum()  # Identify groups of consecutive values
    high_period_lengths = high_days.groupby(high_periods).sum()  # Sum only dry days in each group

    # Filter out groups that are not dry periods
    high_period_lengths[high_period_lengths > 0]

    # Calculate the average duration of dry periods
    high_prec_dur = high_period_lengths.mean()
    df['high_prec_dur']=high_prec_dur


    ############################################################################################################################################
    
    low_prec_freq=(data_precip['pr'] < 1 ).mean()
    df['low_prec_freq']=low_prec_freq

    dry_days = data_precip['pr'] < 1
    dry_periods = (dry_days != dry_days.shift()).cumsum()  # Identify groups of consecutive values
    dry_period_lengths = dry_days.groupby(dry_periods).sum()  # Sum only dry days in each group

    # Filter out groups that are not dry periods
    dry_period_lengths[dry_period_lengths > 0]

    # Calculate the average duration of dry periods
    low_prec_dur = dry_period_lengths.mean()
    df['low_prec_dur']=low_prec_dur
    
    df.to_csv("C:/Users/rg727/Documents/Great Lake Project/Geophysical_Datasets/Expanded/Livneh/canada_geophysical/"+str(grip["StationNum"][g])+'.csv',index=False)
    
###################################################################US_ref###############################################################################  
gauge_list = pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/us_expanded_ref_gauges.csv", dtype={'STAID': str})
gauge_list["STAID"] = gauge_list["STAID"].str.strip()
gauge_ids = gauge_list["STAID"].tolist()

shape = gpd.read_file("C:/Users/rg727/Desktop/Earths_Future_Revision/taxes/boundaries_shapefiles_by_aggeco/boundaries-shapefiles-by-aggeco/bas_ref_all.shp")
shape["GAGE_ID"] = shape["GAGE_ID"].astype(str).str.strip()
shape = shape[shape["GAGE_ID"].isin(gauge_ids)].reset_index()


for g in range(0,len(shape)):
    grip=shape
    grip=grip.iloc[g:g+1,:]
    grip=grip.to_crs("EPSG:4326")
    
    
    data_precip=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Precipitation/Livneh_US_Ref_Expanded/"+str(grip['GAGE_ID'][g])+'.csv')
    data_tmin=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Precipitation/Livneh_US_Ref_Expanded/"+str(grip['GAGE_ID'][g])+'.csv')
    data_tmax=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Precipitation/Livneh_US_Ref_Expanded/"+str(grip['GAGE_ID'][g])+'.csv')
    
    df=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Geophysical_Datasets/Expanded/us_ref_geophysical/"+str(grip['GAGE_ID'][g])+'.csv')
    
    ########################################################LONG TERM CLIMATE VARIABLES#################################################################
    p_mean=data_precip['pr'].mean()
    df['p_mean']=data_precip['pr'].mean()
    
    ##############################################################PET (Hamon Method) ######################################################################################
    if isinstance(grip['geometry'][g], Polygon):
    # For single Polygon
        coords = list(grip['geometry'][g].exterior.coords)
        average_latitude = sum(point[1] for point in coords) / len(coords)

    elif isinstance(grip['geometry'][g], MultiPolygon):
    # For MultiPolygon
        all_coords = []
        for polygon in grip['geometry'][g].geoms:  # Access individual polygons in MultiPolygon
            all_coords.extend(list(polygon.exterior.coords))
            average_latitude = sum(point[1] for point in all_coords) / len(all_coords)

   
    #Calculate the day of the year
    
    dti = pd.date_range('01-01-1950', '12-31-2013')
    doy=dti.dayofyear
    
    
    # Function to calculate day length based on latitude and day of year
    def calculate_day_length(lat, doy):
    # Declination of the sun (in radians)
        declination = 23.45 * sin(radians(360 / 365 * (doy - 81)))
        declination_rad = radians(declination)
    
        # Latitude in radians
        lat_rad = radians(average_latitude)
        
        # Calculate day length in hours
        sunset_angle = acos(-tan(lat_rad) * tan(declination_rad))  # Sunset angle
        day_length = (24 / np.pi) * 2 * sunset_angle
        return day_length

    # Calculate day length for each row
    data_tmax['latitude']=average_latitude
    data_tmax['doy']=doy
    data_tmax['day_length'] = data_tmax.apply(lambda row: calculate_day_length(row['latitude'], row['doy']), axis=1)
    
    # Calculate saturation vapor pressure (e_s) in mb
    data_tmax['saturation_vapor_pressure'] = 6.108 * np.exp((17.27 * ((data_tmax['tmax']+data_tmin['tmin'])/2)) / (((data_tmax['tmax']+data_tmin['tmin'])/2) + 237.3))
    
    # Calculate PET using the Hamon equation
    data_tmax['PET'] = 0.1651 * (data_tmax['day_length'] * data_tmax['saturation_vapor_pressure']) / 100
    
    pet_mean=data_tmax['PET'].mean()
    df['pet_mean']=pet_mean
    
    ###############################################################################################################################################################
        
    aridity= pet_mean/p_mean
    df['aridity']=aridity
    
    t_mean=((data_tmin['tmin']+data_tmax['tmax'])/2).mean()
    df['t_mean']=t_mean
    
    ###############################################################Snow Fraction ###############################################
    
    #calculate total precipitation 
    p_tot=data_precip['pr'].sum()
    
    #keep only days where daily temp is below zero and there is precipitation 
    
    data_precip['average_T']=((data_tmin['tmin']+data_tmax['tmax'])/2)
    
    data_precip_subset= data_precip[(data_precip['pr'] > 0) & (data_precip['average_T'] < 0)]
    frac_snow=data_precip_subset['pr'].sum()/p_tot
    
    df['frac_snow']=frac_snow
    
    #############################################################################################################################

    high_prec_freq= (data_precip['pr'] > 5 * p_mean).mean()
    df['high_prec_freq']=high_prec_freq
    
    ############################################################################################################################################
    high_days = (data_precip['pr'] > 5 * p_mean)
    high_periods = (high_days != high_days.shift()).cumsum()  # Identify groups of consecutive values
    high_period_lengths = high_days.groupby(high_periods).sum()  # Sum only dry days in each group

    # Filter out groups that are not dry periods
    high_period_lengths[high_period_lengths > 0]

    # Calculate the average duration of dry periods
    high_prec_dur = high_period_lengths.mean()
    df['high_prec_dur']=high_prec_dur


    ############################################################################################################################################
    
    low_prec_freq=(data_precip['pr'] < 1 ).mean()
    df['low_prec_freq']=low_prec_freq

    dry_days = data_precip['pr'] < 1
    dry_periods = (dry_days != dry_days.shift()).cumsum()  # Identify groups of consecutive values
    dry_period_lengths = dry_days.groupby(dry_periods).sum()  # Sum only dry days in each group

    # Filter out groups that are not dry periods
    dry_period_lengths[dry_period_lengths > 0]

    # Calculate the average duration of dry periods
    low_prec_dur = dry_period_lengths.mean()
    df['low_prec_dur']=low_prec_dur
    
    df.to_csv("C:/Users/rg727/Documents/Great Lake Project/Geophysical_Datasets/Expanded/Livneh/us_ref/"+str(grip['GAGE_ID'][g])+'.csv',index=False)
    
    ###################################################################US_Non_Ref###############################################################################
    gauge_list = pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/us_expanded_nonref_gauges.csv", dtype={'STAID': str})
    gauge_list["STAID"] = gauge_list["STAID"].str.strip()
    gauge_ids = gauge_list["STAID"].tolist()

    shape = gpd.read_file("C:/Users/rg727/Desktop/Earths_Future_Revision/taxes/boundaries_shapefiles_by_aggeco/boundaries-shapefiles-by-aggeco/bas_nonref_MxWdShld.shp")
    shape["GAGE_ID"] = shape["GAGE_ID"].astype(str).str.strip()
    shape = shape[shape["GAGE_ID"].isin(gauge_ids)].reset_index()


    for g in range(0,len(shape)):
        grip=shape
        grip=grip.iloc[g:g+1,:]
        grip=grip.to_crs("EPSG:4326")
        
        
        data_precip=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Precipitation/Livneh_US_NonRef_Expanded/"+str(grip['GAGE_ID'][g])+'.csv')
        data_tmin=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Precipitation/Livneh_US_NonRef_Expanded/"+str(grip['GAGE_ID'][g])+'.csv')
        data_tmax=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Precipitation/Livneh_US_NonRef_Expanded/"+str(grip['GAGE_ID'][g])+'.csv')
        
        df=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Geophysical_Datasets/Expanded/us_nonref_geophysical/"+str(grip['GAGE_ID'][g])+'.csv')
        
        ########################################################LONG TERM CLIMATE VARIABLES#################################################################
        p_mean=data_precip['pr'].mean()
        df['p_mean']=data_precip['pr'].mean()
        
        ##############################################################PET (Hamon Method) ######################################################################################
        if isinstance(grip['geometry'][g], Polygon):
        # For single Polygon
            coords = list(grip['geometry'][g].exterior.coords)
            average_latitude = sum(point[1] for point in coords) / len(coords)

        elif isinstance(grip['geometry'][g], MultiPolygon):
        # For MultiPolygon
            all_coords = []
            for polygon in grip['geometry'][g].geoms:  # Access individual polygons in MultiPolygon
                all_coords.extend(list(polygon.exterior.coords))
                average_latitude = sum(point[1] for point in all_coords) / len(all_coords)

       
        #Calculate the day of the year
        
        dti = pd.date_range('01-01-1950', '12-31-2013')
        doy=dti.dayofyear
        
        
        # Function to calculate day length based on latitude and day of year
        def calculate_day_length(lat, doy):
        # Declination of the sun (in radians)
            declination = 23.45 * sin(radians(360 / 365 * (doy - 81)))
            declination_rad = radians(declination)
        
            # Latitude in radians
            lat_rad = radians(average_latitude)
            
            # Calculate day length in hours
            sunset_angle = acos(-tan(lat_rad) * tan(declination_rad))  # Sunset angle
            day_length = (24 / np.pi) * 2 * sunset_angle
            return day_length

        # Calculate day length for each row
        data_tmax['latitude']=average_latitude
        data_tmax['doy']=doy
        data_tmax['day_length'] = data_tmax.apply(lambda row: calculate_day_length(row['latitude'], row['doy']), axis=1)
        
        # Calculate saturation vapor pressure (e_s) in mb
        data_tmax['saturation_vapor_pressure'] = 6.108 * np.exp((17.27 * ((data_tmax['tmax']+data_tmin['tmin'])/2)) / (((data_tmax['tmax']+data_tmin['tmin'])/2) + 237.3))
        
        # Calculate PET using the Hamon equation
        data_tmax['PET'] = 0.1651 * (data_tmax['day_length'] * data_tmax['saturation_vapor_pressure']) / 100
        
        pet_mean=data_tmax['PET'].mean()
        df['pet_mean']=pet_mean
        
        ###############################################################################################################################################################
            
        aridity= pet_mean/p_mean
        df['aridity']=aridity
        
        t_mean=((data_tmin['tmin']+data_tmax['tmax'])/2).mean()
        df['t_mean']=t_mean
        
        ###############################################################Snow Fraction ###############################################
        
        #calculate total precipitation 
        p_tot=data_precip['pr'].sum()
        
        #keep only days where daily temp is below zero and there is precipitation 
        
        data_precip['average_T']=((data_tmin['tmin']+data_tmax['tmax'])/2)
        
        data_precip_subset= data_precip[(data_precip['pr'] > 0) & (data_precip['average_T'] < 0)]
        frac_snow=data_precip_subset['pr'].sum()/p_tot
        
        df['frac_snow']=frac_snow
        
        #############################################################################################################################

        high_prec_freq= (data_precip['pr'] > 5 * p_mean).mean()
        df['high_prec_freq']=high_prec_freq
        
        ############################################################################################################################################
        high_days = (data_precip['pr'] > 5 * p_mean)
        high_periods = (high_days != high_days.shift()).cumsum()  # Identify groups of consecutive values
        high_period_lengths = high_days.groupby(high_periods).sum()  # Sum only dry days in each group

        # Filter out groups that are not dry periods
        high_period_lengths[high_period_lengths > 0]

        # Calculate the average duration of dry periods
        high_prec_dur = high_period_lengths.mean()
        df['high_prec_dur']=high_prec_dur


        ############################################################################################################################################
        
        low_prec_freq=(data_precip['pr'] < 1 ).mean()
        df['low_prec_freq']=low_prec_freq

        dry_days = data_precip['pr'] < 1
        dry_periods = (dry_days != dry_days.shift()).cumsum()  # Identify groups of consecutive values
        dry_period_lengths = dry_days.groupby(dry_periods).sum()  # Sum only dry days in each group

        # Filter out groups that are not dry periods
        dry_period_lengths[dry_period_lengths > 0]

        # Calculate the average duration of dry periods
        low_prec_dur = dry_period_lengths.mean()
        df['low_prec_dur']=low_prec_dur
        
        df.to_csv("C:/Users/rg727/Documents/Great Lake Project/Geophysical_Datasets/Expanded/Livneh/us_non_ref/"+str(grip['GAGE_ID'][g])+'.csv',index=False)
        