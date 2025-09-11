# -*- coding: utf-8 -*-
"""
Created on Tue May 13 09:59:58 2025

@author: rg727
"""

#664 gauges in total 

#ID the gauges that we have for each time period. This script is for the last modeling period. 

import os
import pandas as pd
from dateutil import parser
from datetime import datetime
from pathlib import Path
import numpy as np
import glob
from datetime import datetime, timedelta
from scipy.stats import spearmanr, pearsonr
from scipy.stats import norm
from sklearn.model_selection import KFold
import pickle

#First take care of the Cananda files 

'''
folder_path = 'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Runoff/'  # <- Change this

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        filepath = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(filepath)

            # Only process if the column exists
            if "Parameter/Paramètre" in df.columns:
                # Remove rows where the value is exactly "water level/niveau"
                df_filtered = df[df["Parameter/Paramètre"] != "water level/niveau"]

                # Optional: reset index if you care
                df_filtered.reset_index(drop=True, inplace=True)

                # Overwrite the file (or use a different name to save a new copy)
                df_filtered.to_csv(filepath, index=False)

                print(f"Filtered and saved: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
'''
# Define directory and date range
folder_path = 'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Runoff/'
start_date = datetime(1995, 1, 1)
end_date = datetime(2013, 12, 31)


matching_files = []
skipped_files = []

# Loop through CSV files
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        filepath = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(filepath, low_memory=False)

            # Try to find a date column
            date_col = None
            for col in df.columns:
                if col.strip().lower() in ['date', 'datetime']:
                    date_col = col
                    break
            if date_col is None:
                skipped_files.append((filename, 'No date/datetime column'))
                continue

            # Convert to string first, then parse dates
            date_series = df[date_col].astype(str)
            dates = pd.to_datetime(date_series, errors='coerce', infer_datetime_format=True)
            dates = dates.dt.tz_localize(None)

            if dates.notna().sum() == 0:
                skipped_files.append((filename, 'All dates failed to parse'))
                continue

            if ((dates >= start_date) & (dates <= end_date)).any():
                matching_files.append(Path(filename).stem)

        except Exception as e:
            skipped_files.append((filename, f'Error: {e}'))

# Output results
print("✅ Files with dates between 1980-01-01 and 1995-12-31:")
for f in matching_files:
    print(f)

print("\n⚠️ Skipped files and reasons:")
for f, reason in skipped_files:
    print(f"{f}: {reason}")

#How many gauges are there 

len(matching_files)
        
file_path = "C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/1995_2013_gauges.txt"

with open(file_path, "w") as file:
    for item in matching_files:
        file.write(str(item) + "\n")                
        
        
#######Isolate out the len(final_df_validation)% of matching files with the most comprehensive coverage of the time period. Leave the last 20% for validation gauges###### 
'''
folder_path = 'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Runoff/'  # <- update this

start_date = datetime(1980, 1, 1)
end_date = datetime(1995 12, 31)


coverage_list = []

# --- Measure coverage ---
for stem in matching_files:
    filepath = os.path.join(folder_path, f"{stem}.csv")
    try:
        df = pd.read_csv(filepath, low_memory=False)

        # Identify date column
        date_col = None
        for col in df.columns:
            if col.strip().lower() in ['date', 'datetime']:
                date_col = col
                break
        if not date_col:
            continue

        # Parse dates
        dates = pd.to_datetime(df[date_col].astype(str), errors='coerce', infer_datetime_format=True)
        dates = dates.dt.tz_localize(None)

        # Count valid dates in range
        coverage = ((dates >= start_date) & (dates <= end_date)).sum()
        coverage_list.append((stem, coverage))

    except Exception as e:
        print(f"Error reading {stem}.csv: {e}")

# --- Sort and Split ---
coverage_sorted = sorted(coverage_list, key=lambda x: x[1], reverse=True)
n_total = len(coverage_sorted)
n_testing = int(n_total * 0.8)

top_testing = [x[0] for x in coverage_sorted[:n_testing]]
bottom_20 = [x[0] for x in coverage_sorted[n_testing:]]
'''
################Create a 5-fold cross validation##############


# Initialize the KFold splitter
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create dictionaries to store the folds
folds = {}
outs = {}

# Enumerate through the splits
for i, (train_idx, test_idx) in enumerate(kf.split(matching_files), start=1):
    fold_name = f'fold_{i}'
    out_name = f'out_{i}'
    
    folds[fold_name] = [matching_files[j] for j in train_idx]
    outs[out_name] = [matching_files[j] for j in test_idx]

# Save the variable to a file
filename = "C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/folds.pkl"
with open(filename, "wb") as file:
    pickle.dump(folds, file)
    
    
# Save the variable to a file
filename = "C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/outs.pkl"
with open(filename, "wb") as file:
    pickle.dump(outs, file) 
    
    

##############################################################Create Training Data#################################################
###############################################################Create Folders for Each Fold###################################

if not os.path.exists('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/'):
        os.makedirs('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/')
        print("Created folder")
else:
        print("Folder already exists")
        
if not os.path.exists('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Testing/out_5/'):
                os.makedirs('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Testing/out_5/')
                print("Created folder")
else:
                print("Folder already exists")
                
if not os.path.exists('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5_neighbors/'):
                os.makedirs('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5_neighbors/')
                print("Created folder")
else:
                print("Folder already exists")         
        
if not os.path.exists('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Testing/out_5_neighbors/'):
                os.makedirs('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Testing/out_5_neighbors/')
                print("Created folder")
else:
                print("Folder already exists")        
#########################################################

compiled_original_gauges=list(pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations/compiled_gauges_no_duplicates.csv")['ID'])

#Cananda


CA_gauges_training= [gauge for gauge in folds['fold_5'] if gauge.startswith("02")]
CA_gauges_testing=[gauge for gauge in outs['out_5'] if gauge.startswith("02")]


# Generate date range
date_range = pd.date_range(start="1995-01-01", end="2013-12-31", freq="D")

'''
#First remove the front lake info

folder_path = 'C:/Users/rg727/Documents/Great Lake Project/Precipitation/Intercomparison_prcp_Livneh/All/'  # ← update this

for filename in os.listdir(folder_path):
    if filename.endswith('.csv') and '_' in filename:
        old_path = os.path.join(folder_path, filename)
        
        # Get part after first underscore
        new_name = filename.split('_', 1)[1]
        new_path = os.path.join(folder_path+'/renamed/', new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
        



#First remove the front lake info

folder_path = 'C:/Users/rg727/Documents/Great Lake Project/Geophysical_Datasets/Intercomparison_Basins/Livneh_Geophysical/'  # ← update this

for filename in os.listdir(folder_path):
    if filename.endswith('.csv') and '_' in filename:
        old_path = os.path.join(folder_path, filename)
        
        # Get part after first underscore
        new_name = filename.split('_', 1)[1]
        new_path = os.path.join(folder_path+'/renamed/', new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
        

'''


for i in range(0,len(CA_gauges_training)):

    df = pd.DataFrame(0, index=np.arange(len(date_range)), columns=['Year', 'Month', 'Day', 'q', 'pr', 'tmax', 'tmin','p_mean', 'pet_mean', 'aridity','t_mean', 'frac_snow', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'mean_elev', 'std_elev', 'mean_slope', 'std_slope', 'area_km2', 'temperate_forest', 'temperate_grassland', 'wetland', 'cropland', 'barren', 'urban', 'water', 'BD', 'CLAY_SHALLOW','CLAY_DEEP','GRAV_SHALLOW', 'GRAV_DEEP', 'OC_SHALLOW', 'OC_DEEP', 'SAND_SHALLOW', 'SAND_DEEP', 'SILT_SHALLOW', 'SILT_DEEP' ])
   

    precipitation=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Meteorology/'+str(CA_gauges_training[i])+'.csv')
    runoff=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Runoff/'+str(CA_gauges_training[i])+'.csv')
    geophysical_data=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Geophysical_Datasets/'+str(CA_gauges_training[i])+'.csv')
    

    
    df ["Date"]= date_range
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    
    df['pr']=precipitation['pr'][16436:23376].reset_index()['pr']
    df['tmax']=precipitation['tmax'][16436:23376].reset_index()['tmax']
    df['tmin']=precipitation['tmin'][16436:23376].reset_index()['tmin']
     
    start_date = "1995-01-01"
    end_date = "2014-01-01"
    subset_df_range = runoff.loc[(runoff["Date"] >= start_date) & (runoff["Date"] <= end_date) & (runoff["Parameter/Paramètre"]=='discharge/débit')]
    
    subset_df_range['Date'] = pd.to_datetime(subset_df_range['Date'])
    df = df.merge(subset_df_range[['Date', 'Value/Valeur']], on='Date', how='left', suffixes=('', '_from_runoff'))
    df['q'] = df['Value/Valeur']
    df.drop(columns='Value/Valeur', inplace=True)
    
    df.drop('Date', axis=1, inplace=True)
    
    geo_repeated = geophysical_data.loc[geophysical_data.index.repeat(len(date_range))].reset_index(drop=True)
    
    df.iloc[:,7:39]=geo_repeated.values
    
    #df['OC_SHALLOW']=df['OC_SHALLOW']*0.01
    #df['OC_DEEP']=df['OC_DEEP']*0.01
    #df['BD']=df['BD']*0.01
    
    df = df.applymap(lambda x: np.nan if pd.isna(x) or x == '' else x)
    df = df.drop('temperate_grassland', axis=1)
    
    
    if CA_gauges_training[i] in compiled_original_gauges:
        
        df['area_km2']=df['area_km2']/(10**6)
    
    df['q']= df['q']*3600*24/df['area_km2']/1000
    
    if df.drop(columns="q").isna().any().any() == False and df['q'].sum()>0 and df['q'].notna().any():
     #df.to_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/2000_2013_expanded_Livneh/'+str(CA_gauges[i])+'.csv',index=False,na_rep="NaN")
     df.to_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/'+str(CA_gauges_training[i])+'.csv',index=False,na_rep="NaN")

    
    else:
     print("All values are NaN, not saving the file.")


for i in range(0,len(CA_gauges_testing)):

    df = pd.DataFrame(0, index=np.arange(len(date_range)), columns=['Year', 'Month', 'Day', 'q', 'pr', 'tmax', 'tmin','p_mean', 'pet_mean', 'aridity','t_mean', 'frac_snow', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'mean_elev', 'std_elev', 'mean_slope', 'std_slope', 'area_km2', 'temperate_forest', 'temperate_grassland', 'wetland', 'cropland', 'barren', 'urban', 'water', 'BD', 'CLAY_SHALLOW','CLAY_DEEP','GRAV_SHALLOW', 'GRAV_DEEP', 'OC_SHALLOW', 'OC_DEEP', 'SAND_SHALLOW', 'SAND_DEEP', 'SILT_SHALLOW', 'SILT_DEEP' ])
   

    precipitation=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Meteorology/'+str(CA_gauges_testing[i])+'.csv')
    runoff=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Runoff/'+str(CA_gauges_testing[i])+'.csv')
    geophysical_data=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Geophysical_Datasets/'+str(CA_gauges_testing[i])+'.csv')
    

    
    df ["Date"]= date_range
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    
    df['pr']=precipitation['pr'][16436:23376].reset_index()['pr']
    df['tmax']=precipitation['tmax'][16436:23376].reset_index()['tmax']
    df['tmin']=precipitation['tmin'][16436:23376].reset_index()['tmin']
     
    start_date = "1995-01-01"
    end_date = "2014-01-01"
    subset_df_range = runoff.loc[(runoff["Date"] >= start_date) & (runoff["Date"] <= end_date) & (runoff["Parameter/Paramètre"]=='discharge/débit')]
    
    subset_df_range['Date'] = pd.to_datetime(subset_df_range['Date'])
    df = df.merge(subset_df_range[['Date', 'Value/Valeur']], on='Date', how='left', suffixes=('', '_from_runoff'))
    df['q'] = df['Value/Valeur']
    df.drop(columns='Value/Valeur', inplace=True)
    
    df.drop('Date', axis=1, inplace=True)
    
    geo_repeated = geophysical_data.loc[geophysical_data.index.repeat(len(date_range))].reset_index(drop=True)
    
    df.iloc[:,7:39]=geo_repeated.values
    
    #df['OC_SHALLOW']=df['OC_SHALLOW']*0.01
    #df['OC_DEEP']=df['OC_DEEP']*0.01
    #df['BD']=df['BD']*0.01
    
    df = df.applymap(lambda x: np.nan if pd.isna(x) or x == '' else x)
    df = df.drop('temperate_grassland', axis=1)
    
    if CA_gauges_testing[i] in compiled_original_gauges:
        df['area_km2']=df['area_km2']/(10**6)
    
    df['q']= df['q']*3600*24/df['area_km2']/1000
    
    if df.drop(columns="q").isna().any().any() == False and df['q'].sum()>0 and df['q'].notna().any():
     #df.to_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/2000_2013_expanded_Livneh/'+str(CA_gauges[i])+'.csv',index=False,na_rep="NaN")
     df.to_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Testing/out_5/'+str(CA_gauges_testing[i])+'.csv',index=False,na_rep="NaN")

    
    else:
     print("All values are NaN, not saving the file.")


#US 


US_gauges_training= [gauge for gauge in folds['fold_5'] if gauge.startswith("04")]
US_gauges_testing=[gauge for gauge in outs['out_5'] if gauge.startswith("04")]


for i in range(0,len(US_gauges_training)):

    df = pd.DataFrame(0, index=np.arange(len(date_range)), columns=['Year', 'Month', 'Day', 'q', 'pr', 'tmax', 'tmin','p_mean', 'pet_mean', 'aridity','t_mean', 'frac_snow', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'mean_elev', 'std_elev', 'mean_slope', 'std_slope', 'area_km2', 'temperate_forest', 'temperate_grassland', 'wetland', 'cropland', 'barren', 'urban', 'water', 'BD', 'CLAY_SHALLOW','CLAY_DEEP','GRAV_SHALLOW', 'GRAV_DEEP', 'OC_SHALLOW', 'OC_DEEP', 'SAND_SHALLOW', 'SAND_DEEP', 'SILT_SHALLOW', 'SILT_DEEP' ])
   

    precipitation=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Meteorology/'+str(US_gauges_training[i])+'.csv')
    runoff=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Runoff/'+str(US_gauges_training[i])+'.csv')
    geophysical_data=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Geophysical_Datasets/'+str(US_gauges_training[i])+'.csv')
    

    df ["Date"]= date_range
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    
    df['pr']=precipitation['pr'][16436:23376].reset_index()['pr']
    df['tmax']=precipitation['tmax'][16436:23376].reset_index()['tmax']
    df['tmin']=precipitation['tmin'][16436:23376].reset_index()['tmin']
    
      
    start_date = "1995-01-01"
    end_date = "2014-01-01"
    subset_df_range = runoff.loc[(runoff["datetime"] >= start_date) & (runoff["datetime"] <= end_date)]
    
    if '00060_Mean' in subset_df_range.columns:
        subset_df_range['Date'] = pd.to_datetime(subset_df_range['datetime'])
        subset_df_range['Date'] =  subset_df_range['Date'].dt.tz_localize(None)
        df = df.merge(subset_df_range[['Date', '00060_Mean']], on='Date', how='left')
        df['q'] = df['00060_Mean']
        df.drop(columns='00060_Mean', inplace=True)
        
    elif '00060_2_Mean' in subset_df_range.columns: 
        subset_df_range['Date'] = pd.to_datetime(subset_df_range['datetime'])
        subset_df_range['Date'] =  subset_df_range['Date'].dt.tz_localize(None)
        df = df.merge(subset_df_range[['Date', '00060_2_Mean']], on='Date', how='left')
        df['q'] = df['00060_2_Mean']
        df.drop(columns='00060_2_Mean', inplace=True)

    
    df.drop('Date', axis=1, inplace=True)
    
    geo_repeated = geophysical_data.loc[geophysical_data.index.repeat(len(date_range))].reset_index(drop=True)
    
    df.iloc[:,7:39]=geo_repeated.values
    df = df.drop('temperate_grassland', axis=1)
    
    if US_gauges_training[i] in compiled_original_gauges:
        df['area_km2']=df['area_km2']/(10**6)

    
    df['q']= df['q']*3600*24*304.8/df['area_km2']/(3280*3280)

    
    if df.drop(columns="q").isna().any().any() == False and df['q'].sum()>0 and df['q'].notna().any():
     #df.to_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/2000_2013_expanded_Livneh/'+str(CA_gauges[i])+'.csv',index=False,na_rep="NaN")
     df.to_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/'+str(US_gauges_training[i])+'.csv',index=False,na_rep="NaN")

    
    else:
     print("All values are NaN, not saving the file.")


for i in range(0,len(US_gauges_testing)):

    df = pd.DataFrame(0, index=np.arange(len(date_range)), columns=['Year', 'Month', 'Day', 'q', 'pr', 'tmax', 'tmin','p_mean', 'pet_mean', 'aridity','t_mean', 'frac_snow', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'mean_elev', 'std_elev', 'mean_slope', 'std_slope', 'area_km2', 'temperate_forest', 'temperate_grassland', 'wetland', 'cropland', 'barren', 'urban', 'water', 'BD', 'CLAY_SHALLOW','CLAY_DEEP','GRAV_SHALLOW', 'GRAV_DEEP', 'OC_SHALLOW', 'OC_DEEP', 'SAND_SHALLOW', 'SAND_DEEP', 'SILT_SHALLOW', 'SILT_DEEP' ])
   

    precipitation=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Meteorology/'+str(US_gauges_testing[i])+'.csv')
    runoff=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Runoff/'+str(US_gauges_testing[i])+'.csv')
    geophysical_data=pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/Geophysical_Datasets/'+str(US_gauges_testing[i])+'.csv')
    

    df ["Date"]= date_range
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    
    df['pr']=precipitation['pr'][16436:23376].reset_index()['pr']
    df['tmax']=precipitation['tmax'][16436:23376].reset_index()['tmax']
    df['tmin']=precipitation['tmin'][16436:23376].reset_index()['tmin']
    
      
    start_date = "1995-01-01"
    end_date = "2014-01-01"
    subset_df_range = runoff.loc[(runoff["datetime"] >= start_date) & (runoff["datetime"] <= end_date)]

    if '00060_Mean' in subset_df_range.columns:
        subset_df_range['Date'] = pd.to_datetime(subset_df_range['datetime'])
        subset_df_range['Date'] =  subset_df_range['Date'].dt.tz_localize(None)
        df = df.merge(subset_df_range[['Date', '00060_Mean']], on='Date', how='left')
        df['q'] = df['00060_Mean']
        df.drop(columns='00060_Mean', inplace=True)
        
    elif '00060_2_Mean' in subset_df_range.columns: 
        subset_df_range['Date'] = pd.to_datetime(subset_df_range['datetime'])
        subset_df_range['Date'] =  subset_df_range['Date'].dt.tz_localize(None)
        df = df.merge(subset_df_range[['Date', '00060_2_Mean']], on='Date', how='left')
        df['q'] = df['00060_2_Mean']
        df.drop(columns='00060_2_Mean', inplace=True)

    
    df.drop('Date', axis=1, inplace=True)
    
    geo_repeated = geophysical_data.loc[geophysical_data.index.repeat(len(date_range))].reset_index(drop=True)
    
    df.iloc[:,7:39]=geo_repeated.values
    df = df.drop('temperate_grassland', axis=1)
    
    if US_gauges_testing[i] in compiled_original_gauges:
        df['area_km2']=df['area_km2']/(10**6)

    
    df['q']= df['q']*3600*24*304.8/df['area_km2']/(3280*3280)
    
    if df.drop(columns="q").isna().any().any() == False and df['q'].sum()>0 and df['q'].notna().any():
     #df.to_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/2000_2013_expanded_Livneh/'+str(CA_gauges[i])+'.csv',index=False,na_rep="NaN")
     df.to_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Testing/out_5/'+str(US_gauges_testing[i])+'.csv',index=False,na_rep="NaN")

    
    else:
     print("All values are NaN, not saving the file.")

#################################################################Generate Nearest Neighbors#############################################################
#Collect file names for the nearest neighbor training

folder = Path('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5')
training_gauges = [f.stem for f in folder.iterdir() if f.is_file()]


df = pd.DataFrame({'ID': training_gauges})

#Collect the lat/lons for the gauges 

# Step 1: Load all metadata files
csv_files = glob.glob('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/2011_2017_expanded_CPC/metadata/*.csv')

temps = []
for file in csv_files:
    temp = pd.read_csv(file)
    temps.append(temp)

all_data = pd.concat(temps, ignore_index=True)

# Step 2: Assume you already have a DataFrame of gauge IDs (with or without leading zeros)
# e.g., from a list of 331 gauges:
# df_gauges = pd.DataFrame({'ID': gauge_list})

df['ID_clean'] = df['ID'].astype(str).str.lstrip('0')
all_data['ID_clean'] = all_data['ID'].astype(str).str.lstrip('0')

all_data = all_data.drop_duplicates(subset='ID_clean')

# Step 3: Merge on the cleaned ID
final_df = df.merge(
    all_data[['ID_clean', 'Latitude', 'Longitude']],
    on='ID_clean',
    how='left'
)

# Optional: drop the temp 'ID_clean' column or rearrange
final_df = final_df.drop(columns='ID_clean')

###########################################Train the Model###################################



gage_lat_training = final_df['Latitude'].values
gage_lon_training = final_df['Longitude'].values
gage_id_training = final_df['ID']

start_date = datetime(1995, 1, 1)
end_date = datetime(2013, 12, 31)


num_day = (end_date - start_date).days + 1

gage_q_training = np.nan * np.zeros((num_day, len(final_df)))
gage_pr_training = np.nan * np.zeros((num_day, len(final_df)))
gage_dist_training = np.nan * np.zeros((len(final_df), len(final_df)))


#Calculate the euclidian distance between gauges
n = 0
for i in range(len(final_df)):
    d = pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/{gage_id_training[i]}.csv')
    #d=d.iloc[0:4018,:] #Train over 2000-2010 only
    dist_lat = (gage_lat_training - gage_lat_training[i]) ** 2
    dist_lon = (gage_lon_training - gage_lon_training[i]) ** 2
    dist_euclid = np.sqrt(dist_lat + dist_lon)
    gage_dist_training[n+1:, n] = dist_euclid[n+1:]

    gage_q_training[:, n] = d['q'].values
    gage_pr_training[:, n] = d['pr'].values

    dist_euclid[dist_euclid == 0] = np.nan

    n += 1

    
all_nan_cols = np.isnan(gage_q_training).all(axis=0)

# Get the column indices where all values are NaN
nan_col_indices = np.where(all_nan_cols)[0]

# This will return a boolean mask for constant columns
same_val_cols = np.all(gage_q_training == gage_q_training[0, :], axis=0)

# Get column indices where all values are the same
same_val_col_indices = np.where(same_val_cols)[0]

    
#calculate the spearman rank correlation    

gage_q_spearman_training = np.nan * np.zeros((len(final_df), len(final_df)))
gage_q_pearson_training = np.nan * np.zeros((len(final_df), len(final_df)))
gage_pr_spearman_training = np.nan * np.zeros((len(final_df), len(final_df)))


for i in range(len(final_df)):
    q_gage_sel = gage_q_training[:, i]
    pr_gage_sel = gage_pr_training[:, i]

    for j in range(i + 1, len(final_df)):
        q_gage_train_sel = gage_q_training[:, j]
        pr_gage_train_sel = gage_pr_training[:, j]
        
        # Pearson correlation with overlap check
        mask = ~np.isnan(q_gage_sel) & ~np.isnan(q_gage_train_sel)
        if np.sum(mask) > 1:  # at least 2 points required for Pearson correlation
           gage_q_pearson_training[j, i], _ = pearsonr(q_gage_sel[mask], q_gage_train_sel[mask])
           gage_q_spearman_training[j, i],_=spearmanr(q_gage_sel[mask], q_gage_train_sel[mask])
        else:
           gage_q_pearson_training[j, i] = 0.0  # or np.nan, depending on how you want to treat it
           gage_q_spearman_training[j, i] = 0.0

        gage_pr_spearman_training[j, i], _ = spearmanr(pr_gage_sel, pr_gage_train_sel, nan_policy='omit')
        
####################################################################################################################################################################        
gage_temperate_forest_training = np.full((len(final_df), 1), np.nan)
gage_wetland_training = np.full((len(final_df), 1), np.nan)
gage_cropland_training = np.full((len(final_df), 1), np.nan)
gage_barren_training = np.full((len(final_df), 1), np.nan)
gage_urban_training = np.full((len(final_df), 1), np.nan)
gage_water_training = np.full((len(final_df), 1), np.nan)
gage_BD_training = np.full((len(final_df), 1), np.nan)
gage_CLAY_SHALLOW_training = np.full((len(final_df), 1), np.nan)
gage_CLAY_DEEP_training = np.full((len(final_df), 1), np.nan)
gage_GRAV_SHALLOW_training = np.full((len(final_df), 1), np.nan)
gage_GRAV_DEEP_training = np.full((len(final_df), 1), np.nan)
gage_OC_SHALLOW_training = np.full((len(final_df), 1), np.nan)
gage_OC_DEEP_training = np.full((len(final_df), 1), np.nan)
gage_SAND_SHALLOW_training = np.full((len(final_df), 1), np.nan)
gage_SAND_DEEP_training = np.full((len(final_df), 1), np.nan)
gage_SILT_SHALLOW_training = np.full((len(final_df), 1), np.nan)
gage_SILT_DEEP_training = np.full((len(final_df), 1), np.nan)
gage_mean_elev_training = np.full((len(final_df), 1), np.nan)
gage_mean_slope_training = np.full((len(final_df), 1), np.nan)
gage_std_elev_training = np.full((len(final_df), 1), np.nan)
gage_std_slope_training = np.full((len(final_df), 1), np.nan)
gage_area_km2_training = np.full((len(final_df), 1), np.nan)        
        


# Read CSV files and extract values
for i in range(len(final_df)):
    d_training = pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/{gage_id_training[i]}.csv')
    #d_training=d_training.iloc[0:4017,:]
    gage_temperate_forest_training[i] = d_training['temperate_forest'].iloc[0]
    gage_wetland_training[i] = d_training['wetland'].iloc[0]
    gage_cropland_training[i] = d_training['cropland'].iloc[0]
    gage_barren_training[i] = d_training['barren'].iloc[0]
    gage_urban_training[i] = d_training['urban'].iloc[0]
    gage_water_training[i] = d_training['water'].iloc[0]
    gage_BD_training[i] = d_training['BD'].iloc[0]
    gage_CLAY_SHALLOW_training[i] = d_training['CLAY_SHALLOW'].iloc[0]
    gage_CLAY_DEEP_training[i] = d_training['CLAY_DEEP'].iloc[0]
    gage_GRAV_SHALLOW_training[i] = d_training['GRAV_SHALLOW'].iloc[0]
    gage_GRAV_DEEP_training[i] = d_training['GRAV_DEEP'].iloc[0]
    gage_OC_SHALLOW_training[i] = d_training['OC_SHALLOW'].iloc[0]
    gage_OC_DEEP_training[i] = d_training['OC_DEEP'].iloc[0]
    gage_SAND_SHALLOW_training[i] = d_training['SAND_SHALLOW'].iloc[0]
    gage_SAND_DEEP_training[i] = d_training['SAND_DEEP'].iloc[0]
    gage_SILT_SHALLOW_training[i] = d_training['SILT_SHALLOW'].iloc[0]
    gage_SILT_DEEP_training[i] = d_training['SILT_DEEP'].iloc[0]
    gage_mean_elev_training[i] = d_training['mean_elev'].iloc[0]
    gage_mean_slope_training[i] = d_training['mean_slope'].iloc[0]
    gage_std_elev_training[i] = d_training['std_elev'].iloc[0]
    gage_std_slope_training[i] = d_training['std_slope'].iloc[0]
    gage_area_km2_training[i] = d_training['area_km2'].iloc[0]
 


gage_dist_temperate_forest_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_wetland_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_cropland_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_barren_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_urban_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_water_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_BD_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_CLAY_SHALLOW_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_CLAY_DEEP_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_GRAV_SHALLOW_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_GRAV_DEEP_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_OC_SHALLOW_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_OC_DEEP_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_SAND_SHALLOW_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_SAND_DEEP_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_SILT_SHALLOW_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_SILT_DEEP_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_mean_elev_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_mean_slope_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_std_elev_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_std_slope_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_area_km2_training = np.full((len(final_df), len(final_df)), np.nan)


n=0

for i in range (len(final_df)): 
    d =pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/{gage_id_training[i]}.csv')
    #d=d.iloc[0:4017,:] #Train over 2000-2010 only
    diff_temperate_forest = gage_temperate_forest_training-d['temperate_forest'][0]
    diff_wetland=gage_wetland_training-d['wetland'][0]
    diff_cropland=gage_cropland_training-d['cropland'][0]
    diff_barren=gage_barren_training-d['barren'][0]
    diff_urban=gage_urban_training-d['urban'][0]
    diff_water=gage_water_training-d['water'][0]
    diff_BD=gage_BD_training-d['BD'][0]
    diff_CLAY_SHALLOW = gage_CLAY_SHALLOW_training-d['CLAY_SHALLOW'][0]
    diff_CLAY_DEEP = gage_CLAY_DEEP_training-d['CLAY_DEEP'][0]
    diff_GRAV_SHALLOW = gage_GRAV_SHALLOW_training-d['GRAV_SHALLOW'][0]
    diff_GRAV_DEEP = gage_GRAV_DEEP_training-d['GRAV_DEEP'][0]
    diff_OC_SHALLOW = gage_OC_SHALLOW_training-d['OC_SHALLOW'][0]
    diff_OC_DEEP = gage_OC_DEEP_training-d['OC_DEEP'][0]
    diff_SAND_SHALLOW= gage_SAND_SHALLOW_training-d['SAND_SHALLOW'][0]
    diff_SAND_DEEP= gage_SAND_DEEP_training-d['SAND_DEEP'][0]
    diff_SILT_SHALLOW = gage_SILT_SHALLOW_training-d['SILT_SHALLOW'][0]
    diff_SILT_DEEP = gage_SILT_SHALLOW_training-d['SILT_SHALLOW'][0]
    diff_mean_elev = gage_mean_elev_training-d['mean_elev'][0]
    diff_mean_slope = gage_mean_slope_training-d['mean_slope'][0]
    diff_std_elev = gage_std_elev_training-d['std_elev'][0]
    diff_std_slope = gage_std_slope_training-d['std_slope'][0]
    diff_area_km2 = gage_area_km2_training-d['area_km2'][0]
    
    gage_dist_temperate_forest_training[n+1:, n] = diff_temperate_forest[n+1:,0]
    gage_dist_wetland_training[n+1:, n] = diff_wetland[n+1:,0]
    gage_dist_cropland_training[n+1:, n] =diff_cropland[n+1:,0]
    gage_dist_barren_training[n+1:, n] = diff_barren[n+1:,0]
    gage_dist_urban_training[n+1:, n] = diff_urban[n+1:,0]
    gage_dist_water_training[n+1:, n] = diff_water[n+1:,0]
    gage_dist_BD_training[n+1:, n] = diff_BD[n+1:,0]
    gage_dist_CLAY_SHALLOW_training[n+1:, n] = diff_CLAY_SHALLOW[n+1:,0]
    gage_dist_CLAY_DEEP_training[n+1:, n] = diff_CLAY_DEEP[n+1:,0]
    gage_dist_GRAV_SHALLOW_training[n+1:, n] = diff_GRAV_SHALLOW[n+1:,0]
    gage_dist_GRAV_DEEP_training[n+1:, n] = diff_GRAV_DEEP[n+1:,0]
    gage_dist_OC_SHALLOW_training[n+1:, n] = diff_OC_SHALLOW[n+1:,0]
    gage_dist_OC_DEEP_training[n+1:, n] = diff_OC_DEEP[n+1:,0]
    gage_dist_SAND_SHALLOW_training[n+1:, n] = diff_SAND_SHALLOW[n+1:,0]
    gage_dist_SAND_DEEP_training[n+1:, n] = diff_SAND_DEEP[n+1:,0]
    gage_dist_SILT_SHALLOW_training[n+1:, n] = diff_SILT_SHALLOW[n+1:,0]
    gage_dist_SILT_DEEP_training[n+1:, n] = diff_SILT_DEEP[n+1:,0]
    gage_dist_mean_elev_training[n+1:, n] = diff_mean_elev[n+1:,0]
    gage_dist_mean_slope_training[n+1:, n] = diff_mean_slope[n+1:,0]
    gage_dist_std_elev_training[n+1:, n] = diff_std_elev[n+1:,0]
    gage_dist_std_slope_training[n+1:, n] = diff_std_slope[n+1:,0]
    gage_dist_area_km2_training[n+1:, n] = diff_area_km2[n+1:,0]
    
    n=n+1
    
    

mean_pr = np.nanmean(gage_pr_spearman_training)
mean_dist = np.nanmean(gage_dist_training)
mean_temperate_forest = np.nanmean(gage_dist_temperate_forest_training)
mean_wetland = np.nanmean(gage_dist_wetland_training)
mean_cropland = np.nanmean(gage_dist_cropland_training)
mean_barren = np.nanmean(gage_dist_barren_training)
mean_urban = np.nanmean(gage_dist_urban_training)
mean_water = np.nanmean(gage_dist_water_training)
mean_BD = np.nanmean(gage_dist_BD_training)
mean_CLAY_SHALLOW = np.nanmean(gage_dist_CLAY_SHALLOW_training)
mean_CLAY_DEEP = np.nanmean(gage_dist_CLAY_DEEP_training)
mean_GRAV_SHALLOW = np.nanmean(gage_dist_GRAV_SHALLOW_training)
mean_GRAV_DEEP = np.nanmean(gage_dist_GRAV_DEEP_training)
mean_OC_SHALLOW = np.nanmean(gage_dist_OC_SHALLOW_training)
mean_OC_DEEP = np.nanmean(gage_dist_OC_DEEP_training)
mean_SAND_SHALLOW = np.nanmean(gage_dist_SAND_SHALLOW_training)
mean_SAND_DEEP = np.nanmean(gage_dist_SAND_DEEP_training)
mean_SILT_SHALLOW = np.nanmean(gage_dist_SILT_SHALLOW_training)
mean_SILT_DEEP = np.nanmean(gage_dist_SILT_DEEP_training)
mean_mean_elev = np.nanmean(gage_dist_mean_elev_training)
mean_mean_slope = np.nanmean(gage_dist_mean_slope_training)
mean_std_elev = np.nanmean(gage_dist_std_elev_training)
mean_std_slope = np.nanmean(gage_dist_std_slope_training)
mean_area_km2 = np.nanmean(gage_dist_area_km2_training)


std_pr = np.nanstd(gage_pr_spearman_training.flatten('F'))
std_dist = np.nanstd(gage_dist_training.flatten('F'))
std_temperate_forest = np.nanstd(gage_dist_temperate_forest_training.flatten('F'))
std_wetland = np.nanstd(gage_dist_wetland_training.flatten('F'))
std_cropland = np.nanstd(gage_dist_cropland_training.flatten('F'))
std_barren = np.nanstd(gage_dist_barren_training.flatten('F'))
std_urban = np.nanstd(gage_dist_urban_training.flatten('F'))
std_water = np.nanstd(gage_dist_water_training.flatten('F'))
std_BD = np.nanstd(gage_dist_BD_training.flatten('F'))
std_CLAY_SHALLOW = np.nanstd(gage_dist_CLAY_SHALLOW_training.flatten('F'))
std_CLAY_DEEP = np.nanstd(gage_dist_CLAY_DEEP_training.flatten('F'))
std_GRAV_SHALLOW = np.nanstd(gage_dist_GRAV_SHALLOW_training.flatten('F'))
std_GRAV_DEEP = np.nanstd(gage_dist_GRAV_DEEP_training.flatten('F'))
std_OC_SHALLOW = np.nanstd(gage_dist_OC_SHALLOW_training.flatten('F'))
std_OC_DEEP = np.nanstd(gage_dist_OC_DEEP_training.flatten('F'))
std_SAND_SHALLOW = np.nanstd(gage_dist_SAND_SHALLOW_training.flatten('F'))
std_SAND_DEEP = np.nanstd(gage_dist_SAND_DEEP_training.flatten('F'))
std_SILT_SHALLOW = np.nanstd(gage_dist_SILT_SHALLOW_training.flatten('F'))
std_SILT_DEEP = np.nanstd(gage_dist_SILT_DEEP_training.flatten('F'))
std_mean_elev = np.nanstd(gage_dist_mean_elev_training.flatten('F'))
std_mean_slope = np.nanstd(gage_dist_mean_slope_training.flatten('F'))
std_std_elev = np.nanstd(gage_dist_std_elev_training.flatten('F'))
std_std_slope = np.nanstd(gage_dist_std_slope_training.flatten('F'))
std_area_km2 = np.nanstd(gage_dist_area_km2_training.flatten('F'))

stand_pr_training = (gage_pr_spearman_training.flatten('F') - mean_pr) / std_pr
stand_dist_training = (gage_dist_training.flatten('F') - mean_dist) / std_dist
stand_temperate_forest_training = (gage_dist_temperate_forest_training.flatten('F') - mean_temperate_forest) / std_temperate_forest
stand_wetland_training = (gage_dist_wetland_training.flatten('F') - mean_wetland) / std_wetland
stand_cropland_training = (gage_dist_cropland_training.flatten('F') - mean_cropland) / std_cropland
stand_barren_training = (gage_dist_barren_training.flatten('F') - mean_barren) / std_barren
stand_urban_training = (gage_dist_urban_training.flatten('F') - mean_urban) / std_urban
stand_water_training = (gage_dist_water_training.flatten('F') - mean_water) / std_water
stand_BD_training = (gage_dist_BD_training.flatten('F') - mean_BD) / std_BD
stand_CLAY_SHALLOW_training = (gage_dist_CLAY_SHALLOW_training.flatten('F') - mean_CLAY_SHALLOW) / std_CLAY_SHALLOW
stand_CLAY_DEEP_training = (gage_dist_CLAY_DEEP_training.flatten('F') - mean_CLAY_DEEP) / std_CLAY_DEEP
stand_GRAV_SHALLOW_training = (gage_dist_GRAV_SHALLOW_training.flatten('F') - mean_GRAV_SHALLOW) / std_GRAV_SHALLOW
stand_GRAV_DEEP_training = (gage_dist_GRAV_DEEP_training.flatten('F') - mean_GRAV_DEEP) / std_GRAV_DEEP
stand_OC_SHALLOW_training = (gage_dist_OC_SHALLOW_training.flatten('F') - mean_OC_SHALLOW) / std_OC_SHALLOW
stand_OC_DEEP_training = (gage_dist_OC_DEEP_training.flatten('F') - mean_OC_DEEP) / std_OC_DEEP
stand_SAND_SHALLOW_training = (gage_dist_SAND_SHALLOW_training.flatten('F') - mean_SAND_SHALLOW) / std_SAND_SHALLOW
stand_SAND_DEEP_training = (gage_dist_SAND_DEEP_training.flatten('F') - mean_SAND_DEEP) / std_SAND_DEEP
stand_SILT_SHALLOW_training = (gage_dist_SILT_SHALLOW_training.flatten('F') - mean_SILT_SHALLOW) / std_SILT_SHALLOW
stand_SILT_DEEP_training = (gage_dist_SILT_DEEP_training.flatten('F') - mean_SILT_DEEP) / std_SILT_DEEP
stand_mean_elev_training = (gage_dist_mean_elev_training.flatten('F') - mean_mean_elev) / std_mean_elev
stand_mean_slope_training = (gage_dist_mean_slope_training.flatten('F') - mean_mean_slope) / std_mean_slope
stand_std_elev_training = (gage_dist_std_elev_training.flatten('F') - mean_std_elev) / std_std_elev
stand_std_slope_training = (gage_dist_std_slope_training.flatten('F') - mean_std_slope) / std_std_slope
stand_area_km2_training = (gage_dist_area_km2_training.flatten('F') - mean_area_km2) / std_area_km2


X = np.column_stack((
    stand_pr_training, stand_dist_training,
    stand_temperate_forest_training,
    stand_wetland_training, stand_cropland_training, stand_barren_training, stand_urban_training, stand_water_training,
    stand_BD_training, stand_CLAY_SHALLOW_training,stand_CLAY_DEEP_training,stand_GRAV_SHALLOW_training, stand_GRAV_DEEP_training, stand_OC_SHALLOW_training, stand_OC_DEEP_training, stand_SAND_SHALLOW_training,stand_SAND_DEEP_training, stand_SILT_SHALLOW_training,stand_SILT_DEEP_training,
    stand_mean_elev_training, stand_mean_slope_training, stand_std_elev_training, stand_std_slope_training, stand_area_km2_training
))

Y = gage_q_spearman_training.flatten('F')

valid_indices = ~np.isnan(Y)
X_valid = X[valid_indices, :]
Y_valid = Y[valid_indices]

Xtrain = X_valid
Ytrain = Y_valid

############################################MLP#######################
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error



num_epochs=50
num_features=Xtrain.shape[1]


#Mini-batch size for SGD
batch_size = 64
#Adam optimizer learning rate
alpha = .001
B1 = 0.9
B2 = 0.999
e = 10^-8
#Number of hidden layer units
hiddn_size = 35
#Number of output layer units
num_output_lyr_unit = 1
#Dropout rate
dropout_rate = 0.2 # 0 is no dropout node; 0.2: on average 20% of hidden nodes are dropped out



# Standardize the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train = scaler_x.fit_transform(Xtrain)
#x_test = scaler_x.transform(XX)
y_train = scaler_y.fit_transform(Ytrain.reshape(-1, 1))

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
#x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

#Create a dataset
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)


# Create DataLoader
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the MLP model with Dropout
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 35)  
        self.tanh = nn.Tanh()                 
        #self.dropout1 = nn.Dropout(0.2)       # Dropout after first layer
        self.fc2 = nn.Linear(35, 35)         
        self.sigmoid = nn.Sigmoid()           
        #self.dropout2 = nn.Dropout(0.2)       # Dropout after second layer
        self.fc3 = nn.Linear(35, 1)           # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        #x = self.dropout1(x)  
        x = self.fc2(x)
        x = self.sigmoid(x)
        #x = self.dropout2(x)  
        x = self.fc3(x)  
        return x
    
# Initialize the model, loss function, and optimizer
model = MLPModel(Xtrain.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(B1, B2), eps=1e-8)


for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if epoch % 10 == 0:  # Print average loss per epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")


###################################################VALIDATION############################################################################

#Collect file names for the nearest neighbor training

folder = Path('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Testing/out_5')
validation_gauges = [f.stem for f in folder.iterdir() if f.is_file()]


df = pd.DataFrame({'ID': validation_gauges})

#Collect the lat/lons for the gauges 

# Step 1: Load all metadata files
csv_files = glob.glob('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/2011_2017_expanded_CPC/metadata/*.csv')

temps = []
for file in csv_files:
    temp = pd.read_csv(file)
    temps.append(temp)

all_data = pd.concat(temps, ignore_index=True)

# Step 2: Assume you already have a DataFrame of gauge IDs (with or without leading zeros)
# e.g., from a list of 331 gauges:
# df_gauges = pd.DataFrame({'ID': gauge_list})

df['ID_clean'] = df['ID'].astype(str).str.lstrip('0')
all_data['ID_clean'] = all_data['ID'].astype(str).str.lstrip('0')

all_data = all_data.drop_duplicates(subset='ID_clean')

# Step 3: Merge on the cleaned ID
final_df_validation = df.merge(
    all_data[['ID_clean', 'Latitude', 'Longitude']],
    on='ID_clean',
    how='left'
)

# Optional: drop the temp 'ID_clean' column or rearrange
final_df_validation = final_df_validation.drop(columns='ID_clean')



gage_lat_testing= final_df_validation['Latitude'].values
gage_lon_testing = final_df_validation['Longitude'].values
gage_id_testing = final_df_validation['ID']


gage_pr_testing = np.nan * np.zeros((num_day, len(final_df_validation)))
gage_dist_testing = np.nan * np.zeros((len(final_df),len(final_df_validation)))


for i in range(len(final_df_validation)):
    d = pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Testing/out_5/{gage_id_testing[i]}.csv')
    #d=d.iloc[0:4017,:] #Train over 2000-2010 only
    #d=d.iloc[11688:,:]
    gage_pr_testing[:, i] = d['pr']
    dist_lat = (gage_lat_training - gage_lat_testing[i]) ** 2
    dist_lon = (gage_lon_training - gage_lon_testing[i]) ** 2
    dist_euclid = np.sqrt(dist_lat + dist_lon)

    gage_dist_testing[:, i] = dist_euclid
    
    

gage_pr_spearman_testing = np.nan * np.zeros((len(final_df),len(final_df_validation)))


for i in range(len(final_df_validation)):
    pr_gage_sel = gage_pr_testing[:, i]

    for j in range(len(final_df)):
       
        pr_gage_train_sel = gage_pr_training[:, j]
        gage_pr_spearman_testing[j, i], _ = spearmanr(pr_gage_sel, pr_gage_train_sel, nan_policy='omit')
        
        

gage_dist_temperate_forest_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_wetland_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_cropland_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_barren_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_urban_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_water_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_BD_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_CLAY_SHALLOW_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_CLAY_DEEP_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_GRAV_SHALLOW_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_GRAV_DEEP_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_OC_SHALLOW_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_OC_DEEP_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_SAND_SHALLOW_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_SAND_DEEP_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_SILT_SHALLOW_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_SILT_DEEP_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_mean_elev_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_mean_slope_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_std_elev_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_std_slope_testing = np.full((len(final_df), len(final_df_validation)), np.nan)
gage_dist_area_km2_testing = np.full((len(final_df), len(final_df_validation)), np.nan)        
    



for i in range(len(final_df_validation)):
    d = pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Testing/out_5/{gage_id_testing[i]}.csv')
    gage_dist_temperate_forest_testing[:,i] =  gage_temperate_forest_training[:,0]-d['temperate_forest'][0]
    gage_dist_wetland_testing[:,i] = gage_wetland_training[:,0]-d['wetland'][0]
    gage_dist_cropland_testing[:,i] = gage_cropland_training[:,0]-d['cropland'][0]
    gage_dist_barren_testing[:,i] = gage_barren_training[:,0]-d['barren'][0]
    gage_dist_urban_testing[:,i] = gage_urban_training[:,0]-d['urban'][0]
    gage_dist_water_testing[:,i] = gage_water_training[:,0]-d['water'][0]
    gage_dist_BD_testing[:,i] = gage_BD_training[:,0]-d['BD'][0]
    gage_dist_CLAY_SHALLOW_testing[:,i] = gage_CLAY_SHALLOW_training[:,0]-d['CLAY_SHALLOW'][0]
    gage_dist_CLAY_DEEP_testing[:,i] = gage_CLAY_DEEP_training[:,0]-d['CLAY_DEEP'][0]
    gage_dist_GRAV_SHALLOW_testing[:,i] = gage_GRAV_SHALLOW_training[:,0]-d['GRAV_SHALLOW'][0]
    gage_dist_GRAV_DEEP_testing[:,i] = gage_GRAV_DEEP_training[:,0]-d['GRAV_DEEP'][0]
    gage_dist_OC_SHALLOW_testing[:,i] = gage_OC_SHALLOW_training[:,0]-d['OC_SHALLOW'][0]
    gage_dist_OC_DEEP_testing[:,i] = gage_OC_DEEP_training[:,0]-d['OC_DEEP'][0]
    gage_dist_SAND_SHALLOW_testing[:,i] = gage_SAND_SHALLOW_training[:,0]-d['SAND_SHALLOW'][0]
    gage_dist_SAND_DEEP_testing[:,i] = gage_SAND_DEEP_training[:,0]-d['SAND_DEEP'][0]
    gage_dist_SILT_SHALLOW_testing[:,i] = gage_SILT_SHALLOW_training[:,0]-d['SILT_SHALLOW'][0]
    gage_dist_SILT_DEEP_testing[:,i] = gage_SILT_DEEP_training[:,0]-d['SILT_DEEP'][0]
    gage_dist_mean_elev_testing[:,i] = gage_mean_elev_training[:,0]-d['mean_elev'][0]
    gage_dist_mean_slope_testing[:,i] = gage_mean_slope_training[:,0]-d['mean_slope'][0]
    gage_dist_std_elev_testing[:,i] = gage_std_elev_training[:,0]-d['std_elev'][0]
    gage_dist_std_slope_testing[:,i] = gage_std_slope_training[:,0]-d['std_slope'][0]
    gage_dist_area_km2_testing[:,i] = gage_area_km2_training[:,0]-d['area_km2'][0]
    
  


stand_pr_testing = (gage_pr_spearman_testing.flatten('F') - mean_pr) / std_pr
stand_dist_testing = (gage_dist_testing.flatten('F') - mean_dist) / std_dist
stand_temperate_forest_testing = (gage_dist_temperate_forest_testing.flatten('F') - mean_temperate_forest) / std_temperate_forest
stand_wetland_testing = (gage_dist_wetland_testing.flatten('F') - mean_wetland) / std_wetland
stand_cropland_testing = (gage_dist_cropland_testing.flatten('F') - mean_cropland) / std_cropland
stand_barren_testing = (gage_dist_barren_testing.flatten('F') - mean_barren) / std_barren
stand_urban_testing = (gage_dist_urban_testing.flatten('F') - mean_urban) / std_urban
stand_water_testing = (gage_dist_water_testing.flatten('F') - mean_water) / std_water
stand_BD_testing = (gage_dist_BD_testing.flatten('F') - mean_BD) / std_BD
stand_CLAY_SHALLOW_testing = (gage_dist_CLAY_SHALLOW_testing.flatten('F') - mean_CLAY_SHALLOW) / std_CLAY_SHALLOW
stand_CLAY_DEEP_testing = (gage_dist_CLAY_DEEP_testing.flatten('F') - mean_CLAY_DEEP) / std_CLAY_DEEP
stand_GRAV_SHALLOW_testing = (gage_dist_GRAV_SHALLOW_testing.flatten('F') - mean_GRAV_SHALLOW) / std_GRAV_SHALLOW
stand_GRAV_DEEP_testing = (gage_dist_GRAV_DEEP_testing.flatten('F') - mean_GRAV_DEEP) / std_GRAV_DEEP
stand_OC_SHALLOW_testing = (gage_dist_OC_SHALLOW_testing.flatten('F') - mean_OC_SHALLOW) / std_OC_SHALLOW
stand_OC_DEEP_testing = (gage_dist_OC_DEEP_testing.flatten('F') - mean_OC_DEEP) / std_OC_DEEP
stand_SAND_SHALLOW_testing = (gage_dist_SAND_SHALLOW_testing.flatten('F') - mean_SAND_SHALLOW) / std_SAND_SHALLOW
stand_SAND_DEEP_testing = (gage_dist_SAND_DEEP_testing.flatten('F') - mean_SAND_DEEP) / std_SAND_DEEP
stand_SILT_SHALLOW_testing = (gage_dist_SILT_SHALLOW_testing.flatten('F') - mean_SILT_SHALLOW) / std_SILT_SHALLOW
stand_SILT_DEEP_testing = (gage_dist_SILT_DEEP_testing.flatten('F') - mean_SILT_DEEP) / std_SILT_DEEP
stand_mean_elev_testing = (gage_dist_mean_elev_testing.flatten('F') - mean_mean_elev) / std_mean_elev
stand_mean_slope_testing = (gage_dist_mean_slope_testing.flatten('F') - mean_mean_slope) / std_mean_slope
stand_std_elev_testing = (gage_dist_std_elev_testing.flatten('F') - mean_std_elev) / std_std_elev
stand_std_slope_testing = (gage_dist_std_slope_testing.flatten('F') - mean_std_slope) / std_std_slope
stand_area_km2_testing = (gage_dist_area_km2_testing.flatten('F') - mean_area_km2) / std_area_km2


XX = np.column_stack((
    stand_pr_testing, stand_dist_testing,
    stand_temperate_forest_testing,
    stand_wetland_testing, stand_cropland_testing, stand_barren_testing, stand_urban_testing, stand_water_testing,
    stand_BD_testing, stand_CLAY_SHALLOW_testing,stand_CLAY_DEEP_testing,stand_GRAV_SHALLOW_testing, stand_GRAV_DEEP_testing, stand_OC_SHALLOW_testing, stand_OC_DEEP_testing, stand_SAND_SHALLOW_testing,stand_SAND_DEEP_testing, stand_SILT_SHALLOW_testing,stand_SILT_DEEP_testing,
    stand_mean_elev_testing, stand_mean_slope_testing, stand_std_elev_testing, stand_std_slope_testing, stand_area_km2_testing
))
    

################################################################################################

XX_valid=XX[~np.isnan(XX)]


x_test = scaler_x.transform(XX)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

# Evaluate the model
model.eval()
y_pred = model(x_test_tensor).detach().numpy()
y_pred = scaler_y.inverse_transform(y_pred)  # Inverse transform to original scale




y_pred_2_reshp=np.reshape(y_pred, (len(final_df), len(final_df_validation)),order='F')
s_r = np.argsort(-y_pred_2_reshp, axis=0)


rows, cols = s_r.shape  # Both are 318, 128

# Create column indices to match s_r
col_indices = np.tile(np.arange(cols), (rows, 1))  # shape (318, 128)

# Now pick from ypred2rshp using (s_r, col_indices)
correlations = y_pred_2_reshp[s_r, col_indices]  # shape (318, 128)


#Grab the top 5 donor indices for each site 

top_5_donors=s_r
top_5_correlations=correlations

#Go through the columns and create the donor and correlation datasets 

date_range = pd.date_range(start="1995-01-01", end="2013-12-31", freq="D")

weird_gauges=[]

def get_first_5_non_nan(row):
    # Drop NaNs and keep their column names
    non_nan_values = row.dropna().iloc[:5]  # Drop NaNs, get first 5
    indices = [row.index.get_loc(col) for col in non_nan_values.index]  # Get numeric column indices
 
    return indices

for i in range(len(final_df_validation)):
    df=pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Testing/out_5/{gage_id_testing[i]}.csv')
    
    donors= pd.DataFrame()
    correlations= pd.DataFrame()
    
    
    num_valid_donors = 0
    z = 0
    while num_valid_donors < 80 and z < top_5_donors.shape[0]:
        donor_id = gage_id_training[top_5_donors[z, i]]
        
        if donor_id == gage_id_training[i]:
            z += 1
            continue  # Skip if donor is same as target
        
        donor_series = pd.read_csv(
            f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/{donor_id}.csv'
        )['q']
    
        # Mask NaNs
        valid_mask = ~donor_series.isna()
        valid_values = donor_series[valid_mask]
    
        # Sort and compute ranks only on valid data
        sorted_vals = np.sort(valid_values)
        n = len(sorted_vals)
        ranks = np.searchsorted(sorted_vals, valid_values, side='right')
        probs = ranks / (n + 1)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
    
        # Convert to Z-scores
        z_scores_valid = norm.ppf(probs)
    
        # Create full z-score series, preserving NaNs
        z_scores = pd.Series(np.nan, index=donor_series.index)
        z_scores[valid_mask] = z_scores_valid
    
        # Store standardized values and correlation
        donors[f"Donor{num_valid_donors}"] = z_scores
        correlations[f"C{num_valid_donors}"] = np.repeat(top_5_correlations[z, i], len(donor_series))
    
        num_valid_donors += 1
        z += 1
    
       
    df['d_n1']=0
    df['d_n1']=0
    df['d_n2']=0
    df['d_n3']=0
    df['d_n4']=0
    df['d_n5']=0
       
    
    df['qz_n1']=0
    df['qz_n2']=0
    df['qz_n3']=0
    df['qz_n4']=0
    df['qz_n5']=0
    
    
    for t in range(len(date_range)):
     
     if donors.iloc[t,0:5].isna().any()==False:
       df.iloc[t,43:48]=donors.iloc[t,0:5]
       df.iloc[t,38:43]= correlations.iloc[t,0:5]
       
     if donors.iloc[t,0:5].isna().any()==True:
        indices = get_first_5_non_nan(donors.iloc[t])
        df.iloc[t,43:48]=donors.iloc[t,indices]
        df.iloc[t,38:43]= correlations.iloc[t,indices]
       
    
                
    if df.iloc[:,38:48].isna().any().any()==False: 
            df.to_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Testing/out_5_neighbors/{gage_id_testing[i]}.csv',index=False,na_rep="NaN")
    else:
            weird_gauges.append(i)


############Training Gauges###################


gage_pr_training = np.nan * np.zeros((num_day, len(final_df)))
gage_dist_training = np.nan * np.zeros((len(final_df),len(final_df)))


for i in range(len(final_df)):
    d = pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/{gage_id_training[i]}.csv')
    #d=d.iloc[0:3653,:] #Train over 2000-2010 only

    #d=d.iloc[11688:,:]
    gage_pr_training[:, i] = d['pr']
    dist_lat = (gage_lat_training - gage_lat_training[i]) ** 2
    dist_lon = (gage_lon_training - gage_lon_training[i]) ** 2
    dist_euclid = np.sqrt(dist_lat + dist_lon)

    gage_dist_training[:, i] = dist_euclid
    
    

gage_pr_spearman_training = np.nan * np.zeros((len(final_df),len(final_df)))


for i in range(len(final_df)):
    pr_gage_sel = gage_pr_training[:, i]

    for j in range(len(final_df)):
       
        pr_gage_train_sel = gage_pr_training[:, j]
        gage_pr_spearman_training[j, i], _ = spearmanr(pr_gage_sel, pr_gage_train_sel, nan_policy='omit')
        
        

gage_dist_temperate_forest_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_wetland_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_cropland_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_barren_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_urban_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_water_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_BD_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_CLAY_SHALLOW_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_CLAY_DEEP_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_GRAV_SHALLOW_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_GRAV_DEEP_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_OC_SHALLOW_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_OC_DEEP_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_SAND_SHALLOW_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_SAND_DEEP_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_SILT_SHALLOW_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_SILT_DEEP_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_mean_elev_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_mean_slope_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_std_elev_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_std_slope_training = np.full((len(final_df), len(final_df)), np.nan)
gage_dist_area_km2_training = np.full((len(final_df), len(final_df)), np.nan)        
    



for i in range(len(final_df)):
    d = pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/{gage_id_training[i]}.csv')
    gage_dist_temperate_forest_training[:,i] =  gage_temperate_forest_training[:,0]-d['temperate_forest'][0]
    gage_dist_wetland_training[:,i] = gage_wetland_training[:,0]-d['wetland'][0]
    gage_dist_cropland_training[:,i] = gage_cropland_training[:,0]-d['cropland'][0]
    gage_dist_barren_training[:,i] = gage_barren_training[:,0]-d['barren'][0]
    gage_dist_urban_training[:,i] = gage_urban_training[:,0]-d['urban'][0]
    gage_dist_water_training[:,i] = gage_water_training[:,0]-d['water'][0]
    gage_dist_BD_training[:,i] = gage_BD_training[:,0]-d['BD'][0]
    gage_dist_CLAY_SHALLOW_training[:,i] = gage_CLAY_SHALLOW_training[:,0]-d['CLAY_SHALLOW'][0]
    gage_dist_CLAY_DEEP_training[:,i] = gage_CLAY_DEEP_training[:,0]-d['CLAY_DEEP'][0]
    gage_dist_GRAV_SHALLOW_training[:,i] = gage_GRAV_SHALLOW_training[:,0]-d['GRAV_SHALLOW'][0]
    gage_dist_GRAV_DEEP_training[:,i] = gage_GRAV_DEEP_training[:,0]-d['GRAV_DEEP'][0]
    gage_dist_OC_SHALLOW_training[:,i] = gage_OC_SHALLOW_training[:,0]-d['OC_SHALLOW'][0]
    gage_dist_OC_DEEP_training[:,i] = gage_OC_DEEP_training[:,0]-d['OC_DEEP'][0]
    gage_dist_SAND_SHALLOW_training[:,i] = gage_SAND_SHALLOW_training[:,0]-d['SAND_SHALLOW'][0]
    gage_dist_SAND_DEEP_training[:,i] = gage_SAND_DEEP_training[:,0]-d['SAND_DEEP'][0]
    gage_dist_SILT_SHALLOW_training[:,i] = gage_SILT_SHALLOW_training[:,0]-d['SILT_SHALLOW'][0]
    gage_dist_SILT_DEEP_training[:,i] = gage_SILT_DEEP_training[:,0]-d['SILT_DEEP'][0]
    gage_dist_mean_elev_training[:,i] = gage_mean_elev_training[:,0]-d['mean_elev'][0]
    gage_dist_mean_slope_training[:,i] = gage_mean_slope_training[:,0]-d['mean_slope'][0]
    gage_dist_std_elev_training[:,i] = gage_std_elev_training[:,0]-d['std_elev'][0]
    gage_dist_std_slope_training[:,i] = gage_std_slope_training[:,0]-d['std_slope'][0]
    gage_dist_area_km2_training[:,i] = gage_area_km2_training[:,0]-d['area_km2'][0]
    
  


stand_pr_training = (gage_pr_spearman_training.flatten('F') - mean_pr) / std_pr
stand_dist_training = (gage_dist_training.flatten('F') - mean_dist) / std_dist
stand_temperate_forest_training = (gage_dist_temperate_forest_training.flatten('F') - mean_temperate_forest) / std_temperate_forest
stand_wetland_training = (gage_dist_wetland_training.flatten('F') - mean_wetland) / std_wetland
stand_cropland_training = (gage_dist_cropland_training.flatten('F') - mean_cropland) / std_cropland
stand_barren_training = (gage_dist_barren_training.flatten('F') - mean_barren) / std_barren
stand_urban_training = (gage_dist_urban_training.flatten('F') - mean_urban) / std_urban
stand_water_training = (gage_dist_water_training.flatten('F') - mean_water) / std_water
stand_BD_training = (gage_dist_BD_training.flatten('F') - mean_BD) / std_BD
stand_CLAY_SHALLOW_training = (gage_dist_CLAY_SHALLOW_training.flatten('F') - mean_CLAY_SHALLOW) / std_CLAY_SHALLOW
stand_CLAY_DEEP_training = (gage_dist_CLAY_DEEP_training.flatten('F') - mean_CLAY_DEEP) / std_CLAY_DEEP
stand_GRAV_SHALLOW_training = (gage_dist_GRAV_SHALLOW_training.flatten('F') - mean_GRAV_SHALLOW) / std_GRAV_SHALLOW
stand_GRAV_DEEP_training = (gage_dist_GRAV_DEEP_training.flatten('F') - mean_GRAV_DEEP) / std_GRAV_DEEP
stand_OC_SHALLOW_training = (gage_dist_OC_SHALLOW_training.flatten('F') - mean_OC_SHALLOW) / std_OC_SHALLOW
stand_OC_DEEP_training = (gage_dist_OC_DEEP_training.flatten('F') - mean_OC_DEEP) / std_OC_DEEP
stand_SAND_SHALLOW_training = (gage_dist_SAND_SHALLOW_training.flatten('F') - mean_SAND_SHALLOW) / std_SAND_SHALLOW
stand_SAND_DEEP_training = (gage_dist_SAND_DEEP_training.flatten('F') - mean_SAND_DEEP) / std_SAND_DEEP
stand_SILT_SHALLOW_training = (gage_dist_SILT_SHALLOW_training.flatten('F') - mean_SILT_SHALLOW) / std_SILT_SHALLOW
stand_SILT_DEEP_training = (gage_dist_SILT_DEEP_training.flatten('F') - mean_SILT_DEEP) / std_SILT_DEEP
stand_mean_elev_training = (gage_dist_mean_elev_training.flatten('F') - mean_mean_elev) / std_mean_elev
stand_mean_slope_training = (gage_dist_mean_slope_training.flatten('F') - mean_mean_slope) / std_mean_slope
stand_std_elev_training = (gage_dist_std_elev_training.flatten('F') - mean_std_elev) / std_std_elev
stand_std_slope_training = (gage_dist_std_slope_training.flatten('F') - mean_std_slope) / std_std_slope
stand_area_km2_training = (gage_dist_area_km2_training.flatten('F') - mean_area_km2) / std_area_km2


XX = np.column_stack((
    stand_pr_training, stand_dist_training,
    stand_temperate_forest_training,
    stand_wetland_training, stand_cropland_training, stand_barren_training, stand_urban_training, stand_water_training,
    stand_BD_training, stand_CLAY_SHALLOW_training,stand_CLAY_DEEP_training,stand_GRAV_SHALLOW_training, stand_GRAV_DEEP_training, stand_OC_SHALLOW_training, stand_OC_DEEP_training, stand_SAND_SHALLOW_training,stand_SAND_DEEP_training, stand_SILT_SHALLOW_training,stand_SILT_DEEP_training,
    stand_mean_elev_training, stand_mean_slope_training, stand_std_elev_training, stand_std_slope_training, stand_area_km2_training
))
    

#################################################################

x_test = scaler_x.transform(XX)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

# Evaluate the model
model.eval()
y_pred = model(x_test_tensor).detach().numpy()
y_pred = scaler_y.inverse_transform(y_pred)  # Inverse transform to original scale


y_pred_2_reshp=np.reshape(y_pred, (len(final_df), len(final_df)),order='F')
s_r = np.argsort(-y_pred_2_reshp, axis=0)


rows, cols = s_r.shape  # Both are len(final_df), 128

# Create column indices to match s_r
col_indices = np.tile(np.arange(cols), (rows, 1))  # shape (len(final_df), 128)

# Now pick from ypred2rshp using (s_r, col_indices)
correlations = y_pred_2_reshp[s_r, col_indices]  # shape (len(final_df), 128)



top_5_donors=s_r
top_5_correlations=correlations
##########################################################################################

for i in range(0,len(final_df)):
    df=pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/{gage_id_training[i]}.csv')
    
    donors= pd.DataFrame()
    correlations= pd.DataFrame()
    
    
    num_valid_donors = 0
    z = 0
    while num_valid_donors < len(final_df) and z < top_5_donors.shape[0]:
        donor_id = gage_id_training[top_5_donors[z, i]]
        
        if donor_id == gage_id_training[i]:
            z += 1
            continue  # Skip if donor is same as target
        
        donor_series = pd.read_csv(
            f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5/{donor_id}.csv'
        )['q']
    
        # Mask NaNs
        valid_mask = ~donor_series.isna()
        valid_values = donor_series[valid_mask]
    
        # Sort and compute ranks only on valid data
        sorted_vals = np.sort(valid_values)
        n = len(sorted_vals)
        ranks = np.searchsorted(sorted_vals, valid_values, side='right')
        probs = ranks / (n + 1)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
    
        # Convert to Z-scores
        z_scores_valid = norm.ppf(probs)
    
        # Create full z-score series, preserving NaNs
        z_scores = pd.Series(np.nan, index=donor_series.index)
        z_scores[valid_mask] = z_scores_valid
    
        # Store standardized values and correlation
        donors[f"Donor{num_valid_donors}"] = z_scores
        correlations[f"C{num_valid_donors}"] = np.repeat(top_5_correlations[z, i], len(donor_series))
    
        num_valid_donors += 1
        z += 1
    
       
    df['d_n1']=0
    df['d_n1']=0
    df['d_n2']=0
    df['d_n3']=0
    df['d_n4']=0
    df['d_n5']=0
       
    
    df['qz_n1']=0
    df['qz_n2']=0
    df['qz_n3']=0
    df['qz_n4']=0
    df['qz_n5']=0
    
    
    for t in range(len(date_range)):
     
     if donors.iloc[t,0:5].isna().any()==False:
       df.iloc[t,43:48]=donors.iloc[t,0:5]
       df.iloc[t,38:43]= correlations.iloc[t,0:5]
       
     if donors.iloc[t,0:5].isna().any()==True:
        indices = get_first_5_non_nan(donors.iloc[t])
        df.iloc[t,43:48]=donors.iloc[t,indices]
        df.iloc[t,38:43]= correlations.iloc[t,indices]
       
    
                
    if df.iloc[:,38:48].isna().any().any()==False: 
            df.to_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_5_neighbors/{gage_id_training[i]}.csv',index=False,na_rep="NaN")
    else:
            weird_gauges.append(i)


#Create text files with the validation and training gauge names 

file_path = "C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/fold_5_calibration.txt"

with open(file_path, "w") as file:
    for item in gage_id_training:
        file.write(str(item) + "\n")
        
        
file_path = "C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/fold_5_validation.txt"

with open(file_path, "w") as file:
    for item in gage_id_testing:
        file.write(str(item) + "\n")        
        
        
