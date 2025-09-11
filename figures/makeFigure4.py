# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:03:05 2025

@author: rg727
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 12:57:14 2025

@author: rg727
"""

import pandas as pd 
import numpy as np 
from functools import reduce
import matplotlib.pyplot as plt 
import pandas as pd 
import os
from dateutil import parser
from datetime import datetime
from pathlib import Path
import numpy as np
import glob
from datetime import datetime, timedelta
from scipy.stats import pearsonr




def kge(sim, obs):
    sim = np.array(sim)
    obs = np.array(obs)

    # Remove NaNs
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    sim = sim[mask]
    obs = obs[mask]

    if len(sim) == 0 or len(obs) == 0 or np.all(sim == sim[0]) or np.all(obs == obs[0]):
        return np.nan

    r, _ = pearsonr(sim, obs)
    beta = np.mean(sim) / np.mean(obs)
    gamma = (np.std(sim) / np.mean(sim)) / (np.std(obs) / np.mean(obs))

    kge_value = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    return kge_value




df=pd.read_csv("/scratch/bcqp/rg727/1950_1965_Training_Testing/compiled_gauges_no_duplicates.csv")

regulated_ids = list((df.loc[(df['Regulation'] == 'Regulated') | (df['Reference'] == 'Non-ref'),'ID']).astype(str))
natural_ids = list((df.loc[(df['Regulation'] == 'Natural') | (df['Reference'] == 'Ref'),'ID']).astype(str))


#Canada expanded gauges- all natural 
natural_CA_gauges = list(pd.read_csv('/scratch/bcqp/rg727/1950_1965_Training_Testing/canada_expanded_ref_gauges.csv')['STAID'])

#US Natural Gagues 

natural_US_gauges = list(pd.read_csv('/scratch/bcqp/rg727/1950_1965_Training_Testing/us_expanded_ref_gauges.csv')['STAID'])

#US Non-Ref Gagues 

reg_US_gauges = list(pd.read_csv('/scratch/bcqp/rg727/1950_1965_Training_Testing/us_expanded_nonref_gauges.csv')['STAID'])


combined_natural_gauges=natural_ids+natural_CA_gauges+natural_US_gauges

combined_regulated_gauges=regulated_ids+reg_US_gauges


combined_natural_gauges_dataframe=pd.DataFrame({'ID': combined_natural_gauges})
combined_natural_gauges_dataframe['Type']='Natural'

combined_regulated_gauges_dataframe=pd.DataFrame({'ID': combined_regulated_gauges})
combined_regulated_gauges_dataframe['Type']='Regulated'

combined_gauges=pd.concat([combined_natural_gauges_dataframe,combined_regulated_gauges_dataframe])

combined_gauges['ID']=combined_gauges['ID'].astype(str)
#Cross reference with the training and validation sets 
combined_gauges['ID_clean'] = combined_gauges['ID'].astype(str).str.lstrip('0')
combined_gauges = combined_gauges.drop_duplicates(subset='ID_clean')


####1950_1965####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        lstm_nn_oos[k]=kge(df_cleaned['simulation'],df_cleaned['target'])
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "KGE"
lstm_nn_oos['KGE']=abs(lstm_nn_oos['KGE'])
            
            
lstm_nn_oos['Decade']='1950-1965'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/KGE/KGE_train_nn_1950_1965.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        lstm_nn_oos[k]=kge(df_cleaned['simulation'],df_cleaned['target'])
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "KGE"
lstm_nn_oos['KGE']=abs(lstm_nn_oos['KGE'])
            
            

lstm_nn_oos['Decade']='1950-1965'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/KGE/KGE_train_clim_1950_1965.csv',index=False,na_rep="NaN")


#######################################################################################################################################################


####1995_2013####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1995_2013_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        lstm_nn_oos[k]=kge(df_cleaned['simulation'],df_cleaned['target'])
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "KGE"
lstm_nn_oos['KGE']=abs(lstm_nn_oos['KGE'])
            
            
lstm_nn_oos['Decade']='1995-2013'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/KGE/KGE_train_nn_1995_2013.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1995_2013_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        lstm_nn_oos[k]=kge(df_cleaned['simulation'],df_cleaned['target'])
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "KGE"
lstm_nn_oos['KGE']=abs(lstm_nn_oos['KGE'])
            
            

lstm_nn_oos['Decade']='1995-2013'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/KGE/KGE_train_clim_1995_2013.csv',index=False,na_rep="NaN")



#################################################################################################################################

####1965_1980####



#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path

            file_path =f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        lstm_nn_oos[k]=kge(df_cleaned['simulation'],df_cleaned['target'])
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "KGE"
lstm_nn_oos['KGE']=abs(lstm_nn_oos['KGE'])
            
            
lstm_nn_oos['Decade']='1965-1980'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/KGE/KGE_train_nn_1965_1980.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==1:
                file_path = f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        lstm_nn_oos[k]=kge(df_cleaned['simulation'],df_cleaned['target'])
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "KGE"
lstm_nn_oos['KGE']=abs(lstm_nn_oos['KGE'])
            
            

lstm_nn_oos['Decade']='1965-1980'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/KGE/KGE_train_clim_1965_1980.csv',index=False,na_rep="NaN")


####################################################################################################################################



####1980_1995####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==5:
                file_path = f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}_neighbors/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        lstm_nn_oos[k]=kge(df_cleaned['simulation'],df_cleaned['target'])
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "KGE"
lstm_nn_oos['KGE']=abs(lstm_nn_oos['KGE'])
            
            
lstm_nn_oos['Decade']='1980-1995'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/KGE/KGE_train_nn_1980_1995.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==5:
                file_path = f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        lstm_nn_oos[k]=kge(df_cleaned['simulation'],df_cleaned['target'])
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "KGE"
lstm_nn_oos['KGE']=abs(lstm_nn_oos['KGE'])
            
            

lstm_nn_oos['Decade']='1980-1995'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/KGE/KGE_train_clim_1980_1995.csv',index=False,na_rep="NaN")



#########################################################PBIAS#############################################################################
####1950_1965####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "PBIAS"

            
            
lstm_nn_oos['Decade']='1950-1965'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/PBIAS/PBIAS_train_nn_1950_1965.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "PBIAS"

            
            

lstm_nn_oos['Decade']='1950-1965'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/PBIAS/PBIAS_train_clim_1950_1965.csv',index=False,na_rep="NaN")


#######################################################################################################################################################


####1995_2013####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1995_2013_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "PBIAS"

            
            
lstm_nn_oos['Decade']='1995-2013'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/PBIAS/PBIAS_train_nn_1995_2013.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1995_2013_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "PBIAS"

            
            

lstm_nn_oos['Decade']='1995-2013'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/PBIAS/PBIAS_train_clim_1995_2013.csv',index=False,na_rep="NaN")



#################################################################################################################################

####1965_1980####



#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path

            file_path =f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "PBIAS"

            
            
lstm_nn_oos['Decade']='1965-1980'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/PBIAS/PBIAS_train_nn_1965_1980.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==1:
                file_path = f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "PBIAS"

            
            

lstm_nn_oos['Decade']='1965-1980'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/PBIAS/PBIAS_train_clim_1965_1980.csv',index=False,na_rep="NaN")


####################################################################################################################################



####1980_1995####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==5:
                file_path = f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}_neighbors/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "PBIAS"

            
            
lstm_nn_oos['Decade']='1980-1995'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/PBIAS/PBIAS_train_nn_1980_1995.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==5:
                file_path = f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
        
       
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "PBIAS"

            
            

lstm_nn_oos['Decade']='1980-1995'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/PBIAS/PBIAS_train_clim_1980_1995.csv',index=False,na_rep="NaN")


#####################################################FLV#################################################################

####1950_1965####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']>=0.8]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']>=0.8]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "LBIAS"

            
            
lstm_nn_oos['Decade']='1950-1965'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/LBIAS/LBIAS_train_nn_1950_1965.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']>=0.8]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']>=0.8]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "LBIAS"

            
            

lstm_nn_oos['Decade']='1950-1965'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/LBIAS/LBIAS_train_clim_1950_1965.csv',index=False,na_rep="NaN")


#######################################################################################################################################################


####1995_2013####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1995_2013_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']>=0.8]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']>=0.8]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "LBIAS"

            
            
lstm_nn_oos['Decade']='1995-2013'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/LBIAS/LBIAS_train_nn_1995_2013.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1995_2013_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']>=0.8]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']>=0.8]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "LBIAS"

            
            

lstm_nn_oos['Decade']='1995-2013'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/LBIAS/LBIAS_train_clim_1995_2013.csv',index=False,na_rep="NaN")



#################################################################################################################################

####1965_1980####



#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path

            file_path =f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']>=0.8]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']>=0.8]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "LBIAS"

            
            
lstm_nn_oos['Decade']='1965-1980'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/LBIAS/LBIAS_train_nn_1965_1980.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==1:
                file_path = f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']>=0.8]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']>=0.8]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "LBIAS"

            
            

lstm_nn_oos['Decade']='1965-1980'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/LBIAS/LBIAS_train_clim_1965_1980.csv',index=False,na_rep="NaN")


####################################################################################################################################



####1980_1995####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==5:
                file_path = f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}_neighbors/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']>=0.8]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']>=0.8]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "LBIAS"

            
            
lstm_nn_oos['Decade']='1980-1995'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/LBIAS/LBIAS_train_nn_1980_1995.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==5:
                file_path = f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']>=0.8]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']>=0.8]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
        
       
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "LBIAS"

            
            

lstm_nn_oos['Decade']='1980-1995'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/LBIAS/LBIAS_train_clim_1980_1995.csv',index=False,na_rep="NaN")



#########################################################HBIAS#############################################################################

####1950_1965####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']<= 0.02]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']<= 0.02]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "HBIAS"

            
            
lstm_nn_oos['Decade']='1950-1965'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/HBIAS/HBIAS_train_nn_1950_1965.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']<= 0.02]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']<= 0.02]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "HBIAS"

            
            

lstm_nn_oos['Decade']='1950-1965'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/HBIAS/HBIAS_train_clim_1950_1965.csv',index=False,na_rep="NaN")


#######################################################################################################################################################


####1995_2013####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1995_2013_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']<= 0.02]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']<= 0.02]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "HBIAS"

            
            
lstm_nn_oos['Decade']='1995-2013'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/HBIAS/HBIAS_train_nn_1995_2013.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            file_path = f'/scratch/bcqp/rg727/1995_2013_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_{basin_id}.csv'
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']<= 0.02]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']<= 0.02]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "HBIAS"

            
            

lstm_nn_oos['Decade']='1995-2013'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/HBIAS/HBIAS_train_clim_1995_2013.csv',index=False,na_rep="NaN")



#################################################################################################################################

####1965_1980####



#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path

            file_path =f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']<= 0.02]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']<= 0.02]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "HBIAS"

            
            
lstm_nn_oos['Decade']='1965-1980'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/HBIAS/HBIAS_train_nn_1965_1980.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==1:
                file_path = f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']<= 0.02]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']<= 0.02]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "HBIAS"

            
            

lstm_nn_oos['Decade']='1965-1980'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/HBIAS/HBIAS_train_clim_1965_1980.csv',index=False,na_rep="NaN")


####################################################################################################################################



####1980_1995####

#training gauges- calculate high flow bias 

folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    

lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==5:
                file_path = f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}_neighbors/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']<= 0.02]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']<= 0.02]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "HBIAS"

            
            
lstm_nn_oos['Decade']='1980-1995'
lstm_nn_oos['Label']='NN'


lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/HBIAS/HBIAS_train_nn_1980_1995.csv',index=False,na_rep="NaN")


folds = range(1, 6)  # Folds 1 to 5

for fold in folds:
    training_gauges=np.loadtxt(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_calibration.txt',dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
lstm_nn_oos=np.zeros((len(training_gauges),1))    
for k,basin_id in enumerate(final_df_training['ID']):
        columns=[]
        for fold in folds:
        # Define the file path
            if fold==5:
                file_path = f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv' 
        
            else:
                file_path =f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv' 
         
        # Check if the file exists
            if os.path.exists(file_path):
                lstm_nn = pd.read_csv(file_path)
                columns.append(lstm_nn['pred'])
            else:
                print(f"File not found for basin {basin_id} in fold {fold}, skipping.")
        combined_df = pd.concat(columns, axis=1)
        combined_df['average'] = combined_df.mean(axis=1)
            
        target_valid=lstm_nn['obs'][4018:5844]
        output_valid=combined_df['average'][4018:5844]
        nona_dataframe=pd.DataFrame({'target':target_valid,'simulation':output_valid})
        df_cleaned = nona_dataframe.dropna()
        
        sort_target = np.sort(df_cleaned['target'])
        nonexceedence_target = 1-np.arange(1.,len(sort_target)+1) / (len(sort_target) +1)
        
        bias_dataframe=pd.DataFrame({'sort':sort_target,'nonexceedence':nonexceedence_target})
        
        bias_dataframe_HFV=bias_dataframe[bias_dataframe['nonexceedence']<= 0.02]
        
        
        sort_out = np.sort(df_cleaned['simulation'])
        nonexceedence_out = 1-np.arange(1.,len(sort_out)+1) / (len(sort_out) +1)
        bias_dataframe_out=pd.DataFrame({'sort':sort_out,'nonexceedence':nonexceedence_out})
        
        bias_dataframe_HFV_out=bias_dataframe_out[bias_dataframe_out['nonexceedence']<= 0.02]
        
        
        lstm_nn_oos[k]=np.sum(bias_dataframe_HFV_out['sort']-bias_dataframe_HFV['sort'])/np.sum(bias_dataframe_HFV['sort'])*100
        
        
       
lstm_nn_oos=pd.DataFrame(lstm_nn_oos)
lstm_nn_oos['Type']=final_df_training['Type']
lstm_nn_oos['ID']=final_df_training['ID']
    
lstm_nn_oos.columns.values[0] = "HBIAS"

            
            

lstm_nn_oos['Decade']='1980-1995'
lstm_nn_oos['Label']='Clim'

lstm_nn_oos.to_csv('/scratch/bcqp/rg727/Metrics/Training/HBIAS/HBIAS_train_clim_1980_1995.csv',index=False,na_rep="NaN")


################################################################################################################################


HBIAS_train_nn_1950_1965=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/HBIAS/HBIAS_train_nn_1950_1965.csv")
HBIAS_train_clim_1950_1965=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/HBIAS/HBIAS_train_clim_1950_1965.csv")


HBIAS_train_nn_1965_1980=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/HBIAS/HBIAS_train_nn_1965_1980.csv")
HBIAS_train_clim_1965_1980=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/HBIAS/HBIAS_train_clim_1965_1980.csv")

HBIAS_train_nn_1965_1980=HBIAS_train_nn_1965_1980.rename(columns={'FHV': 'HBIAS'})
HBIAS_train_clim_1965_1980=HBIAS_train_clim_1965_1980.rename(columns={'FHV': 'HBIAS'})



HBIAS_train_nn_1980_1995=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/HBIAS/HBIAS_train_nn_1980_1995.csv")
HBIAS_train_clim_1980_1995=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/HBIAS/HBIAS_train_clim_1980_1995.csv")


HBIAS_train_nn_1995_2013=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/HBIAS/HBIAS_train_nn_1995_2013.csv")
HBIAS_train_clim_1995_2013=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/HBIAS/HBIAS_train_clim_1995_2013.csv")



HBIAS_train_nn_1950_1965['Decade']='1950-1965'
HBIAS_train_nn_1950_1965['Label']='NN'

HBIAS_train_clim_1950_1965['Decade']='1950-1965'
HBIAS_train_clim_1950_1965['Label']='Clim'

HBIAS_train_nn_1965_1980['Decade']='1966-1980'
HBIAS_train_nn_1965_1980['Label']='NN'

HBIAS_train_clim_1965_1980['Decade']='1966-1980'
HBIAS_train_clim_1965_1980['Label']='Clim'

HBIAS_train_nn_1980_1995['Decade']='1981-1995'
HBIAS_train_nn_1980_1995['Label']='NN'

HBIAS_train_clim_1980_1995['Decade']='1981-1995'
HBIAS_train_clim_1980_1995['Label']='Clim'

HBIAS_train_nn_1995_2013['Decade']='1996-2013'
HBIAS_train_nn_1995_2013['Label']='NN'

HBIAS_train_clim_1995_2013['Decade']='1996-2013'
HBIAS_train_clim_1995_2013['Label']='Clim'


data=pd.concat([HBIAS_train_nn_1950_1965,HBIAS_train_clim_1950_1965,HBIAS_train_nn_1965_1980,HBIAS_train_clim_1965_1980,HBIAS_train_nn_1980_1995,HBIAS_train_clim_1980_1995,HBIAS_train_nn_1995_2013,HBIAS_train_clim_1995_2013])

data['HBIAS']=abs(data['HBIAS'])


data_cleaned = data.dropna()
data_cleaned=data_cleaned.reset_index(drop=True)

custom_palette = ["#ccd5ae", "#bc6c25"]

# Step 2: Plot using seaborn with outliers removed
fig, ax=plt.subplots(figsize=(10, 6))
sns.boxplot(data=data_cleaned, x='Decade', y='HBIAS', hue='Label', showfliers=False, palette=custom_palette,legend=False)
#plt.title('FLV by Type and Source')
#plt.xlabel('1950')
#plt.ylabel('FLV')
plt.tight_layout()
ax.tick_params(axis='both', which='major', labelsize=24)
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/HBIAS_new.pdf")









