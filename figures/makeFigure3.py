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







#Validation 

folds = range(1, 6)  # Folds 1 to 5
all_clim, all_nn = [], []

for fold in folds:
    if fold ==5:
        clim_path = f"/scratch/bcqp/rg727/1980_1995_Training_Testing/Testing/out_{fold}/OOS/seed2/nse_validation_cv_check.csv"
        nn_path = f"/scratch/bcqp/rg727/1980_1995_Training_Testing/Testing/out_{fold}_neighbors/OOS/seed2/nse_validation_cv_check.csv"

    else:   
        clim_path = f"/scratch/bcqp/rg727/1980_1995_Training_Testing/Testing/out_{fold}/OOS/nse_validation_cv_check.csv"
        nn_path = f"/scratch/bcqp/rg727/1980_1995_Training_Testing/Testing/out_{fold}_neighbors/OOS/nse_validation_cv_check.csv"
        

    clim_df = pd.read_csv(clim_path)
    nn_df = pd.read_csv(nn_path)
    
    clim_df['gage_id']=np.loadtxt(f"/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_validation.txt",dtype=str)
    nn_df['gage_id']=np.loadtxt(f"/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_validation.txt",dtype=str)
    
    clim_df['Fold'] = fold
    nn_df['Fold'] = fold
    
    clim_df['ID_clean'] = clim_df['gage_id'].astype(str).str.lstrip('0')
    nn_df['ID_clean'] = nn_df['gage_id'].astype(str).str.lstrip('0')


    
    validation_gauges=np.loadtxt(f"/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_validation.txt",dtype=str)


    validation_gauges = pd.DataFrame({'ID': validation_gauges})


    validation_gauges['ID_clean'] = validation_gauges['ID'].astype(str).str.lstrip('0')


    final_df_validation = validation_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
    
    clim_df['Type']=final_df_validation['Type']
    nn_df['Type']=final_df_validation['Type']
    all_clim.append(clim_df)
    all_nn.append(nn_df)
    

LSTM_clim_all = pd.concat(all_clim, ignore_index=True)
LSTM_NN_all = pd.concat(all_nn, ignore_index=True)


#Training 

folds = range(1, 6)  # Folds 1 to 5
all_clim_training, all_nn_training = [], []

for fold in folds:
    training_gauges=np.loadtxt(f"/scratch/bcqp/rg727/1980_1995_Training_Testing/fold_{fold}_calibration.txt",dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)

    
    
    lstm_clim_oot=np.zeros((len(training_gauges),1))
    lstm_nn_oot=np.zeros((len(training_gauges),1))

    for k,basin_id in enumerate(training_gauges['ID']):
    
        if fold==5:
            lstm_clim_data=pd.read_csv(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv') 
    
        else:
            lstm_clim_data=pd.read_csv(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv') 
            
        target_valid=lstm_clim_data['obs'][4018:5844]
        output_valid=lstm_clim_data['pred'][4018:5844]
    
        lstm_clim_oot[k] = 1 - np.sum(np.power(target_valid - output_valid,2))/np.sum(np.power(target_valid - target_valid.mean(),2))
    
        if fold==5:
            lstm_nn_data=pd.read_csv(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}_neighbors/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv') 

        else:  
            lstm_nn_data=pd.read_csv(f'/scratch/bcqp/rg727/1980_1995_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv') 

        target_valid=lstm_nn_data['obs'][4018:5844]
        output_valid=lstm_nn_data['pred'][4018:5844]

        lstm_nn_oot[k] = 1 - np.sum(np.power(target_valid - output_valid,2))/np.sum(np.power(target_valid - target_valid.mean(),2))
        
    lstm_nn_oot=pd.DataFrame(lstm_nn_oot)
    lstm_clim_oot=pd.DataFrame(lstm_clim_oot)
    lstm_nn_oot['Type']=final_df_training['Type']
    lstm_clim_oot['Type']=final_df_training['Type']
    
    lstm_clim_oot['ID']=final_df_training['ID']
    lstm_nn_oot['ID']=final_df_training['ID']
    
    lstm_clim_oot.columns.values[0] = "nse"
    lstm_nn_oot.columns.values[0] = "nse"

        
    all_clim_training.append(lstm_clim_oot)
    all_nn_training.append(lstm_nn_oot)
        


# Step 1: Prepare the dataframes to only include necessary columns
dfs = [df[['nse', 'Type', 'ID']] for df in all_clim_training]

# Step 2: Assign a suffix to each dataframe to track its source
for i, df in enumerate(dfs):
    df.rename(columns={'nse': f'nse_{i}', 'Type': f'Type_{i}'}, inplace=True)

# Step 3: Merge all 5 dataframes on 'ID' using OUTER join to keep all IDs
merged_all = reduce(lambda left, right: pd.merge(left, right, on='ID', how='outer'), dfs)

# Step 4: Identify all NSE columns
nse_cols = [col for col in merged_all.columns if col.startswith("nse_")]

# Step 5: Compute average NSE across available (non-NaN) values
merged_all['nse_avg'] = merged_all[nse_cols].mean(axis=1, skipna=True)

# Step 6: Pick the first non-null 'Type' across folds
type_cols = [col for col in merged_all.columns if col.startswith("Type_")]
merged_all['Type'] = merged_all[type_cols].bfill(axis=1).iloc[:, 0]

# Step 7: Final averaged DataFrame
avg_LSTM_clim = merged_all[['ID', 'nse_avg', 'Type']].rename(columns={'nse_avg': 'nse'})


#Do the same for the NN model 

# Step 1: Prepare the dataframes to only include necessary columns
dfs = [df[['nse', 'Type', 'ID']] for df in all_nn_training]

# Step 2: Assign a suffix to each dataframe to track its source
for i, df in enumerate(dfs):
    df.rename(columns={'nse': f'nse_{i}', 'Type': f'Type_{i}'}, inplace=True)

# Step 3: Merge all 5 dataframes on 'ID' using OUTER join to keep all IDs
merged_all = reduce(lambda left, right: pd.merge(left, right, on='ID', how='outer'), dfs)

# Step 4: Identify all NSE columns
nse_cols = [col for col in merged_all.columns if col.startswith("nse_")]

# Step 5: Compute average NSE across available (non-NaN) values
merged_all['nse_avg'] = merged_all[nse_cols].mean(axis=1, skipna=True)

# Step 6: Pick the first non-null 'Type' across folds
type_cols = [col for col in merged_all.columns if col.startswith("Type_")]
merged_all['Type'] = merged_all[type_cols].bfill(axis=1).iloc[:, 0]

# Step 7: Final averaged DataFrame
avg_LSTM_nn = merged_all[['ID', 'nse_avg', 'Type']].rename(columns={'nse_avg': 'nse'})


#########Now Plot#########################################

LSTM_NN_all.to_csv('/scratch/bcqp/rg727/1980_1995_Training_Testing/oos_nse.csv',index=False,na_rep="NaN")
avg_LSTM_nn.to_csv('/scratch/bcqp/rg727/1980_1995_Training_Testing/avg_training_nse.csv',index=False,na_rep="NaN")

LSTM_clim_all.to_csv('/scratch/bcqp/rg727/1980_1995_Training_Testing/oos_nse_clim.csv',index=False,na_rep="NaN")
avg_LSTM_clim.to_csv('/scratch/bcqp/rg727/1980_1995_Training_Testing/avg_training_nse_clim.csv',index=False,na_rep="NaN")


#############################################################################################################################
#Validation 

folds = range(1, 6)  # Folds 1 to 5
all_clim, all_nn = [], []

for fold in folds:
    if fold ==1:
        clim_path = f"/scratch/bcqp/rg727/1965_1980_Training_Testing/Testing/out_{fold}/OOS/seed2/nse_validation_cv_check.csv"
        nn_path = f"/scratch/bcqp/rg727/1965_1980_Training_Testing/Testing/out_{fold}_neighbors/OOS/nse_validation_cv_check.csv"

    else:   
        clim_path = f"/scratch/bcqp/rg727/1965_1980_Training_Testing/Testing/out_{fold}/OOS/nse_validation_cv_check.csv"
        nn_path = f"/scratch/bcqp/rg727/1965_1980_Training_Testing/Testing/out_{fold}_neighbors/OOS/nse_validation_cv_check.csv"
        

    clim_df = pd.read_csv(clim_path)
    nn_df = pd.read_csv(nn_path)
    
    clim_df['gage_id']=np.loadtxt(f"/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_validation.txt",dtype=str)
    nn_df['gage_id']=np.loadtxt(f"/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_validation.txt",dtype=str)
    
    clim_df['Fold'] = fold
    nn_df['Fold'] = fold
    
    clim_df['ID_clean'] = clim_df['gage_id'].astype(str).str.lstrip('0')
    nn_df['ID_clean'] = nn_df['gage_id'].astype(str).str.lstrip('0')


    
    validation_gauges=np.loadtxt(f"/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_validation.txt",dtype=str)


    validation_gauges = pd.DataFrame({'ID': validation_gauges})


    validation_gauges['ID_clean'] = validation_gauges['ID'].astype(str).str.lstrip('0')


    final_df_validation = validation_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
    
    clim_df['Type']=final_df_validation['Type']
    nn_df['Type']=final_df_validation['Type']
    all_clim.append(clim_df)
    all_nn.append(nn_df)
    

LSTM_clim_all = pd.concat(all_clim, ignore_index=True)
LSTM_NN_all = pd.concat(all_nn, ignore_index=True)


#Training 

folds = range(1, 6)  # Folds 1 to 5
all_clim_training, all_nn_training = [], []

for fold in folds:
    training_gauges=np.loadtxt(f"/scratch/bcqp/rg727/1965_1980_Training_Testing/fold_{fold}_calibration.txt",dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)

    
    
    lstm_clim_oot=np.zeros((len(training_gauges),1))
    lstm_nn_oot=np.zeros((len(training_gauges),1))

    for k,basin_id in enumerate(training_gauges['ID']):
    
        if fold==1:
            lstm_clim_data=pd.read_csv(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}/prediction/seed2/lstm_nbr_pred_'+basin_id+'.csv') 
    
        else:
            lstm_clim_data=pd.read_csv(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv') 
            
        target_valid=lstm_clim_data['obs'][4018:5844]
        output_valid=lstm_clim_data['pred'][4018:5844]
    
        lstm_clim_oot[k] = 1 - np.sum(np.power(target_valid - output_valid,2))/np.sum(np.power(target_valid - target_valid.mean(),2))
    
        if fold==1:
            lstm_nn_data=pd.read_csv(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv') 

        else:  
            lstm_nn_data=pd.read_csv(f'/scratch/bcqp/rg727/1965_1980_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv') 

        target_valid=lstm_nn_data['obs'][4018:5844]
        output_valid=lstm_nn_data['pred'][4018:5844]

        lstm_nn_oot[k] = 1 - np.sum(np.power(target_valid - output_valid,2))/np.sum(np.power(target_valid - target_valid.mean(),2))
        
    lstm_nn_oot=pd.DataFrame(lstm_nn_oot)
    lstm_clim_oot=pd.DataFrame(lstm_clim_oot)
    lstm_nn_oot['Type']=final_df_training['Type']
    lstm_clim_oot['Type']=final_df_training['Type']
    
    lstm_clim_oot['ID']=final_df_training['ID']
    lstm_nn_oot['ID']=final_df_training['ID']
    
    lstm_clim_oot.columns.values[0] = "nse"
    lstm_nn_oot.columns.values[0] = "nse"

        
    all_clim_training.append(lstm_clim_oot)
    all_nn_training.append(lstm_nn_oot)
        


# Step 1: Prepare the dataframes to only include necessary columns
dfs = [df[['nse', 'Type', 'ID']] for df in all_clim_training]

# Step 2: Assign a suffix to each dataframe to track its source
for i, df in enumerate(dfs):
    df.rename(columns={'nse': f'nse_{i}', 'Type': f'Type_{i}'}, inplace=True)

# Step 3: Merge all 5 dataframes on 'ID' using OUTER join to keep all IDs
merged_all = reduce(lambda left, right: pd.merge(left, right, on='ID', how='outer'), dfs)

# Step 4: Identify all NSE columns
nse_cols = [col for col in merged_all.columns if col.startswith("nse_")]

# Step 5: Compute average NSE across available (non-NaN) values
merged_all['nse_avg'] = merged_all[nse_cols].mean(axis=1, skipna=True)

# Step 6: Pick the first non-null 'Type' across folds
type_cols = [col for col in merged_all.columns if col.startswith("Type_")]
merged_all['Type'] = merged_all[type_cols].bfill(axis=1).iloc[:, 0]

# Step 7: Final averaged DataFrame
avg_LSTM_clim = merged_all[['ID', 'nse_avg', 'Type']].rename(columns={'nse_avg': 'nse'})


#Do the same for the NN model 

# Step 1: Prepare the dataframes to only include necessary columns
dfs = [df[['nse', 'Type', 'ID']] for df in all_nn_training]

# Step 2: Assign a suffix to each dataframe to track its source
for i, df in enumerate(dfs):
    df.rename(columns={'nse': f'nse_{i}', 'Type': f'Type_{i}'}, inplace=True)

# Step 3: Merge all 5 dataframes on 'ID' using OUTER join to keep all IDs
merged_all = reduce(lambda left, right: pd.merge(left, right, on='ID', how='outer'), dfs)

# Step 4: Identify all NSE columns
nse_cols = [col for col in merged_all.columns if col.startswith("nse_")]

# Step 5: Compute average NSE across available (non-NaN) values
merged_all['nse_avg'] = merged_all[nse_cols].mean(axis=1, skipna=True)

# Step 6: Pick the first non-null 'Type' across folds
type_cols = [col for col in merged_all.columns if col.startswith("Type_")]
merged_all['Type'] = merged_all[type_cols].bfill(axis=1).iloc[:, 0]

# Step 7: Final averaged DataFrame
avg_LSTM_nn = merged_all[['ID', 'nse_avg', 'Type']].rename(columns={'nse_avg': 'nse'})


LSTM_NN_all.to_csv('/scratch/bcqp/rg727/1965_1980_Training_Testing/oos_nse.csv',index=False,na_rep="NaN")
avg_LSTM_nn.to_csv('/scratch/bcqp/rg727/1965_1980_Training_Testing/avg_training_nse.csv',index=False,na_rep="NaN")

LSTM_clim_all.to_csv('/scratch/bcqp/rg727/1965_1980_Training_Testing/oos_nse_clim.csv',index=False,na_rep="NaN")
avg_LSTM_clim.to_csv('/scratch/bcqp/rg727/1965_1980_Training_Testing/avg_training_nse_clim.csv',index=False,na_rep="NaN")

#############################################################################################################################
#Validation 

folds = range(1, 6)  # Folds 1 to 5
all_clim, all_nn = [], []

for fold in folds:   
    clim_path = f"/scratch/bcqp/rg727/1950_1965_Training_Testing/Testing/out_{fold}/OOS/nse_validation_cv_check.csv"
    nn_path = f"/scratch/bcqp/rg727/1950_1965_Training_Testing/Testing/out_{fold}_neighbors/OOS/nse_validation_cv_check.csv"
        

    clim_df = pd.read_csv(clim_path)
    nn_df = pd.read_csv(nn_path)
    
    clim_df['gage_id']=np.loadtxt(f"/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_validation.txt",dtype=str)
    nn_df['gage_id']=np.loadtxt(f"/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_validation.txt",dtype=str)
    
    clim_df['Fold'] = fold
    nn_df['Fold'] = fold
    
    clim_df['ID_clean'] = clim_df['gage_id'].astype(str).str.lstrip('0')
    nn_df['ID_clean'] = nn_df['gage_id'].astype(str).str.lstrip('0')


    
    validation_gauges=np.loadtxt(f"/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_validation.txt",dtype=str)


    validation_gauges = pd.DataFrame({'ID': validation_gauges})


    validation_gauges['ID_clean'] = validation_gauges['ID'].astype(str).str.lstrip('0')


    final_df_validation = validation_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
    
    clim_df['Type']=final_df_validation['Type']
    nn_df['Type']=final_df_validation['Type']
    all_clim.append(clim_df)
    all_nn.append(nn_df)
    

LSTM_clim_all = pd.concat(all_clim, ignore_index=True)
LSTM_NN_all = pd.concat(all_nn, ignore_index=True)


#Training 

folds = range(1, 6)  # Folds 1 to 5
all_clim_training, all_nn_training = [], []

for fold in folds:
    training_gauges=np.loadtxt(f"/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_{fold}_calibration.txt",dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)

    
    
    lstm_clim_oot=np.zeros((len(training_gauges),1))
    lstm_nn_oot=np.zeros((len(training_gauges),1))

    for k,basin_id in enumerate(training_gauges['ID']):
    
        lstm_clim_data=pd.read_csv(f'/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv') 
            
        target_valid=lstm_clim_data['obs'][4018:5844]
        output_valid=lstm_clim_data['pred'][4018:5844]
    
        lstm_clim_oot[k] = 1 - np.sum(np.power(target_valid - output_valid,2))/np.sum(np.power(target_valid - target_valid.mean(),2))

        lstm_nn_data=pd.read_csv(f'/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv') 

        target_valid=lstm_nn_data['obs'][4018:5844]
        output_valid=lstm_nn_data['pred'][4018:5844]

        lstm_nn_oot[k] = 1 - np.sum(np.power(target_valid - output_valid,2))/np.sum(np.power(target_valid - target_valid.mean(),2))
        
    lstm_nn_oot=pd.DataFrame(lstm_nn_oot)
    lstm_clim_oot=pd.DataFrame(lstm_clim_oot)
    lstm_nn_oot['Type']=final_df_training['Type']
    lstm_clim_oot['Type']=final_df_training['Type']
    
    lstm_clim_oot['ID']=final_df_training['ID']
    lstm_nn_oot['ID']=final_df_training['ID']
    
    lstm_clim_oot.columns.values[0] = "nse"
    lstm_nn_oot.columns.values[0] = "nse"

        
    all_clim_training.append(lstm_clim_oot)
    all_nn_training.append(lstm_nn_oot)
        


# Step 1: Prepare the dataframes to only include necessary columns
dfs = [df[['nse', 'Type', 'ID']] for df in all_clim_training]

# Step 2: Assign a suffix to each dataframe to track its source
for i, df in enumerate(dfs):
    df.rename(columns={'nse': f'nse_{i}', 'Type': f'Type_{i}'}, inplace=True)

# Step 3: Merge all 5 dataframes on 'ID' using OUTER join to keep all IDs
merged_all = reduce(lambda left, right: pd.merge(left, right, on='ID', how='outer'), dfs)

# Step 4: Identify all NSE columns
nse_cols = [col for col in merged_all.columns if col.startswith("nse_")]

# Step 5: Compute average NSE across available (non-NaN) values
merged_all['nse_avg'] = merged_all[nse_cols].mean(axis=1, skipna=True)

# Step 6: Pick the first non-null 'Type' across folds
type_cols = [col for col in merged_all.columns if col.startswith("Type_")]
merged_all['Type'] = merged_all[type_cols].bfill(axis=1).iloc[:, 0]

# Step 7: Final averaged DataFrame
avg_LSTM_clim = merged_all[['ID', 'nse_avg', 'Type']].rename(columns={'nse_avg': 'nse'})


#Do the same for the NN model 

# Step 1: Prepare the dataframes to only include necessary columns
dfs = [df[['nse', 'Type', 'ID']] for df in all_nn_training]

# Step 2: Assign a suffix to each dataframe to track its source
for i, df in enumerate(dfs):
    df.rename(columns={'nse': f'nse_{i}', 'Type': f'Type_{i}'}, inplace=True)

# Step 3: Merge all 5 dataframes on 'ID' using OUTER join to keep all IDs
merged_all = reduce(lambda left, right: pd.merge(left, right, on='ID', how='outer'), dfs)

# Step 4: Identify all NSE columns
nse_cols = [col for col in merged_all.columns if col.startswith("nse_")]

# Step 5: Compute average NSE across available (non-NaN) values
merged_all['nse_avg'] = merged_all[nse_cols].mean(axis=1, skipna=True)

# Step 6: Pick the first non-null 'Type' across folds
type_cols = [col for col in merged_all.columns if col.startswith("Type_")]
merged_all['Type'] = merged_all[type_cols].bfill(axis=1).iloc[:, 0]

# Step 7: Final averaged DataFrame
avg_LSTM_nn = merged_all[['ID', 'nse_avg', 'Type']].rename(columns={'nse_avg': 'nse'})


LSTM_NN_all.to_csv('/scratch/bcqp/rg727/1950_1965_Training_Testing/oos_nse.csv',index=False,na_rep="NaN")
avg_LSTM_nn.to_csv('/scratch/bcqp/rg727/1950_1965_Training_Testing/avg_training_nse.csv',index=False,na_rep="NaN")

LSTM_clim_all.to_csv('/scratch/bcqp/rg727/1950_1965_Training_Testing/oos_nse_clim.csv',index=False,na_rep="NaN")
avg_LSTM_clim.to_csv('/scratch/bcqp/rg727/1950_1965_Training_Testing/avg_training_nse_clim.csv',index=False,na_rep="NaN")

#############################################################################################################################

folds = range(1, 6)  # Folds 1 to 5
all_clim, all_nn = [], []

for fold in folds:   
    clim_path = f"/scratch/bcqp/rg727/1995_2013_Training_Testing/Testing/out_{fold}/OOS/nse_validation_cv_check.csv"
    nn_path = f"/scratch/bcqp/rg727/1995_2013_Training_Testing/Testing/out_{fold}_neighbors/OOS/nse_validation_cv_check.csv"
        

    clim_df = pd.read_csv(clim_path)
    nn_df = pd.read_csv(nn_path)
    
    clim_df['gage_id']=np.loadtxt(f"/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_validation.txt",dtype=str)
    nn_df['gage_id']=np.loadtxt(f"/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_validation.txt",dtype=str)
    
    clim_df['Fold'] = fold
    nn_df['Fold'] = fold
    
    clim_df['ID_clean'] = clim_df['gage_id'].astype(str).str.lstrip('0')
    nn_df['ID_clean'] = nn_df['gage_id'].astype(str).str.lstrip('0')


    
    validation_gauges=np.loadtxt(f"/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_validation.txt",dtype=str)


    validation_gauges = pd.DataFrame({'ID': validation_gauges})


    validation_gauges['ID_clean'] = validation_gauges['ID'].astype(str).str.lstrip('0')


    final_df_validation = validation_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)
    
    
    clim_df['Type']=final_df_validation['Type']
    nn_df['Type']=final_df_validation['Type']
    all_clim.append(clim_df)
    all_nn.append(nn_df)
    

LSTM_clim_all = pd.concat(all_clim, ignore_index=True)
LSTM_NN_all = pd.concat(all_nn, ignore_index=True)


#Training 

folds = range(1, 6)  # Folds 1 to 5
all_clim_training, all_nn_training = [], []

for fold in folds:
    training_gauges=np.loadtxt(f"/scratch/bcqp/rg727/1995_2013_Training_Testing/fold_{fold}_calibration.txt",dtype=str)


    training_gauges = pd.DataFrame({'ID': training_gauges})


    training_gauges['ID_clean'] = training_gauges['ID'].astype(str).str.lstrip('0')


    final_df_training = training_gauges.merge(
    combined_gauges[['ID_clean', 'Type']],
    on='ID_clean',
    how='left'
)

    
    
    lstm_clim_oot=np.zeros((len(training_gauges),1))
    lstm_nn_oot=np.zeros((len(training_gauges),1))

    for k,basin_id in enumerate(training_gauges['ID']):
    
        lstm_clim_data=pd.read_csv(f'/scratch/bcqp/rg727/1995_2013_Training_Testing/Training/fold_{fold}/prediction/lstm_nbr_pred_'+basin_id+'.csv') 
            
        target_valid=lstm_clim_data['obs'][4018:5844]
        output_valid=lstm_clim_data['pred'][4018:5844]
    
        lstm_clim_oot[k] = 1 - np.sum(np.power(target_valid - output_valid,2))/np.sum(np.power(target_valid - target_valid.mean(),2))

        lstm_nn_data=pd.read_csv(f'/scratch/bcqp/rg727/1995_2013_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv') 

        target_valid=lstm_nn_data['obs'][4018:5844]
        output_valid=lstm_nn_data['pred'][4018:5844]

        lstm_nn_oot[k] = 1 - np.sum(np.power(target_valid - output_valid,2))/np.sum(np.power(target_valid - target_valid.mean(),2))
        
    lstm_nn_oot=pd.DataFrame(lstm_nn_oot)
    lstm_clim_oot=pd.DataFrame(lstm_clim_oot)
    lstm_nn_oot['Type']=final_df_training['Type']
    lstm_clim_oot['Type']=final_df_training['Type']
    
    lstm_clim_oot['ID']=final_df_training['ID']
    lstm_nn_oot['ID']=final_df_training['ID']
    
    lstm_clim_oot.columns.values[0] = "nse"
    lstm_nn_oot.columns.values[0] = "nse"

        
    all_clim_training.append(lstm_clim_oot)
    all_nn_training.append(lstm_nn_oot)
        


# Step 1: Prepare the dataframes to only include necessary columns
dfs = [df[['nse', 'Type', 'ID']] for df in all_clim_training]

# Step 2: Assign a suffix to each dataframe to track its source
for i, df in enumerate(dfs):
    df.rename(columns={'nse': f'nse_{i}', 'Type': f'Type_{i}'}, inplace=True)

# Step 3: Merge all 5 dataframes on 'ID' using OUTER join to keep all IDs
merged_all = reduce(lambda left, right: pd.merge(left, right, on='ID', how='outer'), dfs)

# Step 4: Identify all NSE columns
nse_cols = [col for col in merged_all.columns if col.startswith("nse_")]

# Step 5: Compute average NSE across available (non-NaN) values
merged_all['nse_avg'] = merged_all[nse_cols].mean(axis=1, skipna=True)

# Step 6: Pick the first non-null 'Type' across folds
type_cols = [col for col in merged_all.columns if col.startswith("Type_")]
merged_all['Type'] = merged_all[type_cols].bfill(axis=1).iloc[:, 0]

# Step 7: Final averaged DataFrame
avg_LSTM_clim = merged_all[['ID', 'nse_avg', 'Type']].rename(columns={'nse_avg': 'nse'})


#Do the same for the NN model 

# Step 1: Prepare the dataframes to only include necessary columns
dfs = [df[['nse', 'Type', 'ID']] for df in all_nn_training]

# Step 2: Assign a suffix to each dataframe to track its source
for i, df in enumerate(dfs):
    df.rename(columns={'nse': f'nse_{i}', 'Type': f'Type_{i}'}, inplace=True)

# Step 3: Merge all 5 dataframes on 'ID' using OUTER join to keep all IDs
merged_all = reduce(lambda left, right: pd.merge(left, right, on='ID', how='outer'), dfs)

# Step 4: Identify all NSE columns
nse_cols = [col for col in merged_all.columns if col.startswith("nse_")]

# Step 5: Compute average NSE across available (non-NaN) values
merged_all['nse_avg'] = merged_all[nse_cols].mean(axis=1, skipna=True)

# Step 6: Pick the first non-null 'Type' across folds
type_cols = [col for col in merged_all.columns if col.startswith("Type_")]
merged_all['Type'] = merged_all[type_cols].bfill(axis=1).iloc[:, 0]

# Step 7: Final averaged DataFrame
avg_LSTM_nn = merged_all[['ID', 'nse_avg', 'Type']].rename(columns={'nse_avg': 'nse'})


LSTM_NN_all.to_csv('/scratch/bcqp/rg727/1995_2013_Training_Testing/oos_nse.csv',index=False,na_rep="NaN")
avg_LSTM_nn.to_csv('/scratch/bcqp/rg727/1995_2013_Training_Testing/avg_training_nse.csv',index=False,na_rep="NaN")

LSTM_clim_all.to_csv('/scratch/bcqp/rg727/1995_2013_Training_Testing/oos_nse_clim.csv',index=False,na_rep="NaN")
avg_LSTM_clim.to_csv('/scratch/bcqp/rg727/1995_2013_Training_Testing/avg_training_nse_clim.csv',index=False,na_rep="NaN")

#############################################Make Plots###########################################################################

#Validation 

LSTM_NN_all=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure3/Data/1980_1995/oos_nse.csv")
avg_LSTM_nn=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure3/Data/1980_1995/avg_training_nse.csv")

LSTM_clim_all=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure3/Data/1980_1995/oos_nse_clim.csv")
avg_LSTM_clim=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure3/Data/1980_1995/avg_training_nse_clim.csv")




fig, axs = plt.subplots(2, 1, figsize=(7, 12))  # Create a 2x2 grid
axs = axs.flatten()

# --- Panel (a) ---
temp = np.sort(LSTM_clim_all['test'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[0].plot(temp, ranks, marker='o', color='#bc6c25', label='LSTM_clim')

temp =np.sort(LSTM_NN_all['test'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[0].plot(temp, ranks, marker='o', color='#283618', label='LSTM_donor')

axs[0].set_title("Out of Sample Validation")
axs[0].set_xlabel("NSE")
axs[0].set_ylabel("Rank")
#axs[2].legend()
axs[0].set_xlim(0, 1)


# --- Panel (b) ---
temp =  np.sort(avg_LSTM_clim['nse'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[1].plot(temp, ranks, marker='o', color='#bc6c25', label='LSTM_clim')

temp = np.sort(avg_LSTM_nn['nse'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[1].plot(temp, ranks, marker='o', color='#283618', label='LSTM_donor')

axs[1].set_title("Out of Time Validation")
axs[1].set_xlabel("NSE")
axs[1].set_ylabel("Rank")
#axs[3].legend()
axs[1].set_xlim(0, 1)

axs[0].tick_params(axis='both', which='major', labelsize=24)
axs[1].tick_params(axis='both', which='major', labelsize=24)
# Final layout
plt.tight_layout()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure3/1980_1995.pdf")

