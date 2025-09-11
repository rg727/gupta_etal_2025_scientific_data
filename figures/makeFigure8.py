# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 12:05:54 2025

@author: rg727
"""




#Average Regulated

basin_id='04079000'
fold=1
training_data=pd.read_csv(f'C:/Users/rg727/Downloads/1995_2013_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv') 
area=pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_{fold}/{basin_id}.csv')['area_km2'][0]
training_data['pred']=training_data['pred']/3600/24/304.8*area*(3280*3280)
training_data['obs']=training_data['obs']/3600/24/304.8*area*(3280*3280)
fold=2
validation_data=pd.read_csv(f'C:/Users/rg727/Downloads/1995_2013_Training_Testing/Testing/out_{fold}_neighbors/OOS/lstm_nbr_pred_'+basin_id+'.csv')    
validation_data['pred']=validation_data['pred']/3600/24/304.8*area*(3280*3280)

fig, ax=plt.subplots()
plt.plot(date_range[5115:6210],validation_data['pred'][5115:6210], label='Prediction (Validation)', color='#219ebc', linestyle='--')

plt.plot(date_range[5115:6210],training_data['obs'][5115:6210], label='Observed', color='#fb8500', linestyle='-')

plt.plot(date_range[5115:6210],training_data['pred'][5115:6210], label='Prediction (Training)', color='#023047', linestyle='--')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Runoff (m$^3$/day)')
plt.title(f'Basin ID={basin_id} (Regulated)')

# Add a legend to distinguish the lines
#plt.legend()
plt.xticks(rotation=45, ha='right') # Rotate by 45 degrees and right-align
ax.tick_params(axis='both', which='major', labelsize=14)
# Display the plot
plt.tight_layout()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure6/US_regulated_average_KGE.pdf")


#Average Natural 

basin_id='02GC030'
fold=1
training_data=pd.read_csv(f'C:/Users/rg727/Downloads/1995_2013_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv') 
area=pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_{fold}/{basin_id}.csv')['area_km2'][0]
training_data['pred']=training_data['pred']/3600/24*area*1000
training_data['obs']=training_data['obs']/3600/24*area*1000
fold=3
validation_data=pd.read_csv(f'C:/Users/rg727/Downloads/1995_2013_Training_Testing/Testing/out_{fold}_neighbors/OOS/lstm_nbr_pred_'+basin_id+'.csv')    
validation_data['pred']=validation_data['pred']/3600/24*area*1000

fig, ax=plt.subplots()
plt.plot(date_range[5115:6210],validation_data['pred'][5115:6210], label='Prediction (Validation)', color='#219ebc', linestyle='--')

plt.plot(date_range[5115:6210],training_data['obs'][5115:6210], label='Observed', color='#fb8500', linestyle='-')

plt.plot(date_range[5115:6210],training_data['pred'][5115:6210], label='Prediction (Training)', color='#023047', linestyle='--')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Runoff (m$^3$/day)')
plt.title(f'Basin ID={basin_id} (Natural)')

# Add a legend to distinguish the lines
#plt.legend()
plt.xticks(rotation=45, ha='right') # Rotate by 45 degrees and right-align
ax.tick_params(axis='both', which='major', labelsize=14)
# Display the plot
plt.tight_layout()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure6/CA_natural_average_KGE.pdf")



#####################################################################################


#Worst Case Regulated

basin_id='02BE002'
fold=1
training_data=pd.read_csv(f'C:/Users/rg727/Downloads/1995_2013_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv') 
area=pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_{fold}/{basin_id}.csv')['area_km2'][0]
training_data['pred']=training_data['pred']/3600/24*area*1000
training_data['obs']=training_data['obs']/3600/24*area*1000
fold=2
validation_data=pd.read_csv(f'C:/Users/rg727/Downloads/1995_2013_Training_Testing/Testing/out_{fold}_neighbors/OOS/lstm_nbr_pred_'+basin_id+'.csv')    
validation_data['pred']=validation_data['pred']/3600/24*area*1000

fig, ax=plt.subplots()
plt.plot(date_range[5115:6210],validation_data['pred'][5115:6210], label='Prediction (Validation)', color='#219ebc', linestyle='--')

plt.plot(date_range[5115:6210],training_data['obs'][5115:6210], label='Observed', color='#fb8500', linestyle='-')

plt.plot(date_range[5115:6210],training_data['pred'][5115:6210], label='Prediction (Training)', color='#023047', linestyle='--')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Runoff (m$^3$/day)')
plt.title(f'Basin ID={basin_id} (Regulated)')

# Add a legend to distinguish the lines
#plt.legend()
plt.xticks(rotation=45, ha='right') # Rotate by 45 degrees and right-align
ax.tick_params(axis='both', which='major', labelsize=14)
# Display the plot
plt.tight_layout()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure6/CA_regulated_worst_KGE.pdf")



##################################


basin_id='04043140'
fold=1
training_data=pd.read_csv(f'C:/Users/rg727/Downloads/1995_2013_Training_Testing/Training/fold_{fold}_neighbors/prediction/lstm_nbr_pred_'+basin_id+'.csv') 
area=pd.read_csv(f'C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1995_2013_Training_Testing/Training/fold_{fold}/{basin_id}.csv')['area_km2'][0]
training_data['pred']=training_data['pred']/3600/24/304.8*area*(3280*3280)
training_data['obs']=training_data['obs']/3600/24/304.8*area*(3280*3280)
fold=2
validation_data=pd.read_csv(f'C:/Users/rg727/Downloads/1995_2013_Training_Testing/Testing/out_{fold}_neighbors/OOS/lstm_nbr_pred_'+basin_id+'.csv')    
validation_data['pred']=validation_data['pred']/3600/24/304.8*area*(3280*3280)

fig, ax=plt.subplots()
plt.plot(date_range[5115:6210],validation_data['pred'][5115:6210], label='Prediction (Validation)', color='#219ebc', linestyle='--')

plt.plot(date_range[5115:6210],training_data['obs'][5115:6210], label='Observed', color='#fb8500', linestyle='-')

plt.plot(date_range[5115:6210],training_data['pred'][5115:6210], label='Prediction (Training)', color='#023047', linestyle='--')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Runoff (m$^3$/day)')
plt.title(f'Basin ID={basin_id} (Natural)')

# Add a legend to distinguish the lines
#plt.legend()
plt.xticks(rotation=45, ha='right') # Rotate by 45 degrees and right-align
ax.tick_params(axis='both', which='major', labelsize=14)
# Display the plot
plt.tight_layout()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure6/US_natural_worst_KGE.pdf")

