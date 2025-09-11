# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 13:10:24 2025

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
import seaborn as sns 
from scipy.stats import pearsonr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import glob
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors

# Load shapefile
lakes = gpd.read_file('C:/Users/rg727/Documents/Great Lake Project/Runoff/GRIP-GL_lakes/StLawrenceBasins_Lake_EPSG_4326.shp')

# Load basins 

sixdomains = gpd.read_file('C:/Users/rg727/Documents/Great Lake Project/Runoff/GRIP-GL_seven_subdomains/GRIP-GL_seven_subdomains.shp')

#This script calculates peak flow bias (Yilmaz, 2008) for the training and validation data 


#Calculate the high flow bias requires determining the indices where the exceedance probabilities are lower than 0.02 (2% highest flows)
#and then calculate the % deviation basically 

# Load validation list and make a dataframe 

df_decade_1=np.loadtxt('C:/Users/rg727/Downloads/1995_2013_Training_Testing/1995_2013_gauges.txt', dtype=str) 


df_decade_1_final = pd.DataFrame({'ID': df_decade_1})



#add the latitude/longitude and regulated/non-regulated information 

df=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations/compiled_gauges_no_duplicates.csv")

regulated_ids = list((df.loc[(df['Regulation'] == 'Regulated') | (df['Reference'] == 'Non-ref'),'ID']).astype(str))
natural_ids = list((df.loc[(df['Regulation'] == 'Natural') | (df['Reference'] == 'Ref'),'ID']).astype(str))


#Canada expanded gauges- all natural 
natural_CA_gauges = list(pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/canada_expanded_ref_gauges.csv')['STAID'])

#US Natural Gagues 

natural_US_gauges = list(pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/us_expanded_ref_gauges.csv')['STAID'])

#US Non-Ref Gagues 

reg_US_gauges = list(pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/us_expanded_nonref_gauges.csv')['STAID'])


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

#df['ID_clean'] = df['ID'].astype(str).str.lstrip('0')
all_data['ID_clean'] = all_data['ID'].astype(str).str.lstrip('0')

all_data = all_data.drop_duplicates(subset='ID_clean')

# Step 3: Merge on the cleaned ID
final_df = combined_gauges.merge(
    all_data[['ID_clean', 'Latitude', 'Longitude']],
    on='ID_clean',
    how='left'
)

# Optional: drop the temp 'ID_clean' column or rearrange
#final_df = final_df.drop(columns='ID_clean')

# Step 3: Merge on the cleaned ID
final_df = combined_gauges.merge(
    all_data[['ID_clean', 'Latitude', 'Longitude']],
    on='ID_clean',
    how='left'
)

df_decade_1_final['ID_clean']=df_decade_1_final['ID'].astype(str).str.lstrip('0')


final_df_decade_1 = df_decade_1_final.merge(
    final_df[['ID_clean', 'Type', 'Latitude', 'Longitude']],
    on='ID_clean',
    how='left'
)

##################################################################################################################################


KGE_train_nn_1995_2013=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/KGE/KGE_train_nn_1995_2013.csv")
KGE_val_nn_1995_2013=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/LSTM_Comparison_Box_Plots/Data/1995_2013/KGE_val_nn_1995_2013.csv")



diff_df = pd.DataFrame(KGE_train_nn_1995_2013['ID'])
diff_df['KGE_diff'] = KGE_train_nn_1995_2013['KGE'] - KGE_val_nn_1995_2013['KGE']


final_df_decade_1_merged = diff_df.merge(
    final_df_decade_1[['ID', 'Type', 'Latitude', 'Longitude']],
    on='ID',
    how='left'
)


diff_df=final_df_decade_1_merged

# Create GeoDataFrame for validation points 
diff_df['geometry'] = [Point(xy) for xy in zip(diff_df['Longitude'], diff_df['Latitude'])]
gdf_points = gpd.GeoDataFrame(diff_df, geometry='geometry', crs='EPSG:4326') 

regulated_indices=diff_df[diff_df['Type']=='Regulated'].index.tolist()
natural_indices=diff_df[diff_df['Type']=='Natural'].index.tolist()


# Separate regulated and naturalized points
regulated = gdf_points.loc[regulated_indices]
natural = gdf_points.loc[natural_indices]


#regulated median
training_reg_mean=np.nanmedian(regulated['KGE_diff'])
#natural 
traning_natural_mean=np.nanmedian(natural['KGE_diff'])
# Plot
fig, ax = plt.subplots(figsize=(15, 10))
sixdomains.plot(ax=ax, edgecolor='gray', facecolor='lightgray')
lakes.plot(ax=ax, edgecolor='black', facecolor='lightblue')

# Set up diverging colormap centered at 0
div_cmap = plt.get_cmap('seismic_r')  # or 'coolwarm', 'bwr', etc.
norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)



# Plot regulated points (triangles)
regulated.plot(
    ax=ax, 
    marker='^', 
    column=regulated['KGE_diff'],
    cmap=div_cmap,
    norm=norm,
    label='Regulated',
    edgecolor='black', 
    markersize=100
)


# Plot naturalized points (circles)
natural.plot(
    ax=ax, 
    marker='o', 
    column=natural['KGE_diff'],
    cmap=div_cmap,
    norm=norm,
    label='Natural',
    edgecolor='black', 
    markersize=100
)


# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks([])  # Remove x-axis (longitude) ticks
ax.set_yticks([])  # Remove y-axis (latitude) ticks    

sm = plt.cm.ScalarMappable(cmap=div_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation='horizontal',  # ⬅️ Set to horizontal
    fraction=0.046,             # ⬅️ Adjust size
    pad=0.04                    # ⬅️ Adjust spacing from plot
)

#cbar.set_label('KGE_diff', weight='bold', fontsize=14)
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/spatial_figures/KGE_diff_train_val_new.pdf")

##############################################################################################################################

#######################Create a bar plot#########################

#Training 

colors = ['#283618', '#bc6c25']

df = pd.DataFrame({'lab':['Natural', 'Regulated'], 'val':[0.10,0.13]})
ax = df.plot.bar(x='lab', y='val', rot=0,color=colors)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend_.remove()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/spatial_figures/KGE_diff_train_val_new_barplot.pdf")


##########################################################################################################

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 13:28:35 2025

@author: rg727
"""

BIAS_train_nn_1995_2013=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/PBIAS/PBIAS_train_nn_1995_2013.csv")
BIAS_val_nn_1995_2013=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/LSTM_Comparison_Box_Plots/Metrics/Bias/BIAS_val_nn_1995_2013.csv")



diff_df = pd.DataFrame(BIAS_train_nn_1995_2013['ID'])
diff_df['BIAS_diff'] = BIAS_val_nn_1995_2013['BIAS'] - BIAS_train_nn_1995_2013['PBIAS']


final_df_decade_1_merged = diff_df.merge(
    final_df_decade_1[['ID', 'Type', 'Latitude', 'Longitude']],
    on='ID',
    how='left'
)


diff_df=final_df_decade_1_merged

# Create GeoDataFrame for validation points 
diff_df['geometry'] = [Point(xy) for xy in zip(diff_df['Longitude'], diff_df['Latitude'])]
gdf_points = gpd.GeoDataFrame(diff_df, geometry='geometry', crs='EPSG:4326') 

regulated_indices=diff_df[diff_df['Type']=='Regulated'].index.tolist()
natural_indices=diff_df[diff_df['Type']=='Natural'].index.tolist()


# Separate regulated and naturalized points
regulated = gdf_points.loc[regulated_indices]
natural = gdf_points.loc[natural_indices]


#regulated median
training_reg_mean=np.nanmedian(regulated['BIAS_diff'])
#natural 
traning_natural_mean=np.nanmedian(natural['BIAS_diff'])
# Plot
fig, ax = plt.subplots(figsize=(15, 10))
sixdomains.plot(ax=ax, edgecolor='gray', facecolor='lightgray')
lakes.plot(ax=ax, edgecolor='black', facecolor='lightblue')

# Set up diverging colormap centered at 0
div_cmap = plt.get_cmap('seismic_r')  # or 'coolwarm', 'bwr', etc.
norm = mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)



# Plot regulated points (triangles)
regulated.plot(
    ax=ax, 
    marker='^', 
    column=regulated['BIAS_diff'],
    cmap=div_cmap,
    norm=norm,
    label='Regulated',
    edgecolor='black', 
    markersize=100
)


# Plot naturalized points (circles)
natural.plot(
    ax=ax, 
    marker='o', 
    column=natural['BIAS_diff'],
    cmap=div_cmap,
    norm=norm,
    label='Natural',
    edgecolor='black', 
    markersize=100
)


# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks([])  # Remove x-axis (longitude) ticks
ax.set_yticks([])  # Remove y-axis (latitude) ticks    

sm = plt.cm.ScalarMappable(cmap=div_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation='horizontal',  # ⬅️ Set to horizontal
    fraction=0.046,             # ⬅️ Adjust size
    pad=0.04                    # ⬅️ Adjust spacing from plot
)

#cbar.set_label('BIAS_diff', weight='bold', fontsize=14)
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/spatial_figures/BIAS_diff_train_val_new.pdf")

##############################################################################################################################

#######################Create a bar plot#########################

#Training 

colors = ['#283618', '#bc6c25']

df = pd.DataFrame({'lab':['Natural', 'Regulated'], 'val':[18,15.7]})
ax = df.plot.bar(x='lab', y='val', rot=0,color=colors)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend_.remove()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/spatial_figures/BIAS_diff_train_val_new_barplot.pdf")


#######################################################################################################
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 13:28:35 2025

@author: rg727
"""

LBIAS_train_nn_1995_2013=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/LBIAS/LBIAS_train_nn_1995_2013.csv")
LBIAS_val_nn_1995_2013=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/LSTM_Comparison_Box_Plots/Metrics/LBIAS/LBIAS_val_nn_1995_2013.csv")



diff_df = pd.DataFrame(LBIAS_train_nn_1995_2013['ID'])
diff_df['LBIAS_diff'] = LBIAS_val_nn_1995_2013['LBIAS'] - LBIAS_train_nn_1995_2013['LBIAS']


final_df_decade_1_merged = diff_df.merge(
    final_df_decade_1[['ID', 'Type', 'Latitude', 'Longitude']],
    on='ID',
    how='left'
)


diff_df=final_df_decade_1_merged

# Create GeoDataFrame for validation points 
diff_df['geometry'] = [Point(xy) for xy in zip(diff_df['Longitude'], diff_df['Latitude'])]
gdf_points = gpd.GeoDataFrame(diff_df, geometry='geometry', crs='EPSG:4326') 

regulated_indices=diff_df[diff_df['Type']=='Regulated'].index.tolist()
natural_indices=diff_df[diff_df['Type']=='Natural'].index.tolist()


# Separate regulated and naturalized points
regulated = gdf_points.loc[regulated_indices]
natural = gdf_points.loc[natural_indices]


#regulated median
training_reg_mean=np.nanmedian(regulated['LBIAS_diff'])
#natural 
traning_natural_mean=np.nanmedian(natural['LBIAS_diff'])
# Plot
fig, ax = plt.subplots(figsize=(15, 10))
sixdomains.plot(ax=ax, edgecolor='gray', facecolor='lightgray')
lakes.plot(ax=ax, edgecolor='black', facecolor='lightblue')

# Set up diverging colormap centered at 0
div_cmap = plt.get_cmap('seismic_r')  # or 'coolwarm', 'bwr', etc.
norm = mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)



# Plot regulated points (triangles)
regulated.plot(
    ax=ax, 
    marker='^', 
    column=regulated['LBIAS_diff'],
    cmap=div_cmap,
    norm=norm,
    label='Regulated',
    edgecolor='black', 
    markersize=100
)


# Plot naturalized points (circles)
natural.plot(
    ax=ax, 
    marker='o', 
    column=natural['LBIAS_diff'],
    cmap=div_cmap,
    norm=norm,
    label='Natural',
    edgecolor='black', 
    markersize=100
)


# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks([])  # Remove x-axis (longitude) ticks
ax.set_yticks([])  # Remove y-axis (latitude) ticks    

sm = plt.cm.ScalarMappable(cmap=div_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation='horizontal',  # ⬅️ Set to horizontal
    fraction=0.046,             # ⬅️ Adjust size
    pad=0.04                    # ⬅️ Adjust spacing from plot
)

#cbar.set_label('LBIAS_diff', weight='bold', fontsize=14)
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/spatial_figures/LBIAS_diff_train_val_new.pdf")

##############################################################################################################################

#######################Create a bar plot#########################

#Training 

colors = ['#283618', '#bc6c25']

df = pd.DataFrame({'lab':['Natural', 'Regulated'], 'val':[61,60]})
ax = df.plot.bar(x='lab', y='val', rot=0,color=colors)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend_.remove()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/spatial_figures/LBIAS_diff_train_val_new_barplot.pdf")


# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 13:28:35 2025

@author: rg727
"""

HBIAS_train_nn_1995_2013=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/HBIAS/HBIAS_train_nn_1995_2013.csv")
HBIAS_val_nn_1995_2013=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/LSTM_Comparison_Box_Plots/Metrics/HBIAS/HBIAS_val_nn_1995_2013.csv")



diff_df = pd.DataFrame(HBIAS_train_nn_1995_2013['ID'])
diff_df['HBIAS_diff'] = HBIAS_val_nn_1995_2013['HBIAS'] - HBIAS_train_nn_1995_2013['HBIAS']


final_df_decade_1_merged = diff_df.merge(
    final_df_decade_1[['ID', 'Type', 'Latitude', 'Longitude']],
    on='ID',
    how='left'
)


diff_df=final_df_decade_1_merged

# Create GeoDataFrame for validation points 
diff_df['geometry'] = [Point(xy) for xy in zip(diff_df['Longitude'], diff_df['Latitude'])]
gdf_points = gpd.GeoDataFrame(diff_df, geometry='geometry', crs='EPSG:4326') 

regulated_indices=diff_df[diff_df['Type']=='Regulated'].index.tolist()
natural_indices=diff_df[diff_df['Type']=='Natural'].index.tolist()


# Separate regulated and naturalized points
regulated = gdf_points.loc[regulated_indices]
natural = gdf_points.loc[natural_indices]


#regulated median
training_reg_mean=np.nanmedian(regulated['HBIAS_diff'])
#natural 
traning_natural_mean=np.nanmedian(natural['HBIAS_diff'])
# Plot
fig, ax = plt.subplots(figsize=(15, 10))
sixdomains.plot(ax=ax, edgecolor='gray', facecolor='lightgray')
lakes.plot(ax=ax, edgecolor='black', facecolor='lightblue')

# Set up diverging colormap centered at 0
div_cmap = plt.get_cmap('seismic_r')  # or 'coolwarm', 'bwr', etc.
norm = mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)



# Plot regulated points (triangles)
regulated.plot(
    ax=ax, 
    marker='^', 
    column=regulated['HBIAS_diff'],
    cmap=div_cmap,
    norm=norm,
    label='Regulated',
    edgecolor='black', 
    markersize=100
)


# Plot naturalized points (circles)
natural.plot(
    ax=ax, 
    marker='o', 
    column=natural['HBIAS_diff'],
    cmap=div_cmap,
    norm=norm,
    label='Natural',
    edgecolor='black', 
    markersize=100
)


# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks([])  # Remove x-axis (longitude) ticks
ax.set_yticks([])  # Remove y-axis (latitude) ticks    

sm = plt.cm.ScalarMappable(cmap=div_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation='horizontal',  # ⬅️ Set to horizontal
    fraction=0.046,             # ⬅️ Adjust size
    pad=0.04                    # ⬅️ Adjust spacing from plot
)

#cbar.set_label('HBIAS_diff', weight='bold', fontsize=14)
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/spatial_figures/HBIAS_diff_train_val_new.pdf")

##############################################################################################################################

#######################Create a bar plot#########################

#Training 

colors = ['#283618', '#bc6c25']

df = pd.DataFrame({'lab':['Natural', 'Regulated'], 'val':[33,24]})
ax = df.plot.bar(x='lab', y='val', rot=0,color=colors)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend_.remove()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/spatial_figures/HBIAS_diff_train_val_new_barplot.pdf")


