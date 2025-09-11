
#Create a map with the validation points, colored by change in NSE. The marker denotes regulated or natural. We also add the training sites and expanded training sites. 


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import glob
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm

# Load shapefile
lakes = gpd.read_file('C:/Users/rg727/Documents/Great Lake Project/Runoff/GRIP-GL_lakes/StLawrenceBasins_Lake_EPSG_4326.shp')

# Load basins 

sixdomains = gpd.read_file('C:/Users/rg727/Documents/Great Lake Project/Runoff/GRIP-GL_seven_subdomains/GRIP-GL_seven_subdomains.shp')


# Load validation list and make a dataframe 

df_decade_1=np.loadtxt('C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/1950_1965_Training_Testing/1950_1965_gauges.txt', dtype=str) 


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


# Create GeoDataFrame for validation points 
final_df_decade_1['geometry'] = [Point(xy) for xy in zip(final_df_decade_1['Longitude'], final_df_decade_1['Latitude'])]
gdf_points = gpd.GeoDataFrame(final_df_decade_1, geometry='geometry', crs='EPSG:4326') 

regulated_indices=final_df_decade_1[final_df_decade_1['Type']=='Regulated'].index.tolist()
natural_indices=final_df_decade_1[final_df_decade_1['Type']=='Natural'].index.tolist()


# Separate regulated and naturalized points
regulated = gdf_points.loc[regulated_indices]
natural = gdf_points.loc[natural_indices]


# Plot
fig, ax = plt.subplots(figsize=(15, 10))
#lakes.plot(ax=ax, edgecolor='black', facecolor='lightblue')
sixdomains.plot(ax=ax, edgecolor='gray', facecolor='lightgray')


# Plot regulated points (triangles)
regulated.plot(
    ax=ax, 
    marker='^', 
    c='#bc6c25',
    label='Regulated',
    edgecolor='black',
    markersize=100
)


# Plot naturalized points (circles)
natural.plot(
    ax=ax, 
    marker='o', 
    c='#283618',
    label='Natural',
    edgecolor='black',
    markersize=100
)


plt.legend(fontsize=18)
plt.title('1950-1965',fontsize=20)
plt.xlabel('Longitude',fontsize=18)
plt.ylabel('Latitude',fontsize=18)
# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure1/Decade1.pdf")

