# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:37:09 2025

@author: rg727
"""



#Can we do a cross-referencing with GAGES-II dataset

NWIS_data=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/US_HUC_04/usgs_gauges_great_lakes.csv",dtype={'site_no': str})
GAGES_II_data=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/Dataset1_BasinID/Dataset1_BasinID/BasinID.txt",dtype={'STAID': str})


NWIS_data.rename(columns={'site_no': 'STAID'}, inplace=True)



merged_data = pd.merge(NWIS_data, GAGES_II_data, on='STAID', how='inner')
reference_gauges = merged_data[merged_data['CLASS'] == 'Ref']

non_reference_gauges = merged_data[merged_data['CLASS'] == 'Non-ref']
# Export to CSV
#reference_gauges.to_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/Dataset1_BasinID/Dataset1_BasinID/reference.csv", index=False)
#non_reference_gauges.to_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/Dataset1_BasinID/Dataset1_BasinID/non_reference.csv", index=False)


#Cross-ref with the intercomparison project gauges 
Mai_data=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations/compiled_gauges_no_duplicates.csv")
#Mai_data=Mai_data.iloc[148:212,:]
Mai_data=Mai_data[Mai_data['Reference']=='Ref']
Mai_data.rename(columns={'ID': 'STAID'}, inplace=True)

# Merge with indicator to track which rows are in df and mai
merged = reference_gauges.merge(Mai_data, on='STAID', how='left', indicator=True)

# Filter to keep only rows that are not in mai
merged = merged[merged['_merge'] == 'left_only']

# Drop the merge indicator and any extra columns from mai
merged_data_final = merged[reference_gauges.columns]



merged_data_final.to_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/us_expanded_ref_gauges.csv", index=False)




#Cross-ref with the intercomparison project gauges 
Mai_data=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations/compiled_gauges_no_duplicates.csv")
#Mai_data=Mai_data.iloc[148:212,:]
Mai_data=Mai_data[Mai_data['Reference']=='Non-ref']
Mai_data.rename(columns={'ID': 'STAID'}, inplace=True)

# Merge with indicator to track which rows are in df and mai
merged = non_reference_gauges.merge(Mai_data, on='STAID', how='left', indicator=True)

# Filter to keep only rows that are not in mai
merged = merged[merged['_merge'] == 'left_only']

# Drop the merge indicator and any extra columns from mai
merged_data_final = merged[reference_gauges.columns]



merged_data_final.to_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/us_expanded_nonref_gauges.csv", index=False)




#######################################################################################################

import pandas as pd 

#Cananda all station data 


ontario_data=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/Canada/station_metadata_real_time.csv",dtype={'site_no': str})


#Canada RHDB network 

natural_gauges=pd.read_csv("C:/Users/rg727/Desktop/RHBN_Metadata.csv")

natural_gauges.rename(columns={'STATION_NUMBER': 'Station ID'}, inplace=True)


#Merge datasets with the ontario dataset 

merged_data = pd.merge(ontario_data, natural_gauges, on='Station ID', how='inner')

merged_data.rename(columns={'Station ID': 'STAID'},inplace=True)

#Merge with the Mai dataset 

#Cross-ref with the intercomparison project gauges 
Mai_data=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations/compiled_gauges_no_duplicates.csv")
#Mai_data=Mai_data[Mai_data['Regulation']=='Regulated']
Mai_data.rename(columns={'ID': 'STAID'}, inplace=True)


# Merge with indicator to track which rows are in df and mai
merged = merged_data.merge(Mai_data, on='STAID', how='left', indicator=True)

# Filter to keep only rows that are not in mai
merged = merged[merged['_merge'] == 'left_only']

# Drop the merge indicator and any extra columns from mai
merged_data_final = merged[merged_data.columns]

merged_data_final.to_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/canada_expanded_non_ref_gauges.csv", index=False)



###Figure out which Canada gauges are in the great lakes basin 

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Read your list of gauges
gauges = merged_data_final  # should have columns like 'gauge_id', 'latitude', 'longitude'

# Create a GeoDataFrame of gauges
geometry = [Point(xy) for xy in zip(gauges.Longitude, gauges.Latitude)]
gauges_gdf = gpd.GeoDataFrame(gauges, geometry=geometry, crs="EPSG:4326")  # assuming lat/lon WGS84

# Read Great Lakes Basin shapefile
basin = gpd.read_file("C:/Users/rg727/Desktop/Earths_Future_Revision/taxes/greatlakes_subbasins/greatlakes_subbasins.shp")

# Make sure CRS matches
basin = basin.to_crs(gauges_gdf.crs)


# Option 1: Spatial join
gauges_in_basin = gpd.sjoin(gauges_gdf, basin, how="inner", predicate="within")

# Option 2: Manual mask
gauges_in_basin = gauges_gdf[gauges_gdf.within(basin.unary_union)]


gauges_in_basin.to_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations_Expanded/canada_expanded_gl_gauges.csv", index=False)

