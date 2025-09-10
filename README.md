[![DOI](https://zenodo.org/badge/265254045.svg)](https://zenodo.org/doi/10.5281/zenodo.10442485)
# Gupta-etal_2025_nature_scientific_data

Rohini S. Gupta <sup>1*</sup>, Sungwook Wi <sup>1*</sup>, Scott Steinschneider<sup>1</sup>  
<sup>1</sup> Department of Biological and Environmental Engineering, Cornell University, Ithaca, NY  
*Corresponding author:* rg727@cornell.edu

---

## Abstract

High-quality regional streamflow datasets are necessary to support local water resources planning and management. However, observed streamflow records are often limited by the availability of surface water gauges, which frequently go in and out of service over time. Long-term reconstruction products that fill in missing data are not consistently available. This challenge is prevalent across the United States, including in the Great Lakes region, which contains 20% of the world’s freshwater and serves as a critical resource for both the United States and Canada. In the Great Lakes, there is also a specific interest in estimating aggregate runoff into the lakes to better understand the regional water balance and lake level variability. Existing aggregate runoff data products are typically derived from runoff area ratios or process-based models, but these approaches are hindered by model parameter uncertainty and a limited ability to capture the vast spatial heterogeneity of the basin. Furthermore, of the products that exist, most do not start until 1980. In this work, we develop a new, historical reconstruction of daily streamflow at over 650 gauged locations throughout the Great Lakes basin using a novel regional Long Short-Term Memory (LSTM) model that integrates local climate data, physical catchment characteristics, and runoff observations from nearby gauged sites. We also estimate monthly runoff into the lakes for the period of 1951–2013. The daily reconstruction product will equip water managers with information to understand emerging hydroclimate trends and provide a basis to support local water resources analyses. The aggregate runoff product shows strong potential for improving estimates of monthly lake-wide runoff, which can ultimately help resolve the complete water balance of the Great Lakes and provide critical context for lake level shifts under a changing climate.

---

## Journal reference

Gupta, R. S., Wi, S., Steinschneider, S. (In Review). *Machine Learning-Based Reconstructions of Historical Daily and Monthly Runoff for the Laurentian Great Lakes Region*. Nature: Scientific Data

---

## Code reference

All code to reproduce the dataset and figures is found in this repository or is linked below.

---

## Data reference

### Output Data

Gupta, R.S., Wi, S., Steinschneider, S. (2025). *Laurentian Great Lakes Daily and Monthly Runoff (1951-2013) (Version v1)* [Data set]. Zenodo.  
🔗 [https://doi.org/10.5281/zenodo.16897690](https://doi.org/10.5281/zenodo.16897690)

---

## Reproduce My Experiment

### 1. Produce Training Data

The training data for the LSTM models is a combination of US and Canada streamflow runoff, meteorology, and physical catchment characteristics. 

| Data Source                | Description                                                                                          
|----------------------------|------------------------------------------------------------------------------------------------------|
| [USGS](https://www.usgs.gov)| Runoff for the US Gauges                                                                             |
| [Water Survey Canada](https://wateroffice.ec.gc.ca/mainmenu/real_time_data_index_e.html)        | Runoff for the Canada Gauges                                                                         | 
| [Livneh](https://climatedataguide.ucar.edu/climate-data/livneh-gridded-precipitation-and-other-meteorological-variables-continental-us-mexico)                     | Meterology                                                                                           | 
| [HYDROSHEDS](https://www.frdr-dfdr.ca/repo/dataset/6632cd3c-9b3b-4cc6-a87a-204c92d30485)                 | Elevation and Slope Data                                                                             | 
| [NALCMS](https://www.frdr-dfdr.ca/repo/dataset/6632cd3c-9b3b-4cc6-a87a-204c92d30485)                     | Land Use Data                                                                                      | 
| [GSDE)](https://cmr.earthdata.nasa.gov/search/concepts/C1214604044-SCIOPS.html))                         | Soil Composition                                                                                      | 

The gauges that are active for each modeling subperiod are in `data/gauges/`

| Script Name                 | Description                                                                                          
|----------------------------|------------------------------------------------------------------------------------------------------|
| `make_subfolder.sh`        | Example script of how to download USGS/WSC runoff data   | 
| `splitting_script.R`       | Example script of how to create the physical characteristic datasets                                 | 


---

### 2. Run Historical CALFEWS

Steps:

1. Clone the CALFEWS repo:  
   ```bash
   git clone https://github.com/hbz5000/CALFEWS
