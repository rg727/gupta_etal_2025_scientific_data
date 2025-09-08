# Gupta-etal_2025_nature_scientific_data

Rohini S. Gupta <sup>1*</sup>, Sungwook Wi <sup>1*</sup>, Scott Steinschneider<sup>1</sup>  
<sup>1</sup> Department of Biological and Environmental Engineering, Cornell University, Ithaca, NY  
*Corresponding author:* rg727@cornell.edu

---

## Abstract

High-quality regional streamflow datasets are necessary to support local water resources planning and management. However, observed streamflow records are often limited by the availability of surface water gauges, which frequently go in and out of service over time. Long-term reconstruction products that fill in missing data are not consistently available. This challenge is prevalent across the United States, including in the Great Lakes region, which contains 20% of the world’s freshwater and serves as a critical resource for both the United States and Canada. In the Great Lakes, there is also a specific interest in estimating aggregate runoff into the lakes to better understand the regional water balance and lake level variability.

Existing aggregate runoff data products are typically derived from runoff area ratios or process-based models, but these approaches are hindered by model parameter uncertainty and a limited ability to capture the vast spatial heterogeneity of the basin. Furthermore, of the products that exist, most do not start until 1980.

In this work, we develop a new, historical reconstruction of daily streamflow at over 650 gauged locations throughout the Great Lakes basin using a novel regional Long Short-Term Memory (LSTM) model that integrates local climate data, physical catchment characteristics, and runoff observations from nearby gauged sites. We also estimate monthly runoff into the lakes for the period of 1950–2013. The daily reconstruction product will equip water managers with information to understand emerging hydroclimate trends and provide a basis to support local water resources analyses. The aggregate runoff product shows strong potential for improving estimates of monthly lake-wide runoff, which can ultimately help resolve the complete water balance of the Great Lakes and provide critical context for lake level shifts under a changing climate.

---

## Journal reference

Gupta, R. S., Wi, S., Steinschneider, S. (In Preparation). *Machine Learning-Based Reconstructions of Historical Daily and Monthly Runoff for the Laurentian Great Lakes Region*. Nature Scientific Data

---

## Code reference

All code to reproduce the results/figures is found in this repository or is linked below.

---

## Data reference

### Input and Output Data

Gupta, R.S., Steinschneider, S., Reed, P.M. (2024). *Gupta-et-al_2024_EarthsFuture (Version v1)* [Data set]. MSD-LIVE Data Repository.  
🔗 [https://doi.org/10.57931/2458138](https://doi.org/10.57931/2458138)

---

## Contributing Modeling Software

| Model                 | Version | Repository Link                         | DOI                      |
|----------------------|---------|------------------------------------------|--------------------------|
| CALFEWS - Historical | 1.0     | https://github.com/hbz5000/CALFEWS      | 10.5281/zenodo.4091708   |

---

## Reproduce My Experiment

There are two components to this study:

1. Produce input files for CALFEWS
2. Run CALFEWS under historical, paleo, and climate change scenarios, and synthesize the output

---

### 1. Produce Input Files

Full natural flow and snow that serve as input into CALFEWS can be found in the `hydroclimate` folder of the MSD-Live repository. The streamflow is sorted into CDEC, Paleo, and 4T, 1CC (the selected climate change scenario). Use the scripts in the `create_inputs` directory.

| Script Name                 | Description                                                                                          | How to Run                         |
|----------------------------|------------------------------------------------------------------------------------------------------|------------------------------------|
| `make_subfolder.sh`        | Creates folder structure for baseline (0T, 0CC) and climate change scenarios with ensemble members.  | `sbatch make_subfolder.sh`         |
| `splitting_script.R`       | Splits data into 30-year periods (baseline).                                                         | `sbatch split_datasets.sh`         |
| `splitting_script_CC.R`    | Splits data into 30-year periods (climate change).                                                   | `sbatch split_datasets.sh`         |
| `initialize_params.sh`     | Generates `base_inflows.json` needed by CALFEWS.                                                     | `python3 step_two.py -o /path/to/my/outputdir` |

---

### 2. Run Historical CALFEWS

Steps:

1. Clone the CALFEWS repo:  
   ```bash
   git clone https://github.com/hbz5000/CALFEWS
