# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 20:28:24 2025

@author: rg727
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
MHU_truth_first_half = pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/MHU_Runoff_1950.csv')
MHU_truth_second_half = pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/MHU_Runoff.csv')
MHU_truth = pd.concat([MHU_truth_first_half, MHU_truth_second_half]).reset_index()
MHU_truth = MHU_truth.iloc[0:756, :]
MHU_truth.index = df.index  # assuming df is already loaded and has the same index

# Define 4 model periods
periods = [
    ("1951-01-01", "1965-12-31"),
    ("1966-01-01", "1980-12-31"),
    ("1981-01-01", "1995-12-31"),
    ("1996-01-01", "2013-12-31")
]

# Create the figure and subplots
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=False)

# Plot each period
for ax, (start, end) in zip(axes, periods):
    period_data = df.loc[start:end]

    # Emphasize model prediction
    ax.plot(period_data.index,
            period_data["MHU"] * MHU_GLERL / MHU_LAKE,
            label="Model Prediction",
            color='#606c38',
            linewidth=2,
            zorder=3)

    # De-emphasize truth datasets
    ax.plot(MHU_truth.loc[start:end].index,
            MHU_truth.loc[start:end]["USACE.GLSHFS"],
            label='GLSHFS',
            color="#bc6c25",
            linewidth=2,
            alpha=0.8)

    ax.plot(MHU_truth.loc[start:end].index,
            MHU_truth.loc[start:end]["USACE.AHPS"],
            label='AHPS',
            color="#dda15e",
            linewidth=1.2,
            alpha=0.8)

    ax.plot(MHU_truth.loc[start:end].index,
            MHU_truth.loc[start:end]["NOAA.GLERL.GLM.HMD"],
            label='GLERL',
            color="#a53860",
            linewidth=1.2,
            alpha=0.8)

    # X-axis formatting
    ax.set_xlim(pd.to_datetime(start), pd.to_datetime(end))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

# Label only the bottom subplot
axes[-1].set_xlabel("Date")

# Add a single shared legend at the top
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)

# Adjust layout to make room for legend
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()  # Uncomment to display


plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/GLERL_Figures/MHU.pdf")

##########################################################################################################
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 20:45:26 2025

@author: rg727
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 20:28:24 2025

@author: rg727
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
SUP_truth_first_half = pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/Superior_Runoff_1950.csv')
SUP_truth_second_half = pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/Superior_Runoff.csv')
SUP_truth = pd.concat([SUP_truth_first_half, SUP_truth_second_half]).reset_index()
SUP_truth = SUP_truth.iloc[0:756, :]
SUP_truth.index = df.index  # assuming df is already loaded and has the same index

# Define 4 model periods
periods = [
    ("1951-01-01", "1965-12-31"),
    ("1966-01-01", "1980-12-31"),
    ("1981-01-01", "1995-12-31"),
    ("1996-01-01", "2013-12-31")
]

# Create the figure and subplots
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=False)

# Plot each period
for ax, (start, end) in zip(axes, periods):
    period_data = df.loc[start:end]

    # Emphasize model prediction
    ax.plot(period_data.index,
            period_data["SUP"] * SUP_GLERL / SUP_LAKE,
            label="Model Prediction",
            color='#606c38',
            linewidth=2,
            zorder=3)

    # De-emphasize truth datasets
    ax.plot(SUP_truth.loc[start:end].index,
            SUP_truth.loc[start:end]["USACE.GLSHFS"],
            label='GLSHFS',
            color="#bc6c25",
            linewidth=2,
            alpha=0.8)

    ax.plot(SUP_truth.loc[start:end].index,
            SUP_truth.loc[start:end]["USACE.AHPS"],
            label='AHPS',
            color="#dda15e",
            linewidth=1.2,
            alpha=0.8)

    ax.plot(SUP_truth.loc[start:end].index,
            SUP_truth.loc[start:end]["NOAA.GLERL.GLM.HMD"],
            label='GLERL',
            color="#a53860",
            linewidth=1.2,
            alpha=0.8)

    # X-axis formatting
    ax.set_xlim(pd.to_datetime(start), pd.to_datetime(end))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

# Label only the bottom subplot
axes[-1].set_xlabel("Date")

# Add a single shared legend at the top
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)

# Adjust layout to make room for legend
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()  # Uncomment to display


plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/GLERL_Figures/SUP.pdf")
#############################################################################################################
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 20:45:26 2025

@author: rg727
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 20:28:24 2025

@author: rg727
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
ER_truth_first_half = pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/Erie_Runoff_1950.csv')
ER_truth_second_half = pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/Erie_Runoff.csv')
ER_truth = pd.concat([ER_truth_first_half, ER_truth_second_half]).reset_index()
ER_truth = ER_truth.iloc[0:756, :]
ER_truth.index = df.index  # assuming df is already loaded and has the same index

# Define 4 model periods
periods = [
    ("1951-01-01", "1965-12-31"),
    ("1966-01-01", "1980-12-31"),
    ("1981-01-01", "1995-12-31"),
    ("1996-01-01", "2013-12-31")
]

# Create the figure and subplots
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=False)

# Plot each period
for ax, (start, end) in zip(axes, periods):
    period_data = df.loc[start:end]

    # Emphasize model prediction
    ax.plot(period_data.index,
            period_data["ER"] * ER_GLERL / ER_LAKE,
            label="Model Prediction",
            color='#606c38',
            linewidth=2,
            zorder=3)

    # De-emphasize truth datasets
    ax.plot(ER_truth.loc[start:end].index,
            ER_truth.loc[start:end]["USACE.GLSHFS"],
            label='GLSHFS',
            color="#bc6c25",
            linewidth=2,
            alpha=0.8)

    ax.plot(ER_truth.loc[start:end].index,
            ER_truth.loc[start:end]["USACE.AHPS"],
            label='AHPS',
            color="#dda15e",
            linewidth=1.2,
            alpha=0.8)

    ax.plot(ER_truth.loc[start:end].index,
            ER_truth.loc[start:end]["NOAA.GLERL.GLM.HMD"],
            label='GLERL',
            color="#a53860",
            linewidth=1.2,
            alpha=0.8)

    # X-axis formatting
    ax.set_xlim(pd.to_datetime(start), pd.to_datetime(end))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

# Label only the bottom subplot
axes[-1].set_xlabel("Date")

# Add a single shared legend at the top
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)

# Adjust layout to make room for legend
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()  # Uncomment to display


plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/GLERL_Figures/ER.pdf")
''# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 20:45:26 2025

@author: rg727
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 20:28:24 2025

@author: rg727
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
STC_truth_first_half = pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/STC_Runoff_1950.csv')
STC_truth_second_half = pd.read_csv('C:/Users/rg727/Documents/Great Lake Project/Full_LSTM_Datasets/STC_Runoff.csv')
STC_truth = pd.concat([STC_truth_first_half, STC_truth_second_half]).reset_index()
STC_truth = STC_truth.iloc[0:756, :]
STC_truth.index = df.index  # assuming df is already loaded and has the same index

# Define 4 model pSTCiods
pSTCiods = [
    ("1951-01-01", "1965-12-31"),
    ("1966-01-01", "1980-12-31"),
    ("1981-01-01", "1995-12-31"),
    ("1996-01-01", "2013-12-31")
]

# Create the figure and subplots
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=False)

# Plot each pSTCiod
for ax, (start, end) in zip(axes, pSTCiods):
    period_data = df.loc[start:end]

    # Emphasize model prediction
    ax.plot(period_data.index,
            period_data["STC"] * STC_GLERL / STC_LAKE,
            label="Model Prediction",
            color='#606c38',
            linewidth=2,
            zorder=3)

    # De-emphasize truth datasets
    ax.plot(STC_truth.loc[start:end].index,
            STC_truth.loc[start:end]["USACE.GLSHFS"],
            label='GLSHFS',
            color="#bc6c25",
            linewidth=2,
            alpha=0.8)

    ax.plot(STC_truth.loc[start:end].index,
            STC_truth.loc[start:end]["USACE.AHPS"],
            label='AHPS',
            color="#dda15e",
            linewidth=1.2,
            alpha=0.8)

    ax.plot(STC_truth.loc[start:end].index,
            STC_truth.loc[start:end]["NOAA.GLERL.GLM.HMD"],
            label='GLERL',
            color="#a53860",
            linewidth=1.2,
            alpha=0.8)

    # X-axis formatting
    ax.set_xlim(pd.to_datetime(start), pd.to_datetime(end))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

# Label only the bottom subplot
axes[-1].set_xlabel("Date")

# Add a single shared legend at the top
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='uppSTC centSTC', ncol=4, frameon=False)

# Adjust layout to make room for legend
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()  # Uncomment to display


plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Runoff/Decadal_Training/GLERL_Figures/STC.pdf")
