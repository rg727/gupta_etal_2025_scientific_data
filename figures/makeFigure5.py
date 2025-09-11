# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:33:54 2025

@author: rg727
"""

#########Now Plot#########################################


#Validation 

LSTM_NN_all=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure3/Data/1995_2013/oos_nse.csv")
avg_LSTM_nn=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/NSE/NSE_train_nn_1995_2013.csv")

LSTM_clim_all=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Figure3/Data/1995_2013/oos_nse_clim.csv")
avg_LSTM_clim=pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/Box_Plots_Training/Metrics/Training/NSE/NSE_train_clim_1995_2013.csv")




fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # Create a 2x2 grid
axs = axs.flatten()

# --- Panel (a) ---
temp = np.sort(LSTM_clim_all.loc[(LSTM_clim_all['Type'] == 'Natural'),'test'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[2].plot(temp, ranks, marker='o', color='#bc6c25', label='LSTM_clim')

temp =np.sort(LSTM_NN_all.loc[(LSTM_NN_all['Type'] == 'Natural'),'test'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[2].plot(temp, ranks, marker='o', color='#283618', label='LSTM_donor')

axs[2].set_title("Out of Sample Validation (1995-2013) - Natural Sites",fontsize=24)
axs[2].set_xlabel("NSE",fontsize=18)
axs[2].set_ylabel("Rank",fontsize=18)
#axs[2].legend()
axs[2].set_xlim(0, 1)

# --- Panel (b) ---
temp =  np.sort(LSTM_clim_all.loc[(LSTM_clim_all['Type'] == 'Regulated'),'test'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[3].plot(temp, ranks, marker='o', color='#bc6c25', label='LSTM_clim')

temp = np.sort(LSTM_NN_all.loc[(LSTM_NN_all['Type'] == 'Regulated'),'test'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[3].plot(temp, ranks, marker='o', color='#283618', label='LSTM_donor')

axs[3].set_title("Out of Sample Validation (1995-2013) - Regulated Sites",fontsize=24)
axs[3].set_xlabel("NSE",fontsize=18)
axs[3].set_ylabel("Rank",fontsize=18)
axs[3].legend()
axs[3].set_xlim(0, 1)

# --- Panel (c) ---
temp = np.sort(avg_LSTM_clim.loc[(avg_LSTM_clim['Type'] == 'Natural'),'NSE'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[0].plot(temp, ranks, marker='o', color='#bc6c25', label='LSTM_clim')

temp =  np.sort(avg_LSTM_nn.loc[(avg_LSTM_nn['Type'] == 'Natural'),'NSE'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[0].plot(temp, ranks, marker='o', color='#283618', label='LSTM_donor')

axs[0].set_title("Out of Time Validation (2005-2013) - Natural Sites",fontsize=24)
axs[0].set_xlabel("NSE",fontsize=18)
axs[0].set_ylabel("Rank",fontsize=18)
axs[0].legend()
axs[0].set_xlim(0, 1)

# --- Panel (d) ---
temp = np.sort(avg_LSTM_clim.loc[(avg_LSTM_clim['Type'] == 'Regulated'),'NSE'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[1].plot(temp, ranks, marker='o', color='#bc6c25', label='LSTM_clim')

temp =np.sort(avg_LSTM_nn.loc[(avg_LSTM_nn['Type'] == 'Regulated'),'NSE'])
ranks = np.arange(0, len(temp))
ranks=ranks/len(ranks)
axs[1].plot(temp, ranks, marker='o', color='#283618', label='LSTM_donor')

axs[1].set_title("Out of Time Validation (2005-2013) - Regulated Sites",fontsize=24)
axs[1].set_xlabel("NSE",fontsize=18)
axs[1].set_ylabel("Rank",fontsize=18)
axs[1].legend()
axs[1].set_xlim(0, 1)

axs[0].tick_params(axis='both', which='major', labelsize=24)
axs[1].tick_params(axis='both', which='major', labelsize=24)
axs[2].tick_params(axis='both', which='major', labelsize=24)
axs[3].tick_params(axis='both', which='major', labelsize=24)

# Final layout
plt.tight_layout()

plt.savefig("C:/Users/rg727/Documents/Great Lake Project/Paper/Figures/modern_nse_gauge_type.pdf")


