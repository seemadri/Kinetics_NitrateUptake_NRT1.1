#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})

# %%
df = pd.read_csv("J_vs_Se.tsv", sep="\t")

Se_values = df["Se"]
J_values1 = df["J1"]
J_values2 = df["J2"]
J_values3 = df["J3"]

plt.figure()
sns.lineplot(x = Se_values, y = J_values1, linewidth = 3, color = "black")
sns.lineplot(x = Se_values, y = J_values2, linewidth = 3, color = "red")
sns.lineplot(x = Se_values, y = J_values3, linewidth = 3, color = "orange")
plt.savefig('3B.png', dpi=800)
plt.close()
sns.lineplot(x = Se_values, y = J_values1, linewidth = 3, color = "black")
sns.lineplot(x = Se_values, y = J_values2, linewidth = 3, color = "red")
sns.lineplot(x = Se_values, y = J_values3, linewidth = 3, color = "orange")
plt.xlim([0, 2.5])
plt.ylim([0, 350])
plt.savefig('3B_inset.png', dpi=800)
plt.close()
# %%
df = pd.read_csv("rate_affinity_tradeoff.tsv", sep="\t")

# keep only rows that have an affinity2 value greater than 0.9910
df = df[df["affinity2"] > 0.997]

range_of_fold_change_in_r_due_to_P = df["fold_change_in_r_due_to_P"]
rate1 = df["rate1"]
rate2 = df["rate2"]
affinity1 = df["affinity1"]
affinity2 = df["affinity2"]

sns.lineplot(x = range_of_fold_change_in_r_due_to_P, y = np.array(rate2)/np.array(rate1), linewidth = 3, color = "black")
plt.savefig('3C_i.png', dpi=800)
plt.close()

sns.lineplot(x = range_of_fold_change_in_r_due_to_P, y = np.array(affinity2)/np.array(affinity1), linewidth = 3, color = "black")
plt.savefig('3C_ii.png', dpi=800)


# %%
df = pd.read_csv('rate_vs_R.tsv', sep='\t')

R_values = df["R"]
rate_values1 = df["rate1"]
rate_values2 = df["rate2"]
rate_values3 = df["rate2/rate1"]

sns.lineplot(x = R_values, y = rate_values3, linewidth = 3, color = "black")
plt.xscale('log')
plt.savefig('3D_i.png', dpi=800)
plt.close()

sns.lineplot(x = R_values, y = rate_values1, linewidth = 3, color = "black")
sns.lineplot(x = R_values, y = rate_values2, linewidth = 3, color = "red")
plt.xscale('log')
plt.savefig('3D_ii.png', dpi=800)

# %%
