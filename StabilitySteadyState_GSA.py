#%%
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from SALib.sample import saltelli
from SALib.analyze import sobol

from matplotlib import rcParams
rcParams.update({'font.size': 16, 'font.family': 'Arial'})
#%%
def model(y, t, r, fi, be, fe, bi, Si, Se):
    Pi, Pe, Ci, Ce = y
    dPidt = r*Pe - r*Pi + fi*Si*Ci - bi*Pi
    dPedt = fe*Se*Ce - be*Pe + r*Pi - r*Pe
    dCidt = bi*Pi - fi*Si*Ci + r*Ce - r*Ci
    dCedt = r*Ci + be*Pe - r*Ce - fe*Se*Ce
    return [dPidt, dPedt, dCidt, dCedt]

# solve the differential equations for steady state
def solve(r, fi, be, fe, bi, Si, Se, y0):
    time= 200000
    steps = 100000
    t = np.linspace(0, time, steps)
    solution = odeint(model, y0, t, args=(r, fi, be, fe, bi, Si, Se))
    return t, solution

def solve_for_different_Se(Se, r, fi, ratio, internal_ratio):
    # Set the parameters
    r = r
    fi = fi
    ratio = ratio
    fe = ratio*fi
    bi = fi
    be = fe
    Se = Se
    Si = internal_ratio*Se

    # Set the initial conditions
    y0 = [0, 1, 0, 0]

    # Solve the differential equations
    t, solution = solve(r, fi, be, fe, bi, Si, Se, y0)

    J = bi*solution[:, 0] - fi*Si*solution[:, 2]

    return J[-1]

# function to fit michaelis menten equation
def michaelis_menten(x, Vmax, Km):
    x = np.array(x)
    return Vmax*x/(Km + x)

# function to estimate Vmax and Km
def estimate_parameters(x, y):
    popt, pcov = curve_fit(michaelis_menten, x, y, bounds=(0, np.inf))
    return popt, pcov

def hills(X, theshold, n):
    return X**n/(theshold**n + X**n)

def plot_for_JasSe(R = 1e2, fold_change_in_r_for_unphos = 1e2, fi = 1e3, ratio = 1e3, internal_ratio = 1e-5):
    # read steady state values
    df = pd.read_csv('steady_states.tsv', sep='\t')
    A = df['A']
    B = df['B']
    P = df['P']
    total = []
    phos_total = []
    unphos_total = []
    for i in range(len(A)):
        Se = A[i]
        J_phos = solve_for_different_Se(Se, R, fi, ratio, internal_ratio)
        J_unphos = solve_for_different_Se(Se, R*fold_change_in_r_for_unphos, fi, ratio, internal_ratio)
        total.append((J_phos*P[i] + J_unphos*B[i])/(P[i] + B[i]))
        phos_total.append(J_unphos*P[i]/(P[i] + B[i]))
        unphos_total.append(J_unphos*B[i]/(P[i] + B[i]))
    plt.scatter(A, total, label='Total')
    plt.scatter(A, phos_total, label='Phosphorylated')
    plt.scatter(A, unphos_total, label='Unphosphorylated')
    plt.legend()

plot_for_JasSe()
#%%
def hill_function(A, At_b, n_ab, f_ab):
    return ((At_b ** n_ab) + f_ab * (A ** n_ab))/((At_b ** n_ab) + (A ** n_ab))

def modelABCDEP(y, t, A, params):
    params = Gb, kb, At_b, n_ab, f_ab, Bt_b, n_bb, f_bb, Gc, Bt_c, n_bc, f_bc, kc, k_CD, k_DE, k_minus_CD, k_minus_DE, k_P, k_minus_P = params
    B, C, D, E, CD, DE, P = y
    dB_dt = Gb*hill_function(A, At_b, n_ab, f_ab)*hill_function(B, Bt_b, n_bb, f_bb) - kb*B - k_P*B*DE + k_minus_P*P
    dC_dt = -k_CD*C*D + k_minus_CD*CD + Gc*hill_function(B, Bt_c, n_bc, f_bc) - kc*C
    dD_dt = -k_CD*C*D + k_minus_CD*CD - k_DE*D*E + k_minus_DE*DE
    dE_dt = -k_DE*D*E + k_minus_DE*DE
    dCD_dt = k_CD*C*D - k_minus_CD*CD
    dDE_dt = k_DE*D*E - k_minus_DE*DE
    dP_dt = k_P*B*DE - k_minus_P*P
    
    return [dB_dt, dC_dt, dD_dt, dE_dt, dCD_dt, dDE_dt, dP_dt]

def return_initial_conditions():
    # Initial concentrations
    B0 = 1000 # initial concentration of NRT1.1
    C0 = 1.0 # initial concentration of CIPK8
    D0 = 2000 # initial concentration of CBL
    E0 = 1000 # initial concentration of CIPK23
    CD0 = 0.0 # initial concentration of CIPK8-CBL complex
    DE0 = 0.0 # initial concentration of CBL-CIPK23 complex
    P0 = 0.0 # initial concentration of Phosphorylated NRT1.1
    return B0, C0, D0, E0, CD0, DE0, P0

#define parameters
Gb = 50  #Growth rate of B
kb = 0.1  #Decay rate of B
At_b = 1.5  #threshold concentration of A binding to B
n_ab = 10  #Hill coefficient for A binding to B
f_ab = 5  #Foldchange of A binding to B
Bt_b= 3000 #threshold concentration of B binding to B
n_bb = 4 #Hill coefficient for B binding to B
f_bb = 5 #Foldchange of B binding to B
Gc = 50 #Growth rate of C
Bt_c = 3500 #threshold concentration of B binding to C
n_bc = 2 #Hill coefficient for B binding to C
f_bc = 30 #Foldchange of B binding to C
kc = 0.1 #Decay rate of C
k_CD = 0.21 #Rate constant for C-D complex formation
k_DE = 0.1 #Rate constant for D-E complex formation
k_minus_CD = 0.01 #Rate constant for C-D complex dissociation
k_minus_DE = 0.01 #Rate constant for D-E complex dissociation
k_P = 0.0001 #1e-4 #Rate constant for phosphorylation of B by DE
k_minus_P = 0.02 #Rate constant for dephosphorylation of B by P

params = [Gb, kb, At_b, n_ab, f_ab, Bt_b, n_bb, f_bb, Gc, Bt_c, n_bc, f_bc, kc, k_CD, k_DE, k_minus_CD, k_minus_DE, k_P, k_minus_P]

# solve for steady states
def return_steady_states(A, params):
    B0, C0, D0, E0, CD0, DE0, P0 = return_initial_conditions()
    y0 = [B0, C0, D0, E0, CD0, DE0, P0]
    t = np.linspace(0, 1000, 1000)
    sol = odeint(modelABCDEP, y0, t, args=(A,params))
    B = sol[:, 0]
    C = sol[:, 1]
    D = sol[:, 2]
    E = sol[:, 3]
    CD = sol[:, 4]
    DE = sol[:, 5]
    P = sol[:, 6]
    return B[-1], C[-1], D[-1], E[-1], CD[-1], DE[-1], P[-1]

def plot_classical(B_array, C_array, D_array, E_array, CD_array, DE_array, P_array, A_array):

    plt.scatter(A_array, B_array, label='B')
    plt.scatter(A_array, C_array, label='C')
    plt.scatter(A_array, P_array, label='P')
    plt.xlabel('A')
    plt.ylabel('B')
    #plt.yscale('log')
    plt.legend()
    plt.show()

    plt.scatter(A_array, B_array/(P_array+B_array), label='P')
    plt.xlabel('A')
    plt.ylabel('P/(B+P)')
    plt.legend()
    plt.show()
    plt.close()

    # plot steady states
    sns.set_palette("Set1")
    plt.figure()
    #plt.scatter(A_array, B_array, label='B')
    #plt.scatter(A_array, C_array, label='C')
    plt.scatter(A_array, D_array, label='D')
    plt.scatter(A_array, E_array, label='E')
    plt.scatter(A_array, CD_array, label='CD')
    plt.scatter(A_array, DE_array, label='DE')
    #plt.scatter(A_array, P_array, label='P')
    plt.xlabel('A')
    plt.ylabel('Concentration')
    plt.legend()
    plt.show()
    plt.close()

    plt.scatter(A_array, P_array/(B_array+P_array), label='P')

def return_differences(params):
    B_array = []
    C_array = []
    D_array = []
    E_array = []
    CD_array = []
    DE_array = []
    P_array = []
    A_array = np.linspace(0, 3, 100)

    for A in A_array:
        B, C, D, E, CD, DE, P = return_steady_states(A, params)
        B_array.append(B)
        C_array.append(C)
        D_array.append(D)
        E_array.append(E)
        CD_array.append(CD)
        DE_array.append(DE)
        P_array.append(P)

    B_array = np.array(B_array)
    P_array = np.array(P_array)
    frac = P_array/(B_array+P_array)

    #plot_classical(B_array, C_array, D_array, E_array, CD_array, DE_array, P_array, A_array)

    # save the values for all the variables
    #df = pd.DataFrame({'A': A_array, 'B': B_array, 'C': C_array, 'D': D_array, 'E': E_array, 'CD': CD_array, 'DE': DE_array, 'P': P_array})
    #df.to_csv('steady_states.tsv', sep='\t', index=False)
    #print(frac[0])
    #print(frac[-1]) 
    return frac[0] - frac[-1]
    
return_differences(params)
#%%
# Global Sensitivity Analysis
"""
# Define the parameters and their bounds for sensitivity analysis (+/- 10% and +/- 20% of the nominal values)
parameters = {
    'Gb': {'bounds': [45, 55]},
    'kb': {'bounds': [0.09, 0.11]},
    'At_b': {'bounds': [1.35, 1.65]},
    'n_ab': {'bounds': [9, 11]},
    'f_ab': {'bounds': [4.5, 5.5]},
    'Bt_b': {'bounds': [2850, 3150]},
    'n_bb': {'bounds': [3.6, 4.4]},
    'f_bb': {'bounds': [4.5, 5.5]},
    'Gc': {'bounds': [45, 55]},
    'Bt_c': {'bounds': [3150, 3850]},
    'n_bc': {'bounds': [1.8, 2.2]},
    'f_bc': {'bounds': [27, 33]},
    'kc': {'bounds': [0.09, 0.11]},
    'k_CD': {'bounds': [0.189, 0.231]},
    'k_DE': {'bounds': [0.09, 0.11]},
    'k_minus_CD': {'bounds': [0.009, 0.011]},
    'k_minus_DE': {'bounds': [0.009, 0.011]},
    'k_P': {'bounds': [9e-05, 1e-04]},
    'k_minus_P': {'bounds': [0.018, 0.022]}
}
"""
parameters = {
    'Gb': {'bounds': [40, 60]},
    'kb': {'bounds': [0.08, 0.12]},
    'At_b': {'bounds': [1.2, 1.8]},
    'n_ab': {'bounds': [8, 12]},
    'f_ab': {'bounds': [4, 6]},
    'Bt_b': {'bounds': [2400, 3600]},
    'n_bb': {'bounds': [3, 5]},
    'f_bb': {'bounds': [4, 6]},
    'Gc': {'bounds': [40, 60]},
    'Bt_c': {'bounds': [2400, 4800]},
    'n_bc': {'bounds': [1.5, 2.5]},
    'f_bc': {'bounds': [20, 40]},
    'kc': {'bounds': [0.08, 0.12]},
    'k_CD': {'bounds': [0.168, 0.252]},
    'k_DE': {'bounds': [0.08, 0.12]},
    'k_minus_CD': {'bounds': [0.008, 0.012]},
    'k_minus_DE': {'bounds': [0.008, 0.012]},
    'k_P': {'bounds': [7.2e-05, 1.2e-04]},
    'k_minus_P': {'bounds': [0.016, 0.024]}
}


# Define the problem for sensitivity analysis
problem = {
    'num_vars': len(parameters),
    'names': list(parameters.keys()),
    'bounds': [param['bounds'] for param in parameters.values()]
}
# Generate samples using Saltelli's method
#Note: The number of samples can be adjusted based on the desired accuracy and computational resources.
#Note: The `calc_second_order` parameter is set to True to calculate second-order Sobol indices.
#N is  base sample size per parameter,Total samples = N * (2D + 2) if calc_second_order=True
N = 1000  # base sample size
param_values = saltelli.sample(problem, N, calc_second_order=True)

# Define a function to compute the output for each set of parameters
def compute_output(param_values):
    outputs = []
    for params in param_values:
        # Unpack the parameters
        Gb, kb, At_b, n_ab, f_ab, Bt_b, n_bb, f_bb, Gc, Bt_c, n_bc, f_bc, kc, k_CD, k_DE, k_minus_CD, k_minus_DE, k_P, k_minus_P = params
        
        # Create a parameter list
        params_list = [Gb, kb, At_b, n_ab, f_ab, Bt_b, n_bb, f_bb, Gc, Bt_c, n_bc, f_bc, kc, k_CD, k_DE, k_minus_CD, k_minus_DE, k_P, k_minus_P]
        
        # Compute the difference in fractional phosphorylation
        diff = return_differences(params_list)
        outputs.append(diff)
    return np.array(outputs)
# Compute the outputs for the generated parameter values
outputs = compute_output(param_values)

# Perform Sobol sensitivity analysis
sobol_indices = sobol.analyze(problem, outputs, calc_second_order=True)

# Print the first-order and total-order Sobol indices
"""
print("First-order Sobol indices:")
for name, S1 in zip(problem['names'], sobol_indices['S1']):
    print(f"{name}: {S1:.4f}")
print("\nTotal-order Sobol indices:")
for name, ST in zip(problem['names'], sobol_indices['ST']):
    print(f"{name}: {ST:.4f}")

"""

# Save the Sobol indices to a CSV file
sobol_df = pd.DataFrame({
    'Parameter': problem['names'],
    'First-order': sobol_indices['S1'],
    'First-order CI': sobol_indices['S1_conf'],
    'Total-order': sobol_indices['ST'],
    'Total-order CI': sobol_indices['ST_conf']
})
sobol_df.to_csv('sobol_indices_1000.csv', index=False)
#%%
#%%
S1 = sobol_indices['S1']
ST = sobol_indices['ST']
param_names = problem['names']

# Colorblind-friendly colors
color_s1 = '#0072B2'  # Blue
color_st = '#D55E00'  # Vermilion

# Use Arial font and size 16
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})

x = np.arange(len(param_names))
width = 0.35

plt.figure(figsize=(14, 6))
plt.bar(x - width/2, S1, width=width, label='First-order (S1)', color=color_s1, alpha=0.85)
plt.bar(x + width/2, ST, width=width, label='Total-order (ST)', color=color_st, alpha=0.85)

# Threshold lines
plt.axhline(0.05, color='gray', linestyle='--', linewidth=1.5, label='Threshold = 0.05')
plt.axhline(0.5, color='black', linestyle=':', linewidth=1.5, label='Threshold = 0.5')

plt.xticks(x, param_names, rotation=90)
plt.ylabel('Sobol Index')
plt.title('Sobol Sensitivity Indices')
plt.legend(frameon=False)
plt.tight_layout()
plt.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.7)

# Optional: Save
plt.savefig("sobol_indices.png", dpi=600)
plt.show()
#%%
df = pd.read_csv("J_vs_Se.tsv", sep="\t")


Se_values = df["Se"]
J_values1 = df["J1"]
J_values2 = df["J2"]
J_values3 = df["J3"]

plt.figure()
sns.lineplot(x = Se_values, y = J_values1, linewidth = 3, color = "black")
sns.lineplot(x = Se_values, y = J_values2, linewidth = 3, color = "red")
sns.lineplot(x = Se_values, y = J_values3, linewidth = 3, color = "orange")
plt.savefig('Supp4B.png', dpi=800)
plt.show()
sns.lineplot(x = Se_values, y = J_values1, linewidth = 3, color = "black")
sns.lineplot(x = Se_values, y = J_values2, linewidth = 3, color = "red")
sns.lineplot(x = Se_values, y = J_values3, linewidth = 3, color = "orange")
plt.xlim([0, 2.5])
plt.ylim([0, 350])
plt.savefig('Supp4B_inset.png', dpi=800)
plt.show()

# Compute numerical slope (dJ1/dSe)
slope = np.gradient(J_values1, Se_values)

print("Slope at highest Se values:")
print(pd.Series(slope[-10:], index=Se_values[-10:]))