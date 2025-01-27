#%%
import numpy as np
from scipy.integrate import odeint
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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
    t = np.linspace(0, 2000, 10000)
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
    #plt.ylim([0,1702000])
    plt.show()

plot_for_JasSe()

# %%
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

    plot_classical(B_array, C_array, D_array, E_array, CD_array, DE_array, P_array, A_array)

    # save the values for all the variables
    df = pd.DataFrame({'A': A_array, 'B': B_array, 'C': C_array, 'D': D_array, 'E': E_array, 'CD': CD_array, 'DE': DE_array, 'P': P_array})
    df.to_csv('steady_states.tsv', sep='\t', index=False) 
    return frac[0] - frac[-1]

return_differences(params)

#%%
# k_minus_P
"""variable = np.linspace(0.01, 0.1, 10)
array_temp = []
for i in variable:
    k_minus_P = i
    params = [Gb, kb, At_b, n_ab, f_ab, Bt_b, n_bb, f_bb, Gc, Bt_c, n_bc, f_bc, kc, k_CD, k_DE, k_minus_CD, k_minus_DE, k_P, k_minus_P]
    array_temp.append(return_differences(params))"""

# k_p
"""variable = np.logspace(-5, -1, 10)
array_temp = []
for i in variable:
    k_P = i
    params = [Gb, kb, At_b, n_ab, f_ab, Bt_b, n_bb, f_bb, Gc, Bt_c, n_bc, f_bc, kc, k_CD, k_DE, k_minus_CD, k_minus_DE, k_P, k_minus_P]
    array_temp.append(return_differences(params))"""

# gB
"""variable = np.linspace(10, 100, 10)
array_temp = []
for i in variable:
    Gb = i
    params = [Gb, kb, At_b, n_ab, f_ab, Bt_b, n_bb, f_bb, Gc, Bt_c, n_bc, f_bc, kc, k_CD, k_DE, k_minus_CD, k_minus_DE, k_P, k_minus_P]
    array_temp.append(return_differences(params))"""

#gC
"""variable = np.linspace(1, 100, 10)
array_temp = []
for i in variable:
    Gc = i
    array_temp.append(return_differences())"""

# Bt_c
"""variable = np.linspace(1000, 5000, 10)
array_temp = []
for i in variable:
    Bt_c = i
    params = [Gb, kb, At_b, n_ab, f_ab, Bt_b, n_bb, f_bb, Gc, Bt_c, n_bc, f_bc, kc, k_CD, k_DE, k_minus_CD, k_minus_DE, k_P, k_minus_P]
    array_temp.append(return_differences(params))"""

# f_bc
"""variable = np.linspace(1, 50, 10)
array_temp = []
for i in variable:
    f_bc = i
    params = [Gb, kb, At_b, n_ab, f_ab, Bt_b, n_bb, f_bb, Gc, Bt_c, n_bc, f_bc, kc, k_CD, k_DE, k_minus_CD, k_minus_DE, k_P, k_minus_P]
    array_temp.append(return_differences(params))"""

# k_CD
"""variable = np.linspace(0.1, 1, 10)
array_temp = []
for i in variable:
    k_CD = i
    params = [Gb, kb, At_b, n_ab, f_ab, Bt_b, n_bb, f_bb, Gc, Bt_c, n_bc, f_bc, kc, k_CD, k_DE, k_minus_CD, k_minus_DE, k_P, k_minus_P]
    array_temp.append(return_differences(params))

plt.scatter(variable, array_temp)
#plt.xscale('log')
plt.show()"""

#%%
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})
df = pd.read_csv('steady_states.tsv', sep='\t')

df1 = df[df['B'] < 2000]
df2 = df[df['B'] > 2000]

sns.lineplot(x='A', y='B', data=df1, linewidth=2.5, color='black')
sns.lineplot(x='A', y='P', data=df1, linewidth=2.5, color='red')
sns.lineplot(x='A', y='B', data=df2, linewidth=2.5, color='black')
sns.lineplot(x='A', y='P', data=df2, linewidth=2.5, color='red')
plt.yscale('log')
plt.savefig("NRT1.1.png", dpi=800) 
plt.close() 

sns.lineplot(x=df1['A'], y=df1['P']/(df1['B']+df1['P']), linewidth=2.5, color='black')
sns.lineplot(x=df2['A'], y=df2['P']/(df2['B']+df2['P']), linewidth=2.5, color='black')
plt.savefig("Prop_of_phosphorylated_NRT1.1.png", dpi=800)   



# %%
