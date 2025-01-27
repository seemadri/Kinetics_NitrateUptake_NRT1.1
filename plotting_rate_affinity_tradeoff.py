#%%
# import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# %%
# function that defines the 4 state model for transport nitrate by NRT1.1
# Pe: Nitrate bound outward facing state
# Pi: Nitrate bound inward facing state
# Ce: Outward facing state (no nitrate bound)
# Ci: Inward facing state (no nitrate bound)
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

    # Calculate the flux of nitrate
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

def plot_for_JasSe(R = 100, fold_change_in_r_for_unphos = 1e2, fi = 1e3, ratio = 1e3, internal_ratio = 1e-5):
    # R swiches a sigmoidal curve as a fucntion of Se
    Se_values = np.linspace(0, 10, 200)
    R_values1 = np.array(hills(Se_values, theshold=300, n=10))
    R_values1 = fold_change_in_r_for_unphos*R_values1*R + R*(1-R_values1)
    J_values1 = []
    R_values2 = np.array(hills(Se_values, theshold=0, n=10))
    R_values2 = fold_change_in_r_for_unphos*R_values2*R + R*(1-R_values2)
    J_values2 = []
    R_values3 = np.array(hills(Se_values, theshold=1.5, n=10))
    R_values3 = fold_change_in_r_for_unphos*R_values3*R + R*(1-R_values3)
    J_values3 = []
    for i,j in enumerate(Se_values):
        J_values1.append(solve_for_different_Se(j, R_values1[i], fi, ratio, internal_ratio))
        J_values2.append(solve_for_different_Se(j, R_values2[i], fi, ratio, internal_ratio))
        J_values3.append(solve_for_different_Se(j, R_values3[i], fi, ratio, internal_ratio))

    # save the values for J and Se
    df = pd.DataFrame({'Se': Se_values, 'J1': J_values1, 'J2': J_values2, 'J3': J_values3})
    df.to_csv('J_vs_Se.tsv', sep='\t')

    # plot the results
    plt.figure()
    plt.scatter(Se_values, J_values1)
    plt.scatter(Se_values, J_values2)
    plt.scatter(Se_values, J_values3)
    plt.show()
    plt.close()
    plt.scatter(Se_values, J_values1)
    plt.scatter(Se_values, J_values2)
    plt.scatter(Se_values, J_values3)
    plt.xlim([0, 3])
    plt.ylim([0, 350])
    plt.show()
    plt.close()

# plot the embo style figures
#plot_for_JasSe()


def plot_for_individual_parameters(R = 100, fold_change_in_r_for_unphos = 1e2, fi = 1e3, ratio = 1e3, internal_ratio = 1e-5):

    Se_values = np.linspace(0, 10, 150)
    J_values = [solve_for_different_Se(Se, R, fi, ratio, internal_ratio) for Se in Se_values]
    J_values_r_alternative = [solve_for_different_Se(Se, R*fold_change_in_r_for_unphos, fi, ratio, internal_ratio) for Se in Se_values]

    x = Se_values - internal_ratio*Se_values
    popt1, pcov1 = estimate_parameters(x, J_values)
    popt2, pcov2 = estimate_parameters(x, J_values_r_alternative)

    rate1 = popt1[0]
    affinity1 = 1/popt1[1]
    rate2 = popt2[0]
    affinity2 = 1/popt2[1]

    """plt.scatter(Se_values, J_values)
    plt.plot(x, michaelis_menten(x, *popt1), color='red')
    plt.scatter(Se_values, J_values_r_alternative)
    plt.plot(x, michaelis_menten(x, *popt2), color='green')
    plt.show()"""

    return rate1, affinity1, rate2, affinity2
#%%
# function to plot the fold change of rate and affinity for different values of fold_change_in_r_for_unphos
def plot_for_different_values_of_fold_change_in_r_for_unphos():
    range_of_fold_change_in_r_for_unphos = np.linspace(1, 100, 200)
    rate1 = []
    affinity1 = []
    rate2 = []
    affinity2 = []

    for i in range_of_fold_change_in_r_for_unphos:
        rate1_temp, affinity1_temp, rate2_temp, affinity2_temp = plot_for_individual_parameters(fold_change_in_r_for_unphos=i)
        rate1.append(rate1_temp)
        affinity1.append(affinity1_temp)
        rate2.append(rate2_temp)
        affinity2.append(affinity2_temp)

    # save the values for rate and affinity tradeoff with fold change in r due to P
    df = pd.DataFrame({'fold_change_in_r_for_unphos': range_of_fold_change_in_r_for_unphos, 'rate1': rate1, 'affinity1': affinity1, 'rate2': rate2, 'affinity2': affinity2})
    df.to_csv('rate_affinity_tradeoff.tsv', sep='\t')

    plt.scatter(range_of_fold_change_in_r_for_unphos, np.array(rate2)/np.array(rate1))
    plt.show()
    plt.close()

    plt.scatter(range_of_fold_change_in_r_for_unphos, np.array(affinity2)/np.array(affinity1))
    plt.show()

plot_for_different_values_of_fold_change_in_r_for_unphos()
#%% 
# function to estimate the rate as a function of absolute value of R (on a log scale)
def estimate_rate_as_function_of_R():
    R_values = 10**np.linspace(0, 5, 200)
    rate1 = []
    rate2 = []

    for i in R_values:
        rate1_temp, _, rate2_temp, _ = plot_for_individual_parameters(R=i)
        rate1.append(rate1_temp)
        rate2.append(rate2_temp)

    # save the values
    df = pd.DataFrame({'R': R_values, 'rate1': rate1, 'rate2': rate2, 'rate2/rate1': np.array(rate2)/np.array(rate1)})
    df.to_csv('rate_vs_R.tsv', sep='\t')

    plt.scatter(R_values, np.array(rate2)/np.array(rate1))
    plt.xscale('log')
    plt.show()
    plt.close()
    plt.scatter(R_values, rate1)
    plt.scatter(R_values, rate2)
    plt.xscale('log')
    plt.show()
    plt.close()

estimate_rate_as_function_of_R()

#%%
# function to estimate the rate as a function of absolute value of fi (on a log scale)
def estimate_rate_as_function_of_fi():
    fi_values = 10**np.linspace(1, 6, 100)
    rate1 = []
    rate2 = []

    for i in fi_values:
        rate1_temp, _, rate2_temp, _ = plot_for_individual_parameters(fi=i)
        rate1.append(rate1_temp)
        rate2.append(rate2_temp)

    sns.lineplot(x = fi_values, y = np.array(rate2)/np.array(rate1), linewidth = 3, color = "black")
    plt.xscale('log')
    plt.savefig('3E_i.png', dpi=800)
    plt.close()
    sns.lineplot(x = fi_values, y = rate1, linewidth = 3, color = "black")
    sns.lineplot(x = fi_values, y = rate2, linewidth = 3, color = "red")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('3E_ii.png', dpi=800)
    plt.close()

estimate_rate_as_function_of_fi()

#%%
# function to estimate the rate as a function of absolute value of internal_ratio (on a log scale)
def estimate_rate_as_function_of_ratio():
    ratio_values = 10**np.linspace(-6, -1, 10)
    rate1 = []
    rate2 = []

    for i in ratio_values:
        rate1_temp, _, rate2_temp, _ = plot_for_individual_parameters(internal_ratio=i)
        rate1.append(rate1_temp)
        rate2.append(rate2_temp)

    plt.scatter(ratio_values, np.array(rate2)/np.array(rate1))
    plt.xscale('log')
    plt.show()
    plt.close()
    plt.scatter(ratio_values, rate1)
    plt.xscale('log')
    plt.show()
    plt.close()

estimate_rate_as_function_of_ratio()
#%%
# individual parameters plotting
plot_for_individual_parameters()
#%%
# make plot of rate and affinity for different values of fold_change_in_r_for_unphos

range_of_fold_change_in_r_for_unphos = np.linspace(1, 50, 10)
rate1 = []
affinity1 = []
rate2 = []
affinity2 = []

for i in range_of_fold_change_in_r_for_unphos:
    rate1_temp, affinity1_temp, rate2_temp, affinity2_temp = plot_for_individual_parameters(fold_change_in_r_for_unphos=i)
    rate1.append(rate1_temp)
    affinity1.append(affinity1_temp)
    rate2.append(rate2_temp)
    affinity2.append(affinity2_temp)

plt.scatter(range_of_fold_change_in_r_for_unphos, np.array(rate2)/np.array(rate1))
plt.show()
plt.close()

plt.scatter(range_of_fold_change_in_r_for_unphos, np.array(affinity2)/np.array(affinity1))
plt.show()

#%%

arr1 = [[],[]]
arr2 = [[],[]]
for i in range(100):
    rate1, affinity1, rate2, affinity2 = sampling()
    arr1[0].append(rate1)
    arr1[1].append(affinity1)
    arr2[0].append(rate2)
    arr2[1].append(affinity2)

plt.scatter(arr1[0], arr1[1])
plt.show()
#%%
Nitrate_high = [0.0380952380952381, 0.06984126984126984, 0.15555555555555556, 0.2571428571428571,  0.6799999999999997, 5.32, 10.120000000000001, 19.72, 29.48]
J_high = [106.66666666666666, 207.4074074074074, 296.2962962962963, 302.22222222222223, 292.53731343283584, 298.5074626865672, 405.9701492537314, 310.44776119402985, 447.7611940298508]

Nitrate_low = [0.24776119402985075, 0.779220779220779, 1.103896103896104, 4.675324675324675, 9.870129870129869, 19.772727272727273, 29.675324675324674]
J_low = [13.422818791946309, 201.34228187919464, 275.1677852348993, 570.4697986577181, 738.255033557047, 953.0201342281879, 1053.6912751677853]

popt1, pcov1 = estimate_parameters(Nitrate_high, J_high)
popt2, pcov2 = estimate_parameters(Nitrate_low, J_low)

print(popt1, popt2)

# plot the results
plt.figure()
# plot the noisy data
plt.scatter(Nitrate_low, J_low, label='Noisy data')

x = np.linspace(0, 30, 30)
# plot the fitted data
plt.plot(x, michaelis_menten(x, *popt2), color='red', label='Fitted data')
# %%
def sampling():
    R = 10**np.random.uniform(2, 4)
    fold_change_in_r_for_unphos = 1e2 #10**np.random.uniform(1, 3)
    fi = 10**np.random.uniform(2, 5)
    ratio = 10**np.random.uniform(2, 4)
    internal_ratio = 1e-5#10**np.random.uniform(-4, -2)

    Se_values = np.linspace(0, 10, 100)
    J_values = [solve_for_different_Se(Se, R, fi, ratio, internal_ratio) for Se in Se_values]
    J_values_r_alternative = [solve_for_different_Se(Se, R*fold_change_in_r_for_unphos, fi, ratio, internal_ratio) for Se in Se_values]

    x = Se_values - internal_ratio*Se_values
    popt1, pcov1 = estimate_parameters(x, J_values)
    popt2, pcov2 = estimate_parameters(x, J_values_r_alternative)

    rate1 = popt1[0]
    affinity1 = 1/popt1[1]
    rate2 = popt2[0]
    affinity2 = 1/popt2[1]

    return rate1, affinity1, rate2, affinity2