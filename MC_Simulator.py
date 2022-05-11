import pca_Solver
import HJM_Calibrator
import CVA_simulator
import IRS_Pricer
import pandas as pd 
import numpy as np 
import math
import matplotlib.pyplot as plt


drifts = pd.read_csv('Mean.csv')
vols = pd.read_csv('FittedVols.csv')
lf_curve = pd.read_csv('latest_forward_curve.csv')
#Start Simulation for CVA
#Contract information of Interest Rate Swap
maturity = 5.0
dTau = 0.5
c_rate = 1.0

#Simulation Parameters
simulation = [1000, 10000]
steps = 100

tenors = np.arange(0.0, 25.5, 0.5)


for sim in simulation:
    IRS_Pricer.swap_payoffSimulation(
        drifts, vols, lf_curve, maturity, c_rate, tenors, dTau, steps, sim)


recovery_rate = 0.4
h_rate_curve = [0.02] * int(maturity / dTau)
df_curve = [0] * (int(maturity / dTau) + 1)
df_curve[0] = 1.0
pd_curve = [0] * (int(maturity / dTau) + 1)
pd_curve[0] = 0.0

for k in range(0, int(maturity / dTau)):
    df_curve[k + 1] = IRS_Pricer.get_discount_factor(
        lf_curve.values[0], dTau, (k + 1) * 0.5)
    pd_curve[k + 1] = math.exp(-h_rate_curve[k] * dTau)


#CVA
for sim in simulation:    
    ee_simulation = pd.read_csv('EE_' + str(sim) + '.csv')
    CVA_simulator.get_IRSCVA_Simulation(ee_simulation.values, df_curve, pd_curve, recovery_rate, sim)








        

        


