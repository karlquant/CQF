import IRS_Pricer
import math
import numpy as np

def get_sP(h_rate_curve, tenor, dTau):
    n = int(tenor / dTau)
    sP = 1
    for i in range(1, n):
        sP =  sP * math.exp(-i*dTau*h_rate_curve[i])
    return sP



def get_IRSCVA(asset_ee, df_curve, pd_curve, recovery_rate):
    cva_Value = 0
    n = len(asset_ee)
    cva_period = [0] * n
    for i in range(0, n-1):
        if i < n-1:
            cva_period[i] = (asset_ee[i] + asset_ee[i + 1]) * 0.5 * \
                df_curve[i + 1] * (1 - recovery_rate) * pd_curve[i + 1]
        else:
            cva_period[i] = (asset_ee[i] + 0) * 0.5 * \
                df_curve[i + 1] * (1 - recovery_rate) * pd_curve[i + 1]
        cva_Value += cva_period[i]
        
    return cva_Value


def get_IRSCVA_Simulation(ee_simulation, df_curve, pd_curve, recovery_rate, sim):
    cVa = [0] * len(ee_simulation)
    for s in range(0, len(ee_simulation)):
        cVa[s] = get_IRSCVA(ee_simulation[s], df_curve, pd_curve, recovery_rate)
        np.savetxt('cva_' + str(sim) + '.csv', np.asarray(cVa), delimiter=",")
