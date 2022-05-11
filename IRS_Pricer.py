import math
import numpy as np 
import IRS_Pricer
import HJM_Calibrator

def swap_payoff(f_rate, c_rate, payer=True):
    if payer == True:
        return (f_rate - c_rate)/100.0
    else:
        return (c_rate - f_rate)/100.0

def get_discount_factor(f_curve, dTau, maturity):
    df = 1
    if maturity <= 0:
        df = 1
    else:
        n = int(maturity/dTau)
        for i in range(1, n):
            df = df*math.exp(-f_curve[i-1]/100*dTau)
    return df

def swap_payoffs(f_curve, maturity, dTau, c_rate, payer=True):
    n = int(maturity/dTau)
    swap_payoffs = [0] * (n*n)
    swap_payoffs = np.reshape(swap_payoffs, (n, n))
    swap_MtM = [0] * n
    swap_Expos = [0] * n
    for i in range(0, n):

        for j in range(0, n-i):
            df = get_discount_factor(f_curve, dTau, j+1)
            swap_payoffs[i][j+i] = swap_payoff(f_curve[i][j], c_rate, payer) * dTau * df
            swap_MtM[i] = swap_MtM + swap_payoffs[i][j+1]
        swap_Expos = max(swap_MtM[i], 0)
    
    return {'payoffs':swap_payoffs, 'MtM':swap_MtM, 'Exposure':swap_Expos }


def swap_payoffsOIS(f_curve, maturity, dTau, c_rate, o_curve, payer=True):
    n = int(maturity / dTau)
    pv_swap_payoffs = np.zeros((n, n))
    swap_payoffs = np.zeros((n, n))
    swap_MtM = [0] * n
    swap_Expos = [0] * n
    for i in range(0, n):

        for j in range(0, n - i):
            df = get_discount_factor(o_curve[i], dTau, (j+1)*0.5)
            swap_pf= swap_payoff(f_curve[i][j], c_rate, payer) * dTau
            swap_payoffs[i][j + i] = swap_pf
            pv_swap_payoffs[i][j + i] = swap_pf * df
            swap_MtM[i] = swap_MtM[i] + pv_swap_payoffs[i][j + i]
        
        swap_Expos[i] = max([swap_MtM[i], 0])

    return {'payoffsPV' : pv_swap_payoffs, 'MtM' : swap_MtM, 'Exposure' : swap_Expos, 'payoffs' : swap_payoffs}


def swap_payoffSimulation(drifts, vols, lf_curve, maturity, c_rate,  tenors, dTau, steps, sim):
    #store CVA data for statistics
    ee_simulation = np.zeros((sim, int(maturity / dTau)))
    mtm_simulation = np.zeros((sim, int(maturity / dTau)))
    for i in range(0, sim):
        #Todo Find the latest instantous Forward Curve
        if i == 0:
            print('Simulation run ', i + 1)
        elif i == 99:
            print('Simulation run ', i + 1)
        elif i == 499:
            print('Simulation run ', i + 1)
        elif i == 999:
            print('Simulation run ', i + 1)
        elif i == 4999:
            print('Simulation run ', i + 1)
        f_curve = np.zeros(
            (int(maturity) * int(1 / dTau) * steps, len(tenors)))
        f_curve[0] = lf_curve.values
        for j in range(1, int(maturity) * int(1 / dTau) * steps):
            f_curve[j] = HJM_Calibrator.fcurve_HJM_MusielaParams(
                f_curve[j - 1], drifts.values[0] * 100.0, np.reshape(vols.values * 100.0, (len(tenors), 3)), 1 / steps)

        #trim the curve as only part of the curves will be used to calculate the PV
        num = int(maturity / dTau) + 1
        ff_curve = np.zeros(
            (num, int(maturity) * int(1 / dTau)))
        for n in range(0, num):
            ff_curve[n] = f_curve[int(n * steps * dTau)
                                  ][:int(maturity) * int(1 / dTau)]

        #Todo Find Forward-OIS Spread
        o_curve = ff_curve - 0.2
        asset_payoffs = IRS_Pricer.swap_payoffsOIS(
            ff_curve, maturity, dTau, c_rate, o_curve, True)
        ee_simulation[i] = asset_payoffs['Exposure']
        mtm_simulation[i] = asset_payoffs['MtM']
    np.savetxt('EE_' + str(sim) + '.csv',
               np.asarray(ee_simulation), delimiter=",")
    np.savetxt('MtM_' + str(sim) + '.csv',
               np.asarray(mtm_simulation), delimiter=",")
    return 0
