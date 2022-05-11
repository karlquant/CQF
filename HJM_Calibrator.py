import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import math


def his_data_loader(path):
    his_data = pd.read_csv(path)
    return his_data

def get_returns(his_data):
    tenors = list(his_data.columns.values)
    n = len(his_data[tenors[0]])
    his_returns = [0.0] * (len(tenors) - 1) * (n - 1)
    his_returns = np.reshape(his_returns, (n-1, len(tenors)-1))
    for i in range(1, n):
        for t in range(1, len(tenors)):
           his_returns[i-1][t-1] = his_data[tenors[t]][i] - his_data[tenors[t]][i-1]
    return his_returns


def get_coVMatrix(his_returns, stan=True):
    if(stan):
        return np.cov(np.transpose(his_returns)) * 252 / 10000
    else:
        return np.cov(np.transpose(his_returns)) * 252 / 10000

def get_EigenValues(eVaMatrix):
    n = len(eVaMatrix['eigenValues'])
    eigenVa = [0] * n
    eigenVe = np.transpose(eVaMatrix['eigenVectors'])
    print(eVaMatrix['eigenValues'])
    for i in range(0, n):
        eigenVa[i] = eVaMatrix['eigenValues'][i][i]
    eigenVa.sort(reverse=True)
    var_i = [0] * 3
    var_v = [0] * 3
    var_e = [0] * 3
    for i in range(0, n):
        if eVaMatrix['eigenValues'][i][i] == eigenVa[0]:
            var_i[0] = i
            var_v[0] = eigenVa[0]
            var_e[0] = eigenVe[i]
        elif eVaMatrix['eigenValues'][i][i] == eigenVa[1]:
            var_i[1] = i
            var_v[1] = eigenVa[1]
            var_e[1] = eigenVe[i]
        elif eVaMatrix['eigenValues'][i][i] == eigenVa[2]:
            var_i[2] = i
            var_v[2] = eigenVa[2]
            var_e[2] = eigenVe[i]
    r2_score = (var_v[0] + var_v[1] + var_v[2]) / sum(eigenVa) * 100
    print("R-Squared %.2f" % r2_score)
    return [var_i, var_v, var_e]      

""" def get_EigenVectors(eVeMatrix, r_tenors, tenor):
    if tenor <= 3 and tenor >= 1:
        t = tenor
        tenors = eVeMatrix['tenors']
        y_train = eVeMatrix['evectors']
        if t == 1:
            x_train = tenors
            x_train.reshape((len(tenors), 1))
            x_test = r_tenors
            x_test.reshape((len(r_tenors), 1))
        else:
            x_train['tau'] = tenors
            x_train['tau_2'] = list(map(lambda x: x**2, tenors))
            x_train['tau_3'] = list(map(lambda x: x**3, tenors))
            x_train.reshape((len(tenors), 3))
            x_test['tau'] = r_tenors
            x_test['tau_2'] = list(map(lambda x: x**2, r_tenors))
            x_test['tau_3'] = list(map(lambda x: x**3, r_tenors))
            x_test.reshape((len(r_tenors), 3))
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        y_predict = regr.predict(x_train)
        print(regr)
        print('Coefficients: \n', regr.coef_)
        print("Mean squared error: %.2f" % mean_squared_error(y_train, y_predict))
        print("R-Squared Score: %.2f" % r2_score(y_train, y_predict))
    return regr.predict(x_test) """

def get_Vol_Function(eVeMatrix, r_tenors, pom):
    if pom <=7 and pom > 0:
        y_train = eVeMatrix
        #print(y_train)
        x_train = r_tenors
        #print(len(x_train))
        c, stats = np.polynomial.polynomial.polyfit(x_train, y_train, pom, full=True)        
        print('The degree of polynomial fitting: ', pom)
        print('Coefficients: \n', c)
        print('Stats: \n', stats)
    return {'coeff': c, 'stats': stats}


def vol_nHJM_Solver(vols, straight=True):
    tenors = vols['tenors']
    y_train = vols['vols']
    if straight==True:
        x_train = tenors
        x_train.reshape((len(tenors), 1))
    else:
        x_train['tau'] = tenors
        x_train['tau_2'] = list(map(lambda x: x**2, tenors))
        x_train['tau_3'] = list(map(lambda x: x**3, tenors))
        x_train.reshape((len(tenors), 3))
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    y_predict = regr.predict(x_train)
    print(regr)
    print('Coefficients: \n', regr.coef_)
    print("Mean Squared Error: %.2f" % mean_squared_error(y_train, y_predict))
    print("R-Squared Score: %.2f" % r2_score(y_train, y_predict))
    return regr

def vol_Generator(vols, Tau):
    betaS = vols[1:]
    i = 1
    vol = vols[0]
    for beta in betaS:
        vol = vol + beta * Tau**i
        i+=1
    return vol
        

def drift_HJM_Generator_TR(vols, tenor, dTau):
#Numerical Implementation Trapezoidal Rule
    if tenor == 0:
        return 0
    else:
        n = int(tenor/dTau)

        tR = vols[0]
        for i in range(1, n):
            tR = tR + 2 * vols[i]
        tR = tR + vols[n]
        tR = tR * 0.5 * dTau
        tR = tR * vols[n]
    return tR

def drift_HJM_Generator_SR(vols, tenor, dTau):
#Numberical Implementation Simpson's Rule
    if tenor == 0:
        return 0
    else:
        n = int(tenor/dTau)

        sR = vol_Generator(vols, 0)
        sR = sR + 4 * vol_Generator(vols, 0.5)
        for i in range(1, n):
            sR = sR + 2 * vol_Generator(vols, i*dTau)
            x = i + 0.5
            sR = sR + 4 * vol_Generator(vols, x)
        x = n - 0.5
        sR = sR + 4 * vol_Generator(vols, x)
        sR = sR + vol_Generator(vols, tenor)
        sR = (sR * dTau)/6
    return sR

def vol_sde_HJM(vols, dTau):
    dVol = (vols[0] * np.random.randn() + vols[1] * np.random.randn() + vols[2] * np.random.randn())
    return dVol

def drift_sde_HJM(vols, tenor, dTau):
    dRfit = 0 
    for vol in vols:
        dRfit += drift_HJM_Generator_TR(vol, tenor, dTau)
    return dRfit

def fcurve_HJM_MusielaParams(c_curve, drift, vols, dTau):
    f_curve = [0] * len(c_curve)
    for j in range(0, len(c_curve)):
        if j == len(c_curve) - 1:
            f_curve[j] = c_curve[j] + drift[j] * dTau + vol_sde_HJM(
                vols[j], dTau) * math.sqrt(dTau) + (c_curve[j] - c_curve[j-1]) * dTau * 2
        else:
            f_curve[j] = c_curve[j] + drift[j] * dTau + vol_sde_HJM(
                vols[j], dTau) * math.sqrt(dTau) + (c_curve[j + 1] - c_curve[j]) * dTau * 2


    return f_curve



#Calculate the parameters of HJM Model, mean, Vol1, Vol2, Vol3
#Todo make up the history data file (past 3-5 years)
his_data = HJM_Calibrator.his_data_loader('boe.csv')
his_returns = HJM_Calibrator.get_returns(his_data)
his_coVar = HJM_Calibrator.get_coVMatrix(his_returns)
eigen_Vs = pca_Solver.Jacobi_Transformation(his_coVar, 1e-20)
np.savetxt('EigenValues.csv', np.asarray(eigen_Vs['eigenValues']), delimiter=",")
np.savetxt('EigenVectors.csv', np.asarray(
    eigen_Vs['eigenVectors']), delimiter=",")
Vars = HJM_Calibrator.get_EigenValues(eigen_Vs)
lam = list(map(lambda x: math.sqrt(x), Vars[1]))
vols_Matrix = [np.array(Vars[2][0]) * lam[0],
               np.dot(lam[1], Vars[2][1]), np.dot(lam[2], Vars[2][2])]


np.savetxt('Vols.csv', np.asarray(
    vols_Matrix), delimiter=",")

tenors = np.arange(0.0, 25.5, 0.5)

#Vol1 is an constant while Vol2&Vol3 are linear functions
vol1 = np.median(vols_Matrix[0])

print("Vol(2) fitting: \n")
vol2 = HJM_Calibrator.get_Vol_Function(
    vols_Matrix[1], tenors, 3)

print("Vol(3) fitting: \n")
vol3 = HJM_Calibrator.get_Vol_Function(
    vols_Matrix[2], tenors, 3)
print(np.polynomial.polynomial.polyval(tenors, [vol1]))
print(np.polynomial.polynomial.polyval(
    tenors, vol2['coeff']))
fittedVols = [tenors, np.polynomial.polynomial.polyval(tenors, [vol1]), np.polynomial.polynomial.polyval(
    tenors, vol2['coeff']), np.polynomial.polynomial.polyval(tenors, vol3['coeff'])]
np.savetxt('FittedVols.csv', np.asarray(
    fittedVols), delimiter=",")


drfits = np.zeros((1, len(tenors)))
vols = []

for i in range(1, len(tenors)):
    print(i)
    r_tenors = np.arange(0.0, tenors[i] + 0.01, 0.01)
    drfits[i] = HJM_Calibrator.drift_HJM_Generator_TR(
        np.polynomial.polynomial.polyval(r_tenors, [vol1]), tenors[i], 0.01)
    + HJM_Calibrator.drift_HJM_Generator_TR(
        np.polynomial.polynomial.polyval(r_tenors, vol2['coeff']), tenors[i], 0.01)
    + HJM_Calibrator.drift_HJM_Generator_TR(
        np.polynomial.polynomial.polyval(r_tenors, vol3['coeff']), tenors[i], 0.01)

np.savetxt('Mean.csv', np.asarray(
    drfits), delimiter=",")
