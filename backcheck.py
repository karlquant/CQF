import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


def cva_check_percentile(sim):
    mtm_simulation = pd.read_csv('MtM_' + str(sim) + '.csv').values
    ee_simulation = pd.read_csv('EE_' + str(sim) + '.csv').values
    plt.subplot(2, 1, 1)
    plt.plot(np.percentile(mtm_simulation, 50, axis=0), label='50%')
    plt.plot(np.percentile(mtm_simulation, 90, axis=0), label='90%')
    plt.plot(np.percentile(mtm_simulation, 95, axis=0), label='95%')
    plt.plot(np.percentile(mtm_simulation, 99, axis=0), label='99%')
    plt.legend()
    plt.title('Present Value of IRS')
    plt.ylabel('MtM')
    plt.subplot(2, 1, 2)
    plt.plot(np.percentile(ee_simulation, 50, axis=0), label='50%')
    plt.plot(np.percentile(ee_simulation, 90, axis=0), label='90%')
    plt.plot(np.percentile(ee_simulation, 95, axis=0), label='95%')
    plt.plot(np.percentile(ee_simulation, 99, axis=0), label='99%')
    plt.legend(shadow=True)
    plt.title('Expected Exposure of IRS')
    plt.ylabel('EE')
    plt.show()


def cva_check(sim):
    mtm_simulation = pd.read_csv('MtM_' + str(sim) + '.csv').values
    ee_simulation = pd.read_csv('EE_' + str(sim) + '.csv').values
    plt.subplot(2, 1, 1)
    plt.plot(np.percentile(mtm_simulation, 50, axis=0), label='50%')
    plt.title('Present Value of IRS')
    plt.ylabel('MtM')
    plt.subplot(2, 1, 2)
    plt.plot(np.percentile(ee_simulation, 50, axis=0), label='50%')
    plt.title('Expected Exposure of IRS')
    plt.ylabel('EE')
    plt.show()

def cva_check_his(sim, tenor):
    mtm_simulation = pd.read_csv('MtM_' + str(sim) + '.csv').values
    ee_simulation = pd.read_csv('EE_' + str(sim) + '.csv').values
    num_bins = 50
    print(np.array(mtm_simulation[:, tenor]))
    plt.subplot(2, 1, 1)
    plt.hist(np.array(mtm_simulation[:, tenor]), num_bins, normed=1)
    plt.title('Histogram of Tenor ' + str(tenor))
    plt.xlabel('MtM')
    plt.ylabel('% Percentage')
    plt.subplot(2, 1, 2)
    plt.plot(np.array(ee_simulation[:, tenor]), num_bins)
    plt.title('Histogram of Tenor ' + str(tenor))
    plt.ylabel('EE')
    plt.show()


def cva_vcheck_his(sim):
    cva_simulation = pd.read_csv('cva_' + str(sim) + '.csv').values
    num_bins = 50
    plt.hist(np.array(cva_simulation), num_bins, normed=1)
    plt.xlabel('CVA')
    plt.ylabel('% Percentage')
    plt.show()
    print(np.percentile(cva_simulation, 50))
    print(np.percentile(cva_simulation, 90))
    print(np.percentile(cva_simulation, 95))
    print(np.percentile(cva_simulation, 99))

cva_check_his(10000, 1)
cva_check_his(10000, 2)
cva_vcheck_his(10000)
