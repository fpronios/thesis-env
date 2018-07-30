import numpy as np
import sklearn as sk
import pandas as pd
import glob
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel

path =r'C:\DRO\DCL_rawdata_files'

class user_model():

    def __init__(self, filepath):
        pass

    def load_dishasher(self):
        pass




if __name__ == '__main__':
    """
    path = 'files/GREEND'
    allFiles = glob.glob(path + "/*.csv")
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0, usecols = [2])
        list_.append(df)
    frame = pd.concat(list_, axis = 1)
    frame = frame.fillna(0.0)
    print(frame.describe)
    print(frame.head)

    #writer = pd.ExcelWriter('files/GREEND/refrigerator_greend.xlsx')
    #frame.to_excel(writer, 'refrigerator')
    # writer.save()

    #print(down_s.describe())
    numpy_all = frame.values
    #tst = frame.iloc[:, [96]]
    #tst.resample('1T').sum()
    print(numpy_all.shape)
    numpy_all = numpy_all.reshape((1603,53,472))
    print(numpy_all.shape)
    numpy_tst = numpy_all
    numpy_tst = numpy_tst.mean(axis = 1)
    print(numpy_tst.shape)

    np.savetxt('dishwasher.out', numpy_tst, delimiter=',')
    print(numpy_tst.shape)
    plt.plot(numpy_tst[:,200])
    plt.show()
    """

    np_all = np.loadtxt(open("dishwasher.out", "rb"), delimiter=",", skiprows=0)
    print(np_all.shape)
    print(np_all)
    plt.plot(np_all[:,266])
    plt.show()

    #np_all = np_all[33:-34,:]
    #print(np_all.shape)
    #"""

    G = BayesianModel()
    G.add_node('time_slot')

    G.add_edge('time_slot', 'device_on')
    G.add_edge('time_slot', 'energy')
    G.add_edge('time_slot', 'duration')
    G.add_edge('device_on', 'energy')
    G.add_edge('device_on', 'duration')


    for i in range(np_all.shape[1]):
        gtz_idxb = np_all[:,i] > 10
        total_energy = np_all[gtz_idxb, i].sum()

        gtz_idx = np.argwhere(np_all[:, i] > 10)
        train_list = []
        if gtz_idx.shape[0] > 0 and total_energy < 60000:

            print('IDX: ',i ,' Total Energy: ',total_energy,'Start: ' , gtz_idx[0] ,'  END:' ,gtz_idx[-1])
            #print('Delta:', gtz_idx[-1] - gtz_idx[0])
            print('Delta:', (gtz_idx[-1] - gtz_idx[0])*(53)* (1 /60 ) //15, '(timeslots)')
            print('Start' ,gtz_idx[0] *(53)* (1 /60 ) //15, '(timeslots)')
            timeslot = gtz_idx[0] *(53)* (1 /60 ) //15
            duration = (gtz_idx[-1] - gtz_idx[0])*(53)* (1 /60 ) //15
            starting = 'yes'
            total_energy = 0
    input("before train")
    data = pd.DataFrame(np_all)
    G.fit(data)
    G.get_cpds()