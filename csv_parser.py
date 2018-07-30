building0 = ["Coffee machine", "washing machine", "radio", "water kettle", "fridge w/ freezer", "dishwasher",
             "kitchen lamp", "TV", "vacuum cleaner"]
building1 = ['Radio', 'freezer', 'dishwasher', 'fridge', 'washing machine', 'water kettle', 'blender', 'network router']
building2 = ['Fridge', 'dishwasher', 'microwave', 'water kettle', 'washing machine', 'radio w/ amplifier', 'dryier',
             'kitchenware (mixer and fruit juicer)', 'bedside light']
building3 = ['TV', 'NAS', 'washing machine', 'drier', 'dishwasher', 'notebook', 'kitchenware', 'coffee machine',
             'bread machine']
building4 = ['Entrance outlet', 'Dishwasher', 'water kettle', 'fridge w/o freezer', 'washing machine', 'hairdrier',
             'computer', 'coffee machine', 'TV']
building5 = ['Total outlets', 'total lights', 'kitchen TV', 'living room TV', 'fridge w/ freezer', 'electric oven',
             'computer w/ scanner and printer', 'washing machine', 'hood']
building6 = ['Plasma TV', 'lamp', 'toaster', 'hob', 'iron', 'computer w/ scanner and printer', 'LCD TV',
             'washing machine', 'fridge w/ freezer']
building7 = ['Hair dryer', 'washing machine', 'videogame console and radio', 'dryer',
             'TV w/ decoder and computer in living room', 'kitchen TV', 'dishwasher', 'total outlets', 'total lights']
building8 = ['Kitchen TV', 'dishwasher', 'living room TV', 'desktop computer w/ screen', 'washing machine',
             'bedroom TV', 'total outlets', 'total lights']
import numpy as np
import sklearn as sk
import pandas as pd
import glob
import time
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel

path =r'C:\DRO\DCL_rawdata_files'




if __name__ == '__main__':
    path = 'files/GREEND/building7'
    allFiles = glob.glob(path + "/*.csv")
    list_ = []
    for file_ in allFiles:
        print(file_)
        np_temp = np.genfromtxt(file_, delimiter=',',skip_header =1,usecols=(0,4) ,filling_values = 0.0)
        #df = pd.read_csv(file_, index_col=None, header=0, usecols = [0,2])
        #list_.append(df)
        #print(df.head())

        #writer = pd.ExcelWriter('files/GREEND/dishwasher_daily.xlsx')
        #df.to_excel(writer, 'refrigerator')
        #writer.save()


        #df = df[df.timestamp.str.contains("timestamp") == False]
        #writer = pd.ExcelWriter('files/GREEND/dishwasher_daily.xlsx')
        #df.to_excel(writer, 'refrigerator')
        #writer.save()
        #asd = input("Asdda")
        np_temp = np_temp[np_temp[:, 0] != 'timestamp']

        time_start = np_temp[0,0]
        #print(time_start)
        np_temp[:,0] = np_temp[:,0] - time_start
        np_temp[np_temp > 86399] = 83999
        buffer_np = np.zeros((86400,1))

        buffer_np[np_temp[:,0].astype(int),0] = np_temp[:,1]

        math_temp_step = buffer_np[np.argwhere(buffer_np == 0)[1:-1,0] +1 ]/2 +  buffer_np[np.argwhere(buffer_np == 0)[1:-1,0] - 1]/2
        buffer_np[buffer_np == 0.0][1:-1] = math_temp_step[:,0]

        tdf = pd.DataFrame(buffer_np)
       # print(tdf.head())
        list_.append(tdf)


    frame = pd.concat(list_, axis = 1)
    frame = frame.fillna(0.0)
    print(frame.describe)


    #frame[frame.timestamp.str.contains("timestamp") == False]
    #writer = pd.ExcelWriter('files/GREEND/refrigerator_greend.xlsx')
    #frame.to_excel(writer, 'refrigerator')
    # writer.save()

    #print(down_s.describe())
    numpy_all = frame.values
    #tst = frame.iloc[:, [96]]
    #tst.resample('1T').sum()
    print(numpy_all.shape)
    numpy_all = numpy_all.reshape((1440,60,133))
    print(numpy_all.shape)
    numpy_tst = numpy_all
    numpy_tst = numpy_tst.mean(axis = 1)
    print(numpy_tst.shape)

    np.savetxt('dryer.out', numpy_tst, delimiter=',')
    print(numpy_tst.shape)
    plt.plot(numpy_tst[:,100])
    plt.show()