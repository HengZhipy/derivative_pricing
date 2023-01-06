from datetime import datetime
from WindPy import w
w.start()


import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
from WindPy import w
w.start()

def pickleload(file,disp=1):
    with open(file, 'rb') as f:
        pickle_data = pickle.load(f)  # 反序列化，与pickle.dump相反
        loaddata=pickle_data.copy()
        del pickle_data
        if disp==1:
            print('Data and modules in '+file+' loaded.')
        return loaddata  # 释放内存
    
def picklesave(file,savedict,disp=1,protocolnum=100):
    if not os.path.isfile(file):      # 判断是否存在此文件，若无则存储，若有则跳过（不存储）
        if disp==1:
            print('Saving data to pickle file...')
        try:
            with open(file, 'wb') as pfile:
                if protocolnum==100:
                    pickle.dump(savedict, pfile, pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(savedict, pfile, protocol=protocolnum)
        except Exception as e:
            print('Unable to save data to', file, ':', e)
            raise
    if disp==1:
        print('Data cached in pickle file.')

def del_files(file_path):
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
        except BaseException as e:
            print(e)
    print('File removed.')

def getdaylist(startTimeStr,endTimeStr,periodStr,type='local'):
  if type=='local':
    daylist_localdata=pickleload('daylist_localdata.pickle')
    if periodStr=='D':
      data=[x for x in daylist_localdata['date_data'] if (x>=datetime.strptime(startTimeStr,'%Y-%m-%d').date())&
                                                         (x<=datetime.strptime(endTimeStr,'%Y-%m-%d').date())]
    elif periodStr=='W':
      data=[x for x in daylist_localdata['week_data'] if (x>=datetime.strptime(startTimeStr,'%Y-%m-%d').date())&
                                                        (x<=datetime.strptime(endTimeStr,'%Y-%m-%d').date())]
    elif periodStr=='M':
      data=[x for x in daylist_localdata['month_data'] if (x>=datetime.strptime(startTimeStr,'%Y-%m-%d').date())&
                                                          (x<=datetime.strptime(endTimeStr,'%Y-%m-%d').date())]
    elif periodStr=='Y':
      data=[x for x in daylist_localdata['year_data'] if (x>=datetime.strptime(startTimeStr,'%Y-%m-%d').date())&
                                                          (x<=datetime.strptime(endTimeStr,'%Y-%m-%d').date())]
  elif type=='wind':
    data=w.tdays(startTimeStr,endTimeStr,'Period='+periodStr).Times
  return data

def update_daylist_localdata():
  nowtime=str(datetime.now().date())
  date_data=getdaylist('2000-1-1',nowtime,periodStr='D',type='wind')
  week_data=getdaylist('2000-1-1',nowtime,periodStr='W',type='wind')
  month_data=getdaylist('2000-1-1',nowtime,periodStr='M',type='wind')
  year_data=getdaylist('2000-1-1',nowtime,periodStr='Y',type='wind')
  del_files('daylist_localdata.pickle')
  picklesave('daylist_localdata.pickle',{'date_data':date_data,
                                          'week_data':week_data,
                                          'month_data':month_data,
                                          'year_data':year_data,})