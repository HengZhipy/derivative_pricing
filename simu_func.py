import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class random_process(object):
  def __init__(self,name,totaltime,timepoint_num,trajectory_num):
    self.name=name
    self.totaltime=totaltime
    self.timepoint_num=timepoint_num
    self.trajectory_num=trajectory_num

class brownian_motion(random_process):
  def __init__(self,name,totaltime,timepoint_num,trajectory_num):
    random_process.__init__(self,name,totaltime,timepoint_num,trajectory_num)
    noise_matrix=np.random.normal(loc=0,scale=1,size=(timepoint_num,trajectory_num))# 每行为固定时点上的1个截面，每列为固定个体的1个时间序列
    dt=totaltime/timepoint_num
    self.process_matrix=np.row_stack((np.zeros((1,trajectory_num)),np.cumsum(noise_matrix,axis=0)))*np.sqrt(dt)
    self.time_line=np.arange(0,totaltime+dt,dt)
    self.trajectory_line=np.arange(0,trajectory_num,1,dtype=np.int32)

  def trajactory_fig(self,which_traj='all'):
    if str(which_traj)=='all':
      fig=plt.figure()
      ax=fig.add_subplot(1,1,1)
      fig_dataframe=pd.DataFrame(self.process_matrix,index=self.time_line,columns=self.trajectory_line)
      fig_dataframe.plot(ax=ax, title=self.name+'_all_trajectories')
      ax.legend_.remove()
    else:
      x=self.time_line
      y=self.process_matrix[:,which_traj]
      plt.figure()
      plt.plot(x,y,linestyle='-',color='red')
      plt.title(self.name+'_trajectory='+str(which_traj))

class geometric_brownian_motion(random_process):
  def __init__(self,name,totaltime,timepoint_num,trajectory_num,drift=0,diffusion=1,start_value=1):
    '''
    如模拟风险中性测度下的股票价格，drift=r-0.5*(sigma**2)，diffusion=sigma，start_value=S0
    '''
    random_process.__init__(self,name,totaltime,timepoint_num,trajectory_num)
    dt=totaltime/timepoint_num
    noise_matrix=np.random.normal(loc=drift*dt,scale=diffusion*np.sqrt(dt),size=(timepoint_num,trajectory_num))# 每行为固定时点上的1个截面，每列为固定个体的1个时间序列
    self.process_matrix=start_value*np.exp(np.row_stack((np.zeros((1,trajectory_num)),np.cumsum(noise_matrix,axis=0))))
    self.time_line=np.arange(0,totaltime+dt,dt)
    self.trajectory_line=np.arange(0,trajectory_num,1,dtype=np.int32)

  def trajactory_fig(self,which_traj='all'):
    if str(which_traj)=='all':
      fig=plt.figure()
      ax=fig.add_subplot(1,1,1)
      fig_dataframe=pd.DataFrame(self.process_matrix,index=self.time_line,columns=self.trajectory_line)
      fig_dataframe.plot(ax=ax, title=self.name+'_all_trajectories')
      ax.legend_.remove()
    else:
      x=self.time_line
      y=self.process_matrix[:,which_traj]
      plt.figure()
      plt.plot(x,y,linestyle='-',color='red')
      plt.title(self.name+'_trajectory='+str(which_traj))


class binary_tree(object):
  def __init__(self,name,totaltime,timepoint_num,asset_para):
    self.name=name
    self.totaltime=totaltime
    self.timepoint_num=timepoint_num
    self.S0, self.u, self.d, self.p=asset_para[0], asset_para[1], asset_para[2], asset_para[3]
    vec_up_down=np.array([[self.u**i,self.d**i] for i in range(timepoint_num+1)])
    self.tree_matrix=self.S0*vec_up_down[:,1].reshape((-1,1))*vec_up_down[:,0].reshape((1,-1))
    for i in range(timepoint_num+1):
      self.tree_matrix[i,:]=np.roll(self.tree_matrix[i,:],i)
    self.tree_matrix=np.triu(self.tree_matrix)
  def tree_fig(self):
    pass
