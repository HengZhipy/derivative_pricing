import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import simu_func as sifu
from scipy import interpolate

### 通用函数
# 插值
def interpolate_price(x, y, s):
    f = interpolate.interp1d(x, y, kind='cubic')
    return float(f(s))
# 绘图
def fig_anyfunc(func,xlim_min,xlim_max,granularity,xlabel='x',ylabel='y',title='func_plot'):
  x=np.arange(xlim_min,xlim_max+granularity,granularity)
  y=np.array([func(xi) for xi in x])
  plt.figure()
  plt.plot(x,y, linestyle='-',color='red')
  plt.xlim((xlim_min,xlim_max))
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)

### 数字期权（行权时回报为1元的二元期权）与股票数字期权
def bs_digital_option_pricing(S0,K,T,sigma,r,type='call',digitaltype='cash',q=0):
  # S0和K同一单位，T按年计算，sigma与r均按不加百分号的原始数值计算
  d1=(np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
  d2=d1-(sigma*np.sqrt(T))
  if type=='call':
    if digitaltype=='cash':
      price=K*np.exp(-r*T)*norm.cdf(d2)
    elif digitaltype=='share':
      price=S0*np.exp(-q*T)*norm.cdf(d1)
  elif type=='put':
    if digitaltype=='cash':
      price=K*np.exp(-r*T)*norm.cdf(-d2)
    elif digitaltype=='share':
      price=S0*np.exp(-q*T)*norm.cdf(-d1)
  return price

### 欧式期权
def bs_european_option_pricing(S0,K,T,sigma,r,type='call',q=0):
  # S0和K同一单位，T按年计算，sigma与r均按不加百分号的原始数值计算
  if type=='call':
    share_digital_price=bs_digital_option_pricing(S0,K,T,sigma,r,type='call',digitaltype='share',q=q)
    digital_price=bs_digital_option_pricing(S0,K,T,sigma,r,type='call',digitaltype='cash',q=q)
    price=share_digital_price-digital_price
  elif type=='put':
    share_digital_price=bs_digital_option_pricing(S0,K,T,sigma,r,type='put',digitaltype='share',q=q)
    digital_price=bs_digital_option_pricing(S0,K,T,sigma,r,type='put',digitaltype='cash',q=q)
    price=-share_digital_price+digital_price
  return price

def bs_european_option_greeks(S0,K,T,sigma,r,type='call',greekstype='delta',q=0):
  # S0和K同一单位，T按年计算，sigma与r均按不加百分号的原始数值计算
  d1=(np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
  share_digital_price=bs_digital_option_pricing(S0,K,T,sigma,r,type='call',digitaltype='share',q=q)
  digital_price=bs_digital_option_pricing(S0,K,T,sigma,r,type='call',digitaltype='cash',q=q)
  if type=='call':
    if greekstype=='delta':
      greeks_value=np.exp(-q*T)*norm.cdf(d1)
    elif greekstype=='gamma':
      greeks_value=np.exp(-q*T)*norm.pdf(d1)/(S0*sigma*np.sqrt(T))
    elif greekstype=='theta':
      greeks_value=-np.exp(-q*T)*S0*norm.pdf(d1)*sigma/(2*np.sqrt(T))+q*share_digital_price-r*digital_price
    elif greekstype=='vega':
      greeks_value=np.exp(-q*T)*S0*norm.pdf(d1)*np.sqrt(T)
    elif greekstype=='rho':
      greeks_value=T*digital_price
  elif type=='put':
    if greekstype=='delta':
      greeks_value=np.exp(-q*T)*norm.cdf(d1)-np.exp(-q*T)
    elif greekstype=='gamma':
      greeks_value=np.exp(-q*T)*norm.pdf(d1)/(S0*sigma*np.sqrt(T))
    elif greekstype=='theta':
      greeks_value=-np.exp(-q*T)*S0*norm.pdf(d1)*sigma/(2*np.sqrt(T))+q*share_digital_price-r*digital_price+r*np.exp(-r*T)*K-q*np.exp(-q*T)*S0
    elif greekstype=='vega':
      greeks_value=np.exp(-q*T)*S0*norm.pdf(d1)*np.sqrt(T)
    elif greekstype=='rho':
      greeks_value=T*digital_price-T*np.exp(-r*T)*K
  return greeks_value

def bs_european_option_impvola(price,S0,K,T,r,type='call',q=0):
  impvola=fsolve(lambda impvola: bs_european_option_pricing(sigma=impvola,S0=S0,K=K,T=T,r=r,type=type,q=q)-price,[0.1])
  return impvola[0]

def fd_european_option_pricing(S0,K,T,sigma,r,type='call',fdtype='explicit',q=0,timepoint_num=0,pricepoint_num=20):
  # 有限差分（finite difference）方法计算欧式期权价格
  # 显式, 隐式, Crank-Nicholson：fdtype='explicit','implicit','Crank-Nicholson'
  # S0和K同一单位，T按年计算，sigma与r均按不加百分号的原始数值计算
  # 时间区间、价格区间划分
  if timepoint_num==0:
    timepoint_num=int(np.floor(1/((sigma**2)*T))) #确保显式方法收敛
  # timepoint_num=10
  delta_t=T/timepoint_num
  Smax=2*K
  delta_S=Smax/pricepoint_num
  if type=='call':
    # 边界条件1：t=T
    cond_t_end=np.array([max(j*delta_S-K,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
    # 边界条件2：S->inf
    cond_S_max=np.array([Smax-K for i in range(timepoint_num+1)]).reshape(1,-1)
    # 边界条件3：S=0
    cond_S_0=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
    if fdtype=='explicit':
      value_mat=cond_t_end
      backward_coef_mat=np.zeros((pricepoint_num-1,pricepoint_num+1))
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-1-j
        col_num1, col_num2, col_num3=row_num, row_num+1, row_num+2
        backward_coef_mat[row_num,col_num1]=(0.5*(r-q)*j+0.5*(sigma**2)*(j**2))*delta_t/(1+r*delta_t)
        backward_coef_mat[row_num,col_num2]=(1-(sigma**2)*(j**2)*delta_t)/(1+r*delta_t)
        backward_coef_mat[row_num,col_num3]=(-0.5*(r-q)*j+0.5*(sigma**2)*(j**2))*delta_t/(1+r*delta_t)
      for i in range(timepoint_num-1,-1,-1):
        value_vec=backward_coef_mat@np.vstack((np.array([cond_S_max[0,i+1]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i+1]])))
        value_mat=np.hstack((value_vec,value_mat))
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
    if fdtype=='implicit':
      value_mat=cond_t_end
      backward_coef_mat=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat[0,0]=1
      backward_coef_mat[pricepoint_num,pricepoint_num]=1
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-j
        col_num1, col_num2, col_num3=row_num-1, row_num, row_num+1
        backward_coef_mat[row_num,col_num1]=(-0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
        backward_coef_mat[row_num,col_num2]=1+(sigma**2)*(j**2)*delta_t+r*delta_t
        backward_coef_mat[row_num,col_num3]=(0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
      for i in range(timepoint_num-1,-1,-1):
        value_vec=np.linalg.inv(backward_coef_mat)@np.vstack((np.array([cond_S_max[0,i]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i]])))
        value_vec=value_vec[1:-1]
        value_mat=np.hstack((value_vec,value_mat))
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
    if fdtype=='crank_nicholson':
      value_mat=cond_t_end
      backward_coef_mat_now=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat_now[0,0]=1
      backward_coef_mat_now[pricepoint_num,pricepoint_num]=1
      backward_coef_mat_next=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat_next[0,0]=1
      backward_coef_mat_next[pricepoint_num,pricepoint_num]=1
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-j
        col_num1, col_num2, col_num3=row_num-1, row_num, row_num+1
        backward_coef_mat_now[row_num,col_num1]=(0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_now[row_num,col_num2]=-1-0.5*(sigma**2)*(j**2)*delta_t-0.5*r*delta_t
        backward_coef_mat_now[row_num,col_num3]=(-0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_next[row_num,col_num1]=-(0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_next[row_num,col_num2]=-(1-0.5*(sigma**2)*(j**2)*delta_t-0.5*r*delta_t)
        backward_coef_mat_next[row_num,col_num3]=-(-0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
      for i in range(timepoint_num-1,-1,-1):
        value_vec=backward_coef_mat_next@np.vstack((np.array([cond_S_max[0,i+1]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i+1]])))
        value_vec[0,0]=value_vec[0,0]+cond_S_max[0,i]-cond_S_max[0,i+1]
        value_vec[pricepoint_num,0]=value_vec[pricepoint_num,0]+cond_S_0[0,i]-cond_S_0[0,i+1]
        value_vec=np.linalg.inv(backward_coef_mat_now)@value_vec
        value_vec=value_vec[1:-1]
        value_mat=np.hstack((value_vec,value_mat))
      backward_coef_mat=(backward_coef_mat_now,backward_coef_mat_next)
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
  elif type=='put':
    # 边界条件1：t=T
    cond_t_end=np.array([max(K-j*delta_S,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
    # 边界条件2：S=0
    cond_S_0=np.array([K for i in range(timepoint_num+1)]).reshape(1,-1)
    # 边界条件3：S->inf
    cond_S_max=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
    if fdtype=='explicit':
      value_mat=cond_t_end
      backward_coef_mat=np.zeros((pricepoint_num-1,pricepoint_num+1))
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-1-j
        col_num1, col_num2, col_num3=row_num, row_num+1, row_num+2
        backward_coef_mat[row_num,col_num1]=(0.5*(r-q)*j+0.5*(sigma**2)*(j**2))*delta_t/(1+r*delta_t)
        backward_coef_mat[row_num,col_num2]=(1-(sigma**2)*(j**2)*delta_t)/(1+r*delta_t)
        backward_coef_mat[row_num,col_num3]=(-0.5*(r-q)*j+0.5*(sigma**2)*(j**2))*delta_t/(1+r*delta_t)
      for i in range(timepoint_num-1,-1,-1):
        value_vec=backward_coef_mat@np.vstack((np.array([cond_S_max[0,i+1]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i+1]])))
        value_mat=np.hstack((value_vec,value_mat))
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
    if fdtype=='implicit':
      value_mat=cond_t_end
      backward_coef_mat=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat[0,0]=1
      backward_coef_mat[pricepoint_num,pricepoint_num]=1
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-j
        col_num1, col_num2, col_num3=row_num-1, row_num, row_num+1
        backward_coef_mat[row_num,col_num1]=(-0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
        backward_coef_mat[row_num,col_num2]=1+(sigma**2)*(j**2)*delta_t+r*delta_t
        backward_coef_mat[row_num,col_num3]=(0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
      for i in range(timepoint_num-1,-1,-1):
        value_vec=np.linalg.inv(backward_coef_mat)@np.vstack((np.array([cond_S_max[0,i]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i]])))
        value_vec=value_vec[1:-1]
        value_mat=np.hstack((value_vec,value_mat))
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
    if fdtype=='crank_nicholson':
      value_mat=cond_t_end
      backward_coef_mat_now=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat_now[0,0]=1
      backward_coef_mat_now[pricepoint_num,pricepoint_num]=1
      backward_coef_mat_next=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat_next[0,0]=1
      backward_coef_mat_next[pricepoint_num,pricepoint_num]=1
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-j
        col_num1, col_num2, col_num3=row_num-1, row_num, row_num+1
        backward_coef_mat_now[row_num,col_num1]=(0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_now[row_num,col_num2]=-1-0.5*(sigma**2)*(j**2)*delta_t-0.5*r*delta_t
        backward_coef_mat_now[row_num,col_num3]=(-0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_next[row_num,col_num1]=-(0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_next[row_num,col_num2]=-(1-0.5*(sigma**2)*(j**2)*delta_t-0.5*r*delta_t)
        backward_coef_mat_next[row_num,col_num3]=-(-0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
      for i in range(timepoint_num-1,-1,-1):
        value_vec=backward_coef_mat_next@np.vstack((np.array([cond_S_max[0,i+1]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i+1]])))
        value_vec[0,0]=value_vec[0,0]+cond_S_max[0,i]-cond_S_max[0,i+1]
        value_vec[pricepoint_num,0]=value_vec[pricepoint_num,0]+cond_S_0[0,i]-cond_S_0[0,i+1]
        value_vec=np.linalg.inv(backward_coef_mat_now)@value_vec
        value_vec=value_vec[1:-1]
        value_mat=np.hstack((value_vec,value_mat))
      backward_coef_mat=(backward_coef_mat_now,backward_coef_mat_next)
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
  return price,value_mat,backward_coef_mat

### 美式期权
def fd_american_option_pricing(S0,K,T,sigma,r,type='call',fdtype='explicit',q=0,timepoint_num=0,pricepoint_num=20):
  # 有限差分（finite difference）方法计算美式期权价格
  # 显式, 隐式, Crank-Nicholson：fdtype='explicit','implicit','Crank-Nicholson'
  # S0和K同一单位，T按年计算，sigma与r均按不加百分号的原始数值计算
  # 时间区间、价格区间划分
  if timepoint_num==0:
    timepoint_num=int(np.floor(1/((sigma**2)*T))) #确保显式方法收敛
  # timepoint_num=10
  delta_t=T/timepoint_num
  Smax=2*K
  delta_S=Smax/pricepoint_num
  if type=='call':
    # 边界条件1：t=T
    cond_t_end=np.array([max(j*delta_S-K,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
    # 边界条件2：S=0
    cond_S_0=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
    # 边界条件3：S->inf
    cond_S_max=np.array([K for i in range(timepoint_num+1)]).reshape(1,-1)
    if fdtype=='explicit':
      value_mat=cond_t_end
      backward_coef_mat=np.zeros((pricepoint_num-1,pricepoint_num+1))
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-1-j
        col_num1, col_num2, col_num3=row_num, row_num+1, row_num+2
        backward_coef_mat[row_num,col_num1]=(0.5*(r-q)*j+0.5*(sigma**2)*(j**2))*delta_t/(1+r*delta_t)
        backward_coef_mat[row_num,col_num2]=(1-(sigma**2)*(j**2)*delta_t)/(1+r*delta_t)
        backward_coef_mat[row_num,col_num3]=(-0.5*(r-q)*j+0.5*(sigma**2)*(j**2))*delta_t/(1+r*delta_t)
      for i in range(timepoint_num-1,-1,-1):
        value_vec=backward_coef_mat@np.vstack((np.array([cond_S_max[0,i+1]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i+1]])))
        intrinsic_value_vec=np.array([max(j*delta_S-K,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
        value_vec=np.max(np.hstack((value_vec,intrinsic_value_vec)),axis=1).reshape(-1,1)
        value_mat=np.hstack((value_vec,value_mat))
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
    if fdtype=='implicit':
      value_mat=cond_t_end
      backward_coef_mat=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat[0,0]=1
      backward_coef_mat[pricepoint_num,pricepoint_num]=1
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-j
        col_num1, col_num2, col_num3=row_num-1, row_num, row_num+1
        backward_coef_mat[row_num,col_num1]=(-0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
        backward_coef_mat[row_num,col_num2]=1+(sigma**2)*(j**2)*delta_t+r*delta_t
        backward_coef_mat[row_num,col_num3]=(0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
      for i in range(timepoint_num-1,-1,-1):
        value_vec=np.linalg.inv(backward_coef_mat)@np.vstack((np.array([cond_S_max[0,i]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i]])))
        value_vec=value_vec[1:-1]
        intrinsic_value_vec=np.array([max(j*delta_S-K,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
        value_vec=np.max(np.hstack((value_vec,intrinsic_value_vec)),axis=1).reshape(-1,1)
        value_mat=np.hstack((value_vec,value_mat))
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
    if fdtype=='crank_nicholson':
      value_mat=cond_t_end
      backward_coef_mat_now=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat_now[0,0]=1
      backward_coef_mat_now[pricepoint_num,pricepoint_num]=1
      backward_coef_mat_next=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat_next[0,0]=1
      backward_coef_mat_next[pricepoint_num,pricepoint_num]=1
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-j
        col_num1, col_num2, col_num3=row_num-1, row_num, row_num+1
        backward_coef_mat_now[row_num,col_num1]=(0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_now[row_num,col_num2]=-1-0.5*(sigma**2)*(j**2)*delta_t-0.5*r*delta_t
        backward_coef_mat_now[row_num,col_num3]=(-0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_next[row_num,col_num1]=-(0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_next[row_num,col_num2]=-(1-0.5*(sigma**2)*(j**2)*delta_t-0.5*r*delta_t)
        backward_coef_mat_next[row_num,col_num3]=-(-0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
      for i in range(timepoint_num-1,-1,-1):
        value_vec=backward_coef_mat_next@np.vstack((np.array([cond_S_max[0,i+1]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i+1]])))
        value_vec[0,0]=value_vec[0,0]+cond_S_max[0,i]-cond_S_max[0,i+1]
        value_vec[pricepoint_num,0]=value_vec[pricepoint_num,0]+cond_S_0[0,i]-cond_S_0[0,i+1]
        value_vec=np.linalg.inv(backward_coef_mat_now)@value_vec
        value_vec=value_vec[1:-1]
        intrinsic_value_vec=np.array([max(j*delta_S-K,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
        value_vec=np.max(np.hstack((value_vec,intrinsic_value_vec)),axis=1).reshape(-1,1)
        value_mat=np.hstack((value_vec,value_mat))
      backward_coef_mat=(backward_coef_mat_now,backward_coef_mat_next)
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
  elif type=='put':
    # 边界条件1：t=T
    cond_t_end=np.array([max(K-j*delta_S,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
    # 边界条件2：S=0
    cond_S_0=np.array([K for i in range(timepoint_num+1)]).reshape(1,-1)
    # 边界条件3：S->inf
    cond_S_max=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
    if fdtype=='explicit':
      value_mat=cond_t_end
      backward_coef_mat=np.zeros((pricepoint_num-1,pricepoint_num+1))
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-1-j
        col_num1, col_num2, col_num3=row_num, row_num+1, row_num+2
        backward_coef_mat[row_num,col_num1]=(0.5*(r-q)*j+0.5*(sigma**2)*(j**2))*delta_t/(1+r*delta_t)
        backward_coef_mat[row_num,col_num2]=(1-(sigma**2)*(j**2)*delta_t)/(1+r*delta_t)
        backward_coef_mat[row_num,col_num3]=(-0.5*(r-q)*j+0.5*(sigma**2)*(j**2))*delta_t/(1+r*delta_t)
      for i in range(timepoint_num-1,-1,-1):
        value_vec=backward_coef_mat@np.vstack((np.array([cond_S_max[0,i+1]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i+1]])))
        intrinsic_value_vec=np.array([max(K-j*delta_S,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
        value_vec=np.max(np.hstack((value_vec,intrinsic_value_vec)),axis=1).reshape(-1,1)
        value_mat=np.hstack((value_vec,value_mat))
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
    if fdtype=='implicit':
      value_mat=cond_t_end
      backward_coef_mat=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat[0,0]=1
      backward_coef_mat[pricepoint_num,pricepoint_num]=1
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-j
        col_num1, col_num2, col_num3=row_num-1, row_num, row_num+1
        backward_coef_mat[row_num,col_num1]=(-0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
        backward_coef_mat[row_num,col_num2]=1+(sigma**2)*(j**2)*delta_t+r*delta_t
        backward_coef_mat[row_num,col_num3]=(0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
      for i in range(timepoint_num-1,-1,-1):
        value_vec=np.linalg.inv(backward_coef_mat)@np.vstack((np.array([cond_S_max[0,i]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i]])))
        value_vec=value_vec[1:-1]
        intrinsic_value_vec=np.array([max(K-j*delta_S,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
        value_vec=np.max(np.hstack((value_vec,intrinsic_value_vec)),axis=1).reshape(-1,1)
        value_mat=np.hstack((value_vec,value_mat))
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
    if fdtype=='crank_nicholson':
      value_mat=cond_t_end
      backward_coef_mat_now=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat_now[0,0]=1
      backward_coef_mat_now[pricepoint_num,pricepoint_num]=1
      backward_coef_mat_next=np.zeros((pricepoint_num+1,pricepoint_num+1))
      backward_coef_mat_next[0,0]=1
      backward_coef_mat_next[pricepoint_num,pricepoint_num]=1
      for j in range(pricepoint_num-1,0,-1):
        row_num=pricepoint_num-j
        col_num1, col_num2, col_num3=row_num-1, row_num, row_num+1
        backward_coef_mat_now[row_num,col_num1]=(0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_now[row_num,col_num2]=-1-0.5*(sigma**2)*(j**2)*delta_t-0.5*r*delta_t
        backward_coef_mat_now[row_num,col_num3]=(-0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_next[row_num,col_num1]=-(0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
        backward_coef_mat_next[row_num,col_num2]=-(1-0.5*(sigma**2)*(j**2)*delta_t-0.5*r*delta_t)
        backward_coef_mat_next[row_num,col_num3]=-(-0.25*(r-q)*j+0.25*(sigma**2)*(j**2))*delta_t
      for i in range(timepoint_num-1,-1,-1):
        value_vec=backward_coef_mat_next@np.vstack((np.array([cond_S_max[0,i+1]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i+1]])))
        value_vec[0,0]=value_vec[0,0]+cond_S_max[0,i]-cond_S_max[0,i+1]
        value_vec[pricepoint_num,0]=value_vec[pricepoint_num,0]+cond_S_0[0,i]-cond_S_0[0,i+1]
        value_vec=np.linalg.inv(backward_coef_mat_now)@value_vec
        value_vec=value_vec[1:-1]
        intrinsic_value_vec=np.array([max(K-j*delta_S,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
        value_vec=np.max(np.hstack((value_vec,intrinsic_value_vec)),axis=1).reshape(-1,1)
        value_mat=np.hstack((value_vec,value_mat))
      backward_coef_mat=(backward_coef_mat_now,backward_coef_mat_next)
      value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
      price=value_mat[pricepoint_num//2,0] if pricepoint_num%2==0 else 0.5*(value_mat[pricepoint_num//2,0]+value_mat[pricepoint_num//2+1,0])
  return price,value_mat,backward_coef_mat

### 雪球期权
KI_obs_day_list_=[x+1 for x in range(252)]
KO_obs_day_list_=[(x+1)*21 for x in range(12)]
def mc_snowball_option_pricing(S0,K,KI,KO,T,coupon,r,sigma,KI_obs_day_list=KI_obs_day_list_,KO_obs_day_list=KO_obs_day_list_,simu_n=500000):
  #假设名义本金为1
  S_process=sifu.geometric_brownian_motion('stock_index',1,252,simu_n,drift=r-0.5*(sigma**2),diffusion=sigma,start_value=S0)
  S_matrix=S_process.process_matrix
  payoff_discount_list=[]
  situation_list=[]
  for i in range(simu_n):
    S_traj=S_matrix[:,i]
    # 发生敲出
    if np.max(S_traj[KO_obs_day_list])>=KO:
      KO_day=KO_obs_day_list[np.argwhere(S_traj[KO_obs_day_list]>=KO)[0][0]]
      payoff=coupon*KO_day/252
      payoff_discount=payoff*np.exp(-r*KO_day/252)
      situation_list.append(0)
    # 从未敲出，且从未敲入
    elif np.min(S_traj[KI_obs_day_list])>KI:
      payoff=coupon*T
      payoff_discount=payoff*np.exp(-r*T)
      situation_list.append(1)
    # 从未敲出，且发生敲入
    else:
      payoff=np.min([S_traj[-1]/K-1,0])
      payoff_discount=payoff*np.exp(-r*T)
      situation_list.append(2)
    payoff_discount_list.append(payoff_discount)
  return payoff_discount_list,situation_list

KI_obs_day_list_=[x+1 for x in range(252)]
KO_obs_day_list_=[(x+1)*21 for x in range(12)]
obs_time_list_=[x+1 for x in range(252)]
def df_snowball_option_timesetting(t,T,obs_time_list=obs_time_list_,KI_obs_day_list=KI_obs_day_list_,KO_obs_day_list=KO_obs_day_list_,yearday=252):
  T_left=T-t
  day_left=int(np.floor(T_left*yearday))
  day_total=len(obs_time_list)
  day_passed=day_total-day_left
  obs_time_list_left=[x-day_passed for x in obs_time_list if x>=day_passed]
  KI_obs_day_list_left=[x-day_passed for x in KI_obs_day_list if x>=day_passed]
  KO_obs_day_list_left=[x-day_passed for x in KO_obs_day_list if x>=day_passed]
  return T_left, obs_time_list_left, KI_obs_day_list_left, KO_obs_day_list_left

def df_snowball_option_pricing(S0,K,KI,KO,T,coupon,r,sigma,t_start=0,q=0,obs_time_list=obs_time_list_,KI_obs_day_list=KI_obs_day_list_,KO_obs_day_list=KO_obs_day_list_,yearday=252,pricepoint_num=500):
  # 假设名义本金为1
  # 雪球=long向上敲出期权+long向上敲出向下敲出期权+short向上敲出看跌期权+long向上敲出向下敲出看跌期权
  # 均采用implicit方式解PDE
  # yearday在基于coupon（年化）计算敲出日应付数值时使用
  # T的单位为年，obs_time_list为其离散化列表，除起始时间（0日）外，每日为其1个元素，KI_obs_day_list为敲入观察日列表，KO_obs_day_list为敲出观察日列表
  T,obs_time_list,KI_obs_day_list,KO_obs_day_list=df_snowball_option_timesetting(t_start,T,obs_time_list,KI_obs_day_list,KO_obs_day_list,yearday)
  #-----
  ## 计算backward矩阵
  timepoint_num=len(obs_time_list)
  delta_t=T/timepoint_num #时间间隔设为定长
  Smax=2*K
  delta_S=Smax/pricepoint_num
  S_line=[delta_S * (pricepoint_num-j) for j in range(pricepoint_num+1)]
  backward_coef_mat=np.zeros((pricepoint_num+1,pricepoint_num+1))
  backward_coef_mat[0,0]=1
  backward_coef_mat[pricepoint_num,pricepoint_num]=1
  for j in range(pricepoint_num-1,0,-1):
    row_num=pricepoint_num-j
    col_num1, col_num2, col_num3=row_num-1, row_num, row_num+1
    backward_coef_mat[row_num,col_num1]=(-0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
    backward_coef_mat[row_num,col_num2]=1+(sigma**2)*(j**2)*delta_t+r*delta_t
    backward_coef_mat[row_num,col_num3]=(0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
  backward_coef_mat_inv=np.linalg.inv(backward_coef_mat)
  #-----
  ## 1. 向上敲出期权定价：敲出价KO，敲出获得coupon，否则获得0
  # 边界条件1：t=T
  cond_t_end=np.array([coupon if j*delta_S-KO>=0 else 0 for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
  # 边界条件2：S=0
  cond_S_0=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
  # 边界条件3：S->inf
  cond_S_max=np.array([coupon*i/yearday/T if i in KO_obs_day_list else 0 for i in range(timepoint_num+1)]).reshape(1,-1)
  value_mat=cond_t_end
  for i in range(timepoint_num-1,-1,-1):
    value_vec=backward_coef_mat_inv@np.vstack((np.array([cond_S_max[0,i]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i]])))
    if i in KO_obs_day_list:
      KO_slice=[pricepoint_num-j for j in range(pricepoint_num+1) if j*delta_S>=KO]
      value_vec[KO_slice]=coupon*i/yearday/T
    value_vec=value_vec[1:-1]
    value_mat=np.hstack((value_vec,value_mat))
  value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
  value_mat_autocall=value_mat
  price_autocall=interpolate_price(S_line, value_mat_autocall[:,0], S0)
  #-----
  ## 2. 向上敲出向下敲出期权定价：敲出价KO、KI，敲出获得0，否则获得coupon
  # 边界条件1：t=T
  cond_t_end=np.array([coupon if (j*delta_S-KO<0) & (j*delta_S-KI>0) else 0 for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
  # 边界条件2：S=0
  cond_S_0=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
  # 边界条件3：S->inf
  cond_S_max=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
  value_mat=cond_t_end
  for i in range(timepoint_num-1,-1,-1):
    value_vec=backward_coef_mat_inv@np.vstack((np.array([cond_S_max[0,i]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i]])))
    if i in KO_obs_day_list:
      KO_slice=[pricepoint_num-j for j in range(pricepoint_num+1) if j*delta_S>=KO]
      value_vec[KO_slice]=0
    if i in KI_obs_day_list:
      KI_slice=[pricepoint_num-j for j in range(pricepoint_num+1) if j*delta_S<=KI]
      value_vec[KI_slice]=0
    value_vec=value_vec[1:-1]
    value_mat=np.hstack((value_vec,value_mat))
  value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
  value_mat_KOKI=value_mat
  price_KOKI=interpolate_price(S_line, value_mat_KOKI[:,0], S0)
  #-----
  ## 3. 向上敲出看跌期权定价：敲出价KO，敲出获得0，否则获得一个执行价为K的看跌期权
  # 边界条件1：t=T
  cond_t_end=np.array([0 if j*delta_S-KO>=0 else max(K-j*delta_S,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
  # 边界条件2：S=0
  cond_S_0=np.array([K*np.exp(-r*(T-i*delta_t)) for i in range(timepoint_num+1)]).reshape(1,-1)
  # 边界条件3：S->inf
  cond_S_max=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
  value_mat=cond_t_end
  for i in range(timepoint_num-1,-1,-1):
    value_vec=backward_coef_mat_inv@np.vstack((np.array([cond_S_max[0,i]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i]])))
    if i in KO_obs_day_list:
      KO_slice=[pricepoint_num-j for j in range(pricepoint_num+1) if j*delta_S>=KO]
      value_vec[KO_slice]=0
    value_vec=value_vec[1:-1]
    value_mat=np.hstack((value_vec,value_mat))
  value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
  value_mat_put_KO=value_mat
  price_put_KO=interpolate_price(S_line, value_mat_put_KO[:,0], S0)
  #-----
  ## 4. 向上敲出向下敲出看跌期权定价
  # 边界条件1：t=T
  cond_t_end=np.array([0 if (j*delta_S-KO>=0)&(j*delta_S-KI<=0) else max(K-j*delta_S,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
  # 边界条件2：S=0
  cond_S_0=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
  # 边界条件3：S->inf
  cond_S_max=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
  value_mat=cond_t_end
  for i in range(timepoint_num-1,-1,-1):
    value_vec=backward_coef_mat_inv@np.vstack((np.array([cond_S_max[0,i]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i]])))
    if i in KO_obs_day_list:
      KO_slice=[pricepoint_num-j for j in range(pricepoint_num+1) if j*delta_S>=KO]
      value_vec[KO_slice]=0
    if i in KI_obs_day_list:
      KI_slice=[pricepoint_num-j for j in range(pricepoint_num+1) if j*delta_S<=KI]
      value_vec[KI_slice]=0
    value_vec=value_vec[1:-1]
    value_mat=np.hstack((value_vec,value_mat))
  value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
  value_mat_put_KOKI=value_mat
  # price_put_KOKI=interpolate_price(S_line, value_mat_put_KOKI[:,0], S0)
  # 期权价格合成与希腊字母
  value_mat_comb=value_mat_autocall+value_mat_KOKI-value_mat_put_KO+value_mat_put_KOKI
  price=interpolate_price(S_line, value_mat_comb[:,0], S0)
  return price, (value_mat_comb, S_line, delta_t, value_mat_autocall,value_mat_KOKI,value_mat_put_KO,value_mat_put_KOKI)

def df_snowball_option_greeks(S0,K,KI,KO,T,coupon,r,sigma,t_start=0,q=0,obs_time_list=obs_time_list_,KI_obs_day_list=KI_obs_day_list_,KO_obs_day_list=KO_obs_day_list_,yearday=252,pricepoint_num=500):
  p0,greek_tuple = df_snowball_option_pricing(S0,K,KI,KO,T,coupon,r,sigma,t_start=t_start,q=q,obs_time_list=obs_time_list,KI_obs_day_list=KI_obs_day_list,KO_obs_day_list=KO_obs_day_list,yearday=yearday,pricepoint_num=pricepoint_num)
  epsilon = 0.01
  p_delta_bias_pos = interpolate_price(greek_tuple[1],greek_tuple[0][:,0], S0 * (1 + epsilon))
  p_delta_bias_neg = interpolate_price(greek_tuple[1],greek_tuple[0][:,0], S0 * (1 - epsilon))
  delta = (p_delta_bias_pos - p_delta_bias_neg) / (2 *S0 *epsilon)
  gamma = (p_delta_bias_pos + p_delta_bias_neg - 2 * p0) / ((S0 **2) * (epsilon**2))
  theta = r*p0-delta*r*S0-0.5*(sigma**2)*(S0**2)*gamma
  epsilon_v = 0.001
  sigma_bias = sigma + epsilon_v
  p_vega_bias,_ = df_snowball_option_pricing(S0,K,KI,KO,T,coupon,r,sigma_bias,t_start=t_start,q=q,obs_time_list=obs_time_list,KI_obs_day_list=KI_obs_day_list,KO_obs_day_list=KO_obs_day_list,yearday=yearday,pricepoint_num=pricepoint_num)
  vega = (p_vega_bias - p0) / epsilon_v
  epsilon_r = 0.0005
  r_bias = r + epsilon_r
  p_rho_bias,_ = df_snowball_option_pricing(S0,K,KI,KO,T,coupon,r_bias,sigma,t_start=t_start,q=q,obs_time_list=obs_time_list,KI_obs_day_list=KI_obs_day_list,KO_obs_day_list=KO_obs_day_list,yearday=yearday,pricepoint_num=pricepoint_num)
  rho = (p_rho_bias - p0) / epsilon_r
  return delta, gamma, vega, theta, rho, p0

def df_snowball_option_knockedin_pricing(S0,K,KI,KO,T,coupon,r,sigma,q=0,obs_time_list=obs_time_list_,KI_obs_day_list=KI_obs_day_list_,KO_obs_day_list=KO_obs_day_list_,yearday=252,pricepoint_num=500):
  # 假设名义本金为1
  # 雪球=long向上敲出期权+long向上敲出向下敲出期权+short向上敲出看跌期权+long向上敲出向下敲出看跌期权
  # 均采用implicit方式解PDE
  # yearday在基于coupon（年化）计算敲出日应付数值时使用
  # T的单位为年，obs_time_list为其离散化列表，除起始时间（0日）外，每日为其1个元素，KI_obs_day_list为敲入观察日列表，KO_obs_day_list为敲出观察日列表
  #-----
  ## 计算backward矩阵
  timepoint_num=len(obs_time_list)
  delta_t=T/timepoint_num #时间间隔设为定长
  Smax=2*K
  delta_S=Smax/pricepoint_num
  S_line=[delta_S * (pricepoint_num-j) for j in range(pricepoint_num+1)]
  backward_coef_mat=np.zeros((pricepoint_num+1,pricepoint_num+1))
  backward_coef_mat[0,0]=1
  backward_coef_mat[pricepoint_num,pricepoint_num]=1
  for j in range(pricepoint_num-1,0,-1):
    row_num=pricepoint_num-j
    col_num1, col_num2, col_num3=row_num-1, row_num, row_num+1
    backward_coef_mat[row_num,col_num1]=(-0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
    backward_coef_mat[row_num,col_num2]=1+(sigma**2)*(j**2)*delta_t+r*delta_t
    backward_coef_mat[row_num,col_num3]=(0.5*(r-q)*j-0.5*(sigma**2)*(j**2))*delta_t
  backward_coef_mat_inv=np.linalg.inv(backward_coef_mat)
  #-----
  ## 1. 向上敲出期权定价：敲出价KO，敲出获得coupon，否则获得0
  # 边界条件1：t=T
  cond_t_end=np.array([coupon if j*delta_S-KO>=0 else 0 for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
  # 边界条件2：S=0
  cond_S_0=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
  # 边界条件3：S->inf
  cond_S_max=np.array([coupon*i/yearday/T if i in KO_obs_day_list else 0 for i in range(timepoint_num+1)]).reshape(1,-1)
  value_mat=cond_t_end
  for i in range(timepoint_num-1,-1,-1):
    value_vec=backward_coef_mat_inv@np.vstack((np.array([cond_S_max[0,i]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i]])))
    if i in KO_obs_day_list:
      KO_slice=[pricepoint_num-j for j in range(pricepoint_num+1) if j*delta_S>=KO]
      value_vec[KO_slice]=coupon*i/yearday/T
    value_vec=value_vec[1:-1]
    value_mat=np.hstack((value_vec,value_mat))
  value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
  value_mat_autocall=value_mat
  price_autocall=interpolate_price(S_line, value_mat_autocall[:,0], S0)
  #-----
  ## 3. 向上敲出看跌期权定价：敲出价KO，敲出获得0，否则获得一个执行价为K的看跌期权
  # 边界条件1：t=T
  cond_t_end=np.array([0 if j*delta_S-KO>=0 else max(K-j*delta_S,0) for j in range(pricepoint_num-1,0,-1)]).reshape(-1,1)
  # 边界条件2：S=0
  cond_S_0=np.array([K*np.exp(-r*(T-i*delta_t)) for i in range(timepoint_num+1)]).reshape(1,-1)
  # 边界条件3：S->inf
  cond_S_max=np.array([0 for i in range(timepoint_num+1)]).reshape(1,-1)
  value_mat=cond_t_end
  for i in range(timepoint_num-1,-1,-1):
    value_vec=backward_coef_mat_inv@np.vstack((np.array([cond_S_max[0,i]]),value_mat[:,0].reshape(-1,1),np.array([cond_S_0[0,i]])))
    if i in KO_obs_day_list:
      KO_slice=[pricepoint_num-j for j in range(pricepoint_num+1) if j*delta_S>=KO]
      value_vec[KO_slice]=0
    value_vec=value_vec[1:-1]
    value_mat=np.hstack((value_vec,value_mat))
  value_mat=np.vstack((cond_S_max,value_mat,cond_S_0))
  value_mat_put_KO=value_mat
  price_put_KO=interpolate_price(S_line, value_mat_put_KO[:,0], S0)
  price=price_autocall-price_put_KO
  value_mat_comb=value_mat_autocall-value_mat_put_KO
  return price, (value_mat_comb, S_line, delta_t)

def df_snowball_option_knockedin_greeks(S0,K,KI,KO,T,coupon,r,sigma,q=0,obs_time_list=obs_time_list_,KI_obs_day_list=KI_obs_day_list_,KO_obs_day_list=KO_obs_day_list_,yearday=252):
  p0,greek_tuple = df_snowball_option_knockedin_pricing(S0,K,KI,KO,T,coupon,r,sigma,q=q,obs_time_list=obs_time_list,KI_obs_day_list=KI_obs_day_list,KO_obs_day_list=KO_obs_day_list,yearday=yearday)
  epsilon = 0.01
  p_delta_bias_pos = interpolate_price(greek_tuple[1],greek_tuple[0][:,0], S0 * (1 + epsilon))
  p_delta_bias_neg = interpolate_price(greek_tuple[1],greek_tuple[0][:,0], S0 * (1 - epsilon))
  delta = (p_delta_bias_pos - p_delta_bias_neg) / (2 *S0 *epsilon)
  gamma = (p_delta_bias_pos + p_delta_bias_neg - 2 * p0) / ((S0 **2) * (epsilon**2))
  theta = r*p0-delta*r*S0-0.5*(sigma**2)*(S0**2)*gamma
  epsilon_v = 0.001
  sigma_bias = sigma + epsilon_v
  p_vega_bias,_ = df_snowball_option_knockedin_pricing(S0,K,KI,KO,T,coupon,r,sigma_bias,q=q,obs_time_list=obs_time_list,KI_obs_day_list=KI_obs_day_list,KO_obs_day_list=KO_obs_day_list,yearday=yearday)
  vega = (p_vega_bias - p0) / epsilon_v
  epsilon_r = 0.0005
  r_bias = r + epsilon_r
  p_rho_bias,_ = df_snowball_option_knockedin_pricing(S0,K,KI,KO,T,coupon,r_bias,sigma,q=q,obs_time_list=obs_time_list,KI_obs_day_list=KI_obs_day_list,KO_obs_day_list=KO_obs_day_list,yearday=yearday)
  rho = (p_rho_bias - p0) / epsilon_r
  return delta, gamma, vega, theta, rho, p0

def mc_asia_option_pricing(S0,K,T,r,sigma,simu_n=500000,asia_type='aveprice',opt_type='call'):
  #假设名义本金为1
  #行权方式为欧式
  S_process=sifu.geometric_brownian_motion('stock_index',1,20,simu_n,drift=r-0.5*(sigma**2),diffusion=sigma,start_value=S0)
  S_matrix=S_process.process_matrix
  payoff_discount_list=[]
  if asia_type=='avestrike':
    #平均执行价格亚式期权asia_type='avstrike'
    if opt_type=='call':
      for i in range(simu_n):
        S_traj=S_matrix[:,i]
        strike_ave=np.mean(S_traj)
        payoff_discount=max(S_traj[-1]-strike_ave,0)*np.exp(-r*T)
        payoff_discount_list.append(payoff_discount)
    elif opt_type=='put':
      for i in range(simu_n):
        S_traj=S_matrix[:,i]
        strike_ave=np.mean(S_traj)
        payoff_discount=max(strike_ave-S_traj[-1],0)*np.exp(-r*T)
        payoff_discount_list.append(payoff_discount)
  elif asia_type=='aveprice':
    #平均价格亚式期权asia_type='aveprice'(default)
    if opt_type=='call':
      for i in range(simu_n):
        S_traj=S_matrix[:,i]
        aveprice=np.mean(S_traj)
        payoff_discount=max(aveprice-K,0)*np.exp(-r*T)
        payoff_discount_list.append(payoff_discount)
    elif opt_type=='put':
      for i in range(simu_n):
        S_traj=S_matrix[:,i]
        aveprice=np.mean(S_traj)
        payoff_discount=max(K-aveprice,0)*np.exp(-r*T)
        payoff_discount_list.append(payoff_discount)
  return np.mean(payoff_discount_list), payoff_discount_list

if __name__ == '__main__':
  print('derivative functions')