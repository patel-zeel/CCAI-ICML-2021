
import GPy
from polire.placement.base import Base
from NSGPy.NumPy import LLS
import numpy as np
import pandas as pd
import sys
import os
np.random.seed(0)

model_name = sys.argv[0].split('/')[-1].replace('.py','')
path = sys.argv[1]
fold = sys.argv[2]
f_id = sys.argv[3]

trn_X = np.load(path+'data10/fold_'+fold+'/train/X/'+f_id+'.npz')['arr_0']
trn_y = np.load(path+'data10/fold_'+fold+'/train/y/'+f_id+'.npz')['arr_0']
tst_X = np.load(path+'data10/fold_'+fold+'/test/X/'+f_id+'.npz')['arr_0']
emp_cov = np.load(path+'data10/fold_'+fold+'/train/X/'+f_id+'emp_cov.npz')['arr_0']
mean_y = trn_y.mean()

scaler = pd.read_pickle(path+'data10/fold_'+fold+'/scaler/'+f_id+'.pickle')

# m = GPy.models.GPRegression(trn_X, trn_y-mean_y, GPy.kern.RBF(trn_X.shape[1], ARD=True))
# m.optimize_restarts(3)
# K = m.kern.K(trn_X)

n = 4
# greedy = Base(verbose=False)
# greedy.cov_np = K
# inds, _ = greedy.place(trn_X, N=n)

model = LLS(trn_X.shape[1], N_l_bar=n, optimizer='lsq')#, N_l_bar_method='greedy')
model.fit(trn_X, trn_y-mean_y, n_restarts=10, cov=emp_cov) #, near_opt_inds=inds)

if not os.path.exists(path+'data10/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data10/results/'+model_name+'/fold_'+fold+'/')
#         pd.to_pickle(model.params, path+'data10/results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')

pred_y = scaler.inverse_transform(model.predict(tst_X)[0] + mean_y)

if not os.path.exists(path+'data10/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data10/results/'+model_name+'/fold_'+fold+'/')

np.savez_compressed(path+'data10/results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
