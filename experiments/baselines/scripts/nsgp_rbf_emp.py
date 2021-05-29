
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

n = 4
greedy = Base(verbose=False)
greedy.cov_np = emp_cov
inds, _ = greedy.place(trn_X, N=n)

model = LLS(trn_X.shape[1], N_l_bar=n, N_l_bar_method='greedy')
model.fit(trn_X, trn_y-mean_y, n_restarts=10, near_opt_inds=inds)

if not os.path.exists(path+'data10/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data10/results/'+model_name+'/fold_'+fold+'/')
#         pd.to_pickle(model.params, path+'data10/results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')

pred_y, var_y = model.predict(tst_X)

pred_y = scaler.inverse_transform(pred_y) + mean_y
var_y = var_y * np.var(scaler.inverse_transform(trn_y))

if not os.path.exists(path+'data10/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data10/results/'+model_name+'/fold_'+fold+'/')

np.savez_compressed(path+'data10/results/'+model_name+'/fold_'+fold+'/'+f_id+'_var.npz', pred_y)
np.savez_compressed(path+'data10/results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
