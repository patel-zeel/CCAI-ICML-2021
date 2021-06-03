
# import GPy
# from polire.placement.base import Base
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

# if os.path.exists(path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz'):
#     sys.exit()

trn_X = np.load(path+'data2/fold_'+fold+'/train/X/'+f_id+'.npz')['arr_0']
trn_y = np.load(path+'data2/fold_'+fold+'/train/y/'+f_id+'.npz')['arr_0']
tst_X = np.load(path+'data2/fold_'+fold+'/test/X/'+f_id+'.npz')['arr_0']
# emp_cov = np.load(path+'data2/fold_'+fold+'/train/X/'+f_id+'emp_cov.npz')['arr_0']
mean_y = trn_y.mean()

scaler = pd.read_pickle(path+'data2/fold_'+fold+'/scaler/'+f_id+'.pickle')

# m = GPy.models.GPRegression(trn_X, trn_y-mean_y, GPy.kern.RBF(trn_X.shape[1], ARD=True))
# m.optimize_restarts(3)
# K = m.kern.K(trn_X)

n = 5
# greedy = Base(verbose=False)
# greedy.cov_np = K
# inds, _ = greedy.place(trn_X, N=n)

best_model = None
best_loss = np.inf
for i in range(3,7):
    try:
        model = LLS(trn_X.shape[1], N_l_bar=i)
        model.fit(trn_X, trn_y-mean_y, n_restarts=1)
        if model.params['likelihood (mll)']<best_loss:
            best_loss = model.params['likelihood (mll)']
            best_model = model
    except:
        pass
if not os.path.exists(path+'data2/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data2/results/'+model_name+'/fold_'+fold+'/')
#         pd.to_pickle(model.params, path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')

pred_y, var_y = best_model.predict(tst_X)

pred_y = scaler.inverse_transform(pred_y + mean_y)
var_y = var_y * np.var(scaler.inverse_transform(trn_y))

if not os.path.exists(path+'data2/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data2/results/'+model_name+'/fold_'+fold+'/')

np.savez_compressed(path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'_var.npz', pred_y)
np.savez_compressed(path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
pd.to_pickle(model, path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
