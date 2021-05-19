# from stheno import Measure, GP, EQ, Delta
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
N_l_bar_max = 8

trn_X = np.load(path+'data/fold_'+fold+'/train/X/'+f_id+'.npz')['arr_0']
trn_y = np.load(path+'data/fold_'+fold+'/train/y/'+f_id+'.npz')['arr_0']
tst_X = np.load(path+'data/fold_'+fold+'/test/X/'+f_id+'.npz')['arr_0']

scaler = pd.read_pickle(path+'data/fold_'+fold+'/scaler/'+f_id+'.pickle')
 
m = GPy.models.GPRegression(trn_X, trn_y, GPy.kern.RBF(trn_X.shape[1], ARD=True))
m.optimize()
K = m.kern.K(trn_X)

greedy = Base(verbose=False)
greedy.cov_np = K
inds, _ = greedy.place(trn_X, N=20)

best_nlml = np.inf
best_model = None
for n in range(2,21):
    model = LLS(trn_X.shape[1], N_l_bar=n, N_l_bar_method='greedy')
    try:
        model.fit(trn_X, trn_y, n_restarts=2, near_opt_inds=inds[:n])
    except:
        continue
    nlml = model.params['likelihood (mll)']
    print(nlml)
    if nlml< best_nlml:
        best_nlml = nlml
        best_model = model
        if not os.path.exists(path+'results/'+model_name+'/fold_'+fold+'/'):
            os.makedirs(path+'results/'+model_name+'/fold_'+fold+'/')
#         pd.to_pickle(model.params, path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')

pred_y = scaler.inverse_transform(best_model.predict(tst_X)[0])

if not os.path.exists(path+'results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'results/'+model_name+'/fold_'+fold+'/')

# np.savez_compressed(path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
