import GPy
import numpy as np
import pandas as pd
import sys
import os
np.random.seed(0)

model_name = sys.argv[0].split('/')[-1].replace('.py','')
path = sys.argv[1]
fold = sys.argv[2]
f_id = sys.argv[3]

trn_X = np.load(path+'data/fold_'+fold+'/train/X/'+f_id+'.npz')['arr_0']
trn_y = np.load(path+'data/fold_'+fold+'/train/y/'+f_id+'.npz')['arr_0']
tst_X = np.load(path+'data/fold_'+fold+'/test/X/'+f_id+'.npz')['arr_0']
mean_y = trn_y.mean()

scaler = pd.read_pickle(path+'data/fold_'+fold+'/scaler/'+f_id+'.pickle')

k = GPy.kern.RatQuad(trn_X.shape[1], ARD=True, active_dims=[0,1])
model = GPy.models.GPRegression(trn_X, trn_y-mean_y, k)
# model.kern.lengthscale.constrain_bounded(10**-5, 20)
model.optimize_restarts(5, robust=True, verbose=False)
pred = model.predict(tst_X)[0] + mean_y

pred_y = scaler.inverse_transform(pred)

if not os.path.exists(path+'results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'results/'+model_name+'/fold_'+fold+'/')

np.savez_compressed(path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
pd.to_pickle(model, path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
