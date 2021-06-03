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

trn_X = np.load(path+'data2/fold_'+fold+'/train/X/'+f_id+'.npz')['arr_0']
trn_y = np.load(path+'data2/fold_'+fold+'/train/y/'+f_id+'.npz')['arr_0']
tst_X = np.load(path+'data2/fold_'+fold+'/test/X/'+f_id+'.npz')['arr_0']
mean_y = trn_y.mean()

scaler = pd.read_pickle(path+'data2/fold_'+fold+'/scaler/'+f_id+'.pickle')

model = GPy.models.GPRegression(trn_X, trn_y-mean_y, GPy.kern.Matern32(trn_X.shape[1], ARD=True), normalizer=False)
# model.kern.lengthscale.constrain_bounded(10**-5, 20)
model.optimize_restarts(5, robust=True, verbose=False)
pred, var = model.predict(tst_X)

pred_y = scaler.inverse_transform(pred+mean_y)
pred_var = np.var(scaler.inverse_transform(trn_y))*var

if not os.path.exists(path+'data2/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data2/results/'+model_name+'/fold_'+fold+'/')

np.savez_compressed(path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
np.savez_compressed(path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'_var.npz', pred_y)
pd.to_pickle(model, path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
