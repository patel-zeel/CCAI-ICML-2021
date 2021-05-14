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

trn_X = np.load(path+'data/fold_'+fold+'/train/X/'+f_id+'.npy')
trn_y = np.load(path+'data/fold_'+fold+'/train/y/'+f_id+'.npy')
tst_X = np.load(path+'data/fold_'+fold+'/test/X/'+f_id+'.npy')

scaler = pd.read_pickle(path+'data/fold_'+fold+'/scaler/'+f_id+'.pickle')

model = LLS(trn_X.shape[1], N_l_bar=3)
model.fit(trn_X, trn_y, n_restarts=10)

pred_y = scaler.inverse_transform(model.predict(tst_X)[0])

if not os.path.exists(path+'results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'results/'+model_name+'/fold_'+fold+'/')

np.save(path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.npy', pred_y)
# pd.to_pickle(model.param_array, path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
