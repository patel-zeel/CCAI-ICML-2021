from stheno import Measure, GP, Matern32, Delta
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

prior = Measure()                  # Construct a prior.
f1 = GP(Matern32(), measure=prior)        # Define our probabilistic model.
f2 = GP(Delta(), measure=prior)
f = f1+f2
post = prior | (f(trn_X), trn_y)           # Compute the posterior distribution.
pred = post(f).mean(tst_X).mat

pred_y = scaler.inverse_transform(pred)

if not os.path.exists(path+'results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'results/'+model_name+'/fold_'+fold+'/')

np.save(path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.npy', pred_y)
# pd.to_pickle(model.param_array, path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
