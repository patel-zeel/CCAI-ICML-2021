from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import sys
import os

param_grid = {'max_depth':[1,2,3,4,5,20],
              'max_features':[1,2,3,4,5,6]}
dt = DecisionTreeRegressor(random_state=0)

cvmodel = GridSearchCV(dt, param_grid, refit=True, cv=5)

model_name = sys.argv[0].split('/')[-1].replace('.py','')
path = sys.argv[1]
fold = sys.argv[2]
f_id = sys.argv[3]

trn_X = np.load(path+'data/fold_'+fold+'/train/X/'+f_id+'.npy')
trn_y = np.load(path+'data/fold_'+fold+'/train/y/'+f_id+'.npy')
tst_X = np.load(path+'data/fold_'+fold+'/test/X/'+f_id+'.npy')

scaler = pd.read_pickle(path+'data/fold_'+fold+'/scaler/'+f_id+'.pickle')

cvmodel.fit(trn_X, trn_y.ravel())

pred_y = scaler.inverse_transform(cvmodel.predict(tst_X))

if not os.path.exists(path+'results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'results/'+model_name+'/fold_'+fold+'/')

np.save(path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.npy', pred_y)
pd.to_pickle(cvmodel, path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
