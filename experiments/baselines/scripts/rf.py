from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import sys
import os

kernel = ('linear', 'rbf', 'sigmoid')
param_grid = {'n_estimators':[50,100,150,200], 
              'max_depth':[1,2,3,4,5,20],
              'max_features':[1,2,3,4,5,6]}
rf = RandomForestRegressor(random_state=0)

cvmodel = GridSearchCV(rf, param_grid, refit=True, cv=5)

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
