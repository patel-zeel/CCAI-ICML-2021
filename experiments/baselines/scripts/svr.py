from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import sys
import os

C = (0.1, 0.5, 1, 5, 10)
kernel = ('linear', 'rbf', 'sigmoid')
param_grid = {'C':C, 'kernel':kernel}
svr = SVR()
cvmodel = GridSearchCV(svr, param_grid, refit=True, cv=5)
model_name = sys.argv[0].split('/')[-1].replace('.py','')
path = sys.argv[1]
fold = sys.argv[2]
f_id = sys.argv[3]
trn_X = np.load(path+'data2/fold_'+fold+'/train/X/'+f_id+'.npz')['arr_0']
trn_y = np.load(path+'data2/fold_'+fold+'/train/y/'+f_id+'.npz')['arr_0']
tst_X = np.load(path+'data2/fold_'+fold+'/test/X/'+f_id+'.npz')['arr_0']
scaler = pd.read_pickle(path+'data2/fold_'+fold+'/scaler/'+f_id+'.pickle')

cvmodel.fit(trn_X, trn_y.ravel())
pred_y = scaler.inverse_transform(cvmodel.predict(tst_X))

if not os.path.exists(path+'data2/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data2/results/'+model_name+'/fold_'+fold+'/')
np.savez_compressed(path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
pd.to_pickle(cvmodel, path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
