from scipy.optimize import differential_evolution
import numpy as np
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import ConstantKernel as C, Matern
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

from gp_extras.kernels import LocalLengthScalesKernel
import pandas as pd

np.random.seed(0)

model_name = sys.argv[0].split('/')[-1].replace('.py','')
path = sys.argv[1]
fold = sys.argv[2]
f_id = sys.argv[3]

trn_X = np.load(path+'data2/fold_'+fold+'/train/X/'+f_id+'.npz')['arr_0']
trn_y = np.load(path+'data2/fold_'+fold+'/train/y/'+f_id+'.npz')['arr_0']
tst_X = np.load(path+'data2/fold_'+fold+'/test/X/'+f_id+'.npz')['arr_0']
# emp_cov = np.load(path+'data2/fold_'+fold+'/train/X/'+f_id+'emp_cov.npz')['arr_0']
mean_y = trn_y.mean()

scaler = pd.read_pickle(path+'data2/fold_'+fold+'/scaler/'+f_id+'.pickle')

def de_optimizer(obj_func, initial_theta, bounds):
    res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
                                 bounds, maxiter=50, disp=False, polish=False)
    return res.x, obj_func(res.x, eval_gradient=False)

kernel_lls = C(1.0, (1e-10, 1000)) \
  * LocalLengthScalesKernel.construct(trn_X, l_L=0.1, l_U=2.0, l_samples=4, isotropic=False, l_isotropic=False)
model = GaussianProcessRegressor(kernel=kernel_lls, optimizer=de_optimizer, random_state=0)

model.fit(trn_X, trn_y-mean_y)

import os
if not os.path.exists(path+'data2/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data2/results/'+model_name+'/fold_'+fold+'/')
#         pd.to_pickle(model.params, path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')

pred_y, var_y = model.predict(tst_X, return_cov=True)

pred_y = scaler.inverse_transform(pred_y + mean_y)
var_y = var_y * np.var(scaler.inverse_transform(trn_y))

if not os.path.exists(path+'data2/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data2/results/'+model_name+'/fold_'+fold+'/')

np.savez_compressed(path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'_var.npz', pred_y)
np.savez_compressed(path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
pd.to_pickle(model, path+'data2/results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
