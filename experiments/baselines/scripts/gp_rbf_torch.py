import torch
import numpy as np
import pandas as pd
import sys
import os
np.random.seed(0)

model_name = sys.argv[0].split('/')[-1].replace('.py','')
path = sys.argv[1]
fold = sys.argv[2]
f_id = sys.argv[3]

trn_X = torch.tensor(np.load(path+'data/fold_'+fold+'/train/X/'+f_id+'.npz')['arr_0'])
trn_y = torch.tensor(np.load(path+'data/fold_'+fold+'/train/y/'+f_id+'.npz')['arr_0'])
tst_X = torch.tensor(np.load(path+'data/fold_'+fold+'/test/X/'+f_id+'.npz')['arr_0'])

scaler = pd.read_pickle(path+'data/fold_'+fold+'/scaler/'+f_id+'.pickle')
pi = torch.tensor(np.pi)

def rbf_kernel(Xi, Xj, ls, std):
    dist = torch.square(Xi[:,np.newaxis,:] - Xj[np.newaxis,:,:])
    scaled_dist = dist[:,:,0]/(2*ls**2)
    
    return std**2 * torch.exp(-scaled_dist)

def nlml(ls, std, sigma_n):
    K = rbf_kernel(trn_X, trn_X, ls, std)
    K += sigma_n**2 * torch.eye(K.shape[0])
    L = torch.cholesky(K)
    alpha = torch.cholesky_solve(trn_y, L)
    
    ans = 0.5*(trn_y.T@alpha + torch.sum(torch.log(torch.diag(L))) + trn_X.shape[0]*torch.log(2*pi))
    return ans[0,0]

ls = torch.tensor(1., requires_grad=True)              
std = torch.tensor(1., requires_grad=True)
sigma_n = torch.tensor(1., requires_grad=True)
              
optimizer = torch.optim.Adam([ls, std, sigma_n], lr=0.1)            

for i in range(100):
    optimizer.zero_grad()
    loss = nlml(ls, std, sigma_n)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    K = rbf_kernel(trn_X, trn_X, ls, std)
    K += torch.eye(trn_X.shape[0])*sigma_n**2
    L = torch.cholesky(K)
    alpha = torch.cholesky_solve(trn_y, L)
    K_ = rbf_kernel(tst_X, trn_X, ls, std)
    
    pred = (K_@alpha).numpy()
    
pred_y = scaler.inverse_transform(pred)

if not os.path.exists(path+'results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'results/'+model_name+'/fold_'+fold+'/')

np.savez_compressed(path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
# pd.to_pickle(model.param_array, path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
