import math
import tqdm
import torch
import gpytorch
import numpy as np
import pandas as pd
import sys
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np.random.seed(0)
torch.manual_seed(0)

model_name = sys.argv[0].split('/')[-1].replace('.py','')
training_iterations = 500
path = sys.argv[1]
fold = sys.argv[2]
f_id = sys.argv[3]

trn_X = torch.tensor(np.load(path+'data10/fold_'+fold+'/train/X/'+f_id+'.npz')['arr_0'], dtype=torch.float32)
trn_y = torch.tensor(np.load(path+'data10/fold_'+fold+'/train/y/'+f_id+'.npz')['arr_0'], dtype=torch.float32)
tst_X = torch.tensor(np.load(path+'data10/fold_'+fold+'/test/X/'+f_id+'.npz')['arr_0'], dtype=torch.float32)
tst_y = torch.tensor(np.load(path+'data10/fold_'+fold+'/test/y/'+f_id+'.npz')['arr_0'], dtype=torch.float32)
mean_y = trn_y.mean()
data_dim = trn_X.size(-1)
n_dim = 1

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, n_dim))

feature_extractor = LargeFeatureExtractor()

scaler = pd.read_pickle(path+'data10/fold_'+fold+'/scaler/'+f_id+'.pickle')

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.GridInterpolationKernel(
#             gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=n_dim)),
#             num_dims=n_dim, grid_size=100
#         )
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=n_dim))
        self.feature_extractor = feature_extractor

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(trn_X, trn_y.ravel()-mean_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def train():
    iterator = tqdm.notebook.tqdm(range(training_iterations))
    best_loss = np.inf
    best_state = None
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(trn_X)
#         print(output.shape, type(output), output)
        # Calc loss and backprop derivatives
        loss = -mll(output, trn_y.ravel()-mean_y)
        if loss<best_loss:
            best_loss = loss
            best_state = model.state_dict()
#         losses.append(loss)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
    model.load_state_dict(best_state)

# losses = train()
# print(tst_X.shape, trn_X.shape)    
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
#     print(mean_y)
    preds = model(tst_X).mean.numpy() + mean_y.numpy()

pred_y = scaler.inverse_transform(preds)
# print(mean_squared_error(tst_y.ravel(), pred_y, squared=False))
if not os.path.exists(path+'data10/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data10/results/'+model_name+'/fold_'+fold+'/')

np.savez_compressed(path+'data10/results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
pd.to_pickle(model.state_dict(), path+'data10/results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
# os.system('gzip -f '+path+'data10/results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
