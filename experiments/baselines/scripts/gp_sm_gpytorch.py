import torch
import gpytorch
import numpy as np
import pandas as pd
import sys
import os
np.random.seed(0)

model_name = sys.argv[0].split('/')[-1].replace('.py','')
path = sys.argv[1]
fold = sys.argv[2]
f_id = sys.argv[3]
training_iter = 200

trn_X = torch.tensor(np.load(path+'data/fold_'+fold+'/train/X/'+f_id+'.npz')['arr_0'])
trn_y = torch.tensor(np.load(path+'data/fold_'+fold+'/train/y/'+f_id+'.npz')['arr_0'].ravel())
tst_X = torch.tensor(np.load(path+'data/fold_'+fold+'/test/X/'+f_id+'.npz')['arr_0'])
mean_y = trn_y.mean()

scaler = pd.read_pickle(path+'data/fold_'+fold+'/scaler/'+f_id+'.pickle')

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=train_x.shape[1])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(trn_X, trn_y-mean_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(trn_X)
    # Calc loss and backprop gradients
    loss = -mll(output, trn_y-mean_y)
    print(i, loss.item())
    loss.backward()
    optimizer.step()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = model(tst_X).mean.numpy() + mean_y.numpy()

pred_y = scaler.inverse_transform(pred)

if not os.path.exists(path+'results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'results/'+model_name+'/fold_'+fold+'/')

np.savez_compressed(path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
# pd.to_pickle(model.param_array, path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
