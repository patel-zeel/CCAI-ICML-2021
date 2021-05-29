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

# model definition
class MLP(torch.nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(n_inputs, 128)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(128,32)
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(32,1)
        
    # forward propagate input
    def forward(self, X):
        X = self.layer1(X)
        X = self.act1(X)
        X = self.layer2(X)
        X = self.act2(X)
        X = self.layer3(X)
        return X

model = MLP(trn_X.shape[1])
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

model.train()

for it in range(100):
    optimizer.zero_grad()
    outs = model(trn_X)
    loss = criterion(outs, trn_y.ravel())
    print(it, loss)
    loss.backward()
    optimizer.step()

model.eval()
pred = model(tst_X)
pred_y = scaler.inverse_transform(pred)

if not os.path.exists(path+'results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'results/'+model_name+'/fold_'+fold+'/')

np.savez_compressed(path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
pd.to_pickle(model, path+'results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
