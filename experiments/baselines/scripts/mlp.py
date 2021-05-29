import torch
from torch.functional import F
import numpy as np
import pandas as pd
import sys
import os
np.random.seed(0)
torch.manual_seed(0)

model_name = sys.argv[0].split('/')[-1].replace('.py','')
path = sys.argv[1]
fold = sys.argv[2]
f_id = sys.argv[3]

trn_X = torch.tensor(np.load(path+'data10/fold_'+fold+'/train/X/'+f_id+'.npz')['arr_0']).to(torch.float32)
trn_y = torch.tensor(np.load(path+'data10/fold_'+fold+'/train/y/'+f_id+'.npz')['arr_0']).to(torch.float32)
tst_X = torch.tensor(np.load(path+'data10/fold_'+fold+'/test/X/'+f_id+'.npz')['arr_0']).to(torch.float32)

scaler = pd.read_pickle(path+'data10/fold_'+fold+'/scaler/'+f_id+'.pickle')

# model definition
class MLP(torch.nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(n_inputs, 128)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(128,32)
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(32,8)
        self.act3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Linear(8,1)
#         self.act4 = torch.nn.ReLU()
#         self.layer5 = torch.nn.Linear(16,8)
#         self.act5 = torch.nn.ReLU()
#         self.layer6 = torch.nn.Linear(8,4)
#         self.act6 = torch.nn.ReLU()
#         self.layer7 = torch.nn.Linear(4,1)
#         self.act7 = torch.nn.ReLU()
        
    # forward propagate input
    def forward(self, X):
        d = 0
        X = self.layer1(X)
        X = F.dropout(self.act1(X), d)
        X = self.layer2(X)
        X = F.dropout(self.act2(X), d)
        X = self.layer3(X)
        X = F.dropout(self.act3(X), d)
        X = self.layer4(X)
#         X = F.dropout(self.act1(X), d)
#         X = self.layer5(X)
#         X = F.dropout(self.act1(X), d)
#         X = self.layer6(X)
#         X = F.dropout(self.act1(X), d)
#         X = self.layer7(X)
#         X = self.act7(X)
        return X

model = MLP(trn_X.shape[1])
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

device='cpu'

model.to(device)
model.train()

losses = []
for it in range(150):
    optimizer.zero_grad()
    outs = model(trn_X.to(device))
    loss = criterion(outs, trn_y.to(device))
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

model.eval()
pred = model(tst_X.to(device)).cpu().detach().numpy()
pred_y = scaler.inverse_transform(pred)

if not os.path.exists(path+'data10/results/'+model_name+'/fold_'+fold+'/'):
    os.makedirs(path+'data10/results/'+model_name+'/fold_'+fold+'/')

np.savez_compressed(path+'data10/results/'+model_name+'/fold_'+fold+'/'+f_id+'.npz', pred_y)
pd.to_pickle({'model':model, 'loss':losses}, path+'data10/results/'+model_name+'/fold_'+fold+'/'+f_id+'.model')
