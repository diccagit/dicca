#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: luke
"""
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
#from pac.bars_data import (sample_bars_image, sample_many_bars_images,
#                           sample_one_bar_image)
from pri.distributions import Normal
from pri.model_m import BayesianGroupLassoGenerator, NormalNet
from pri.pcca_m import NormalPriorTheta, PCCA
from pri.utils import Lambda,load_data,ConcatDataset
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

drug = torch.load("data/cll/drug_minmax.pt")
methy = torch.load("data/cll/methylation_minmax.pt")
mrna = torch.load("data/cll/mrna_minmax.pt")
mutation = torch.load("data/cll/mutation_minmax.pt") 

drug_train, durg_test = drug[:144], drug[144:180]
methy_train, methy_test = methy[:144], methy[144:180]
mrna_train, mrna_test = mrna[:144], mrna[144:180]
mutation_train, mutation_test = mutation[:144],mutation[144:180]

all_train = torch.cat((drug_train, methy_train, mrna_train, mutation_train),1)
#all_train = torch.cat((methy_train, mrna_train),1)
batch_size = 12
dim_z = 10
dim_h = 200
dim_drug = 310
dim_methy = 4248
dim_mrna = 5000
dim_mutation = 69
prior_theta_scale = 1
lam = 1
lam_adjustment = 1

mc_samples = 1
n_layers = 1

train_loader = torch.utils.data.DataLoader(
               ConcatDataset(
                       drug_train,
                       methy_train,
                       mrna_train,
                       mutation_train),
                batch_size=batch_size, shuffle=True)


groups = [
  ['drug'],
  ['methylation'],
  ['mRNA'],
  ['mutation']
 ]


group_names = [g[0] for g in groups]

num_groups = len(groups)
stddev_multiple = 1
group_input_dim = 10
group_dims = [310,4248,5000,69]


encoder1 = NormalNet(
  mu_net=nn.Sequential(
    nn.Linear(dim_drug, dim_h),
    nn.Linear(dim_h, dim_h),
    nn.ReLU(),
    nn.Linear(dim_h, dim_z)
  ),
#
#
#  # Learned standard deviation as a function of the input
  sigma_net=torch.nn.Sequential(
    nn.Linear(dim_drug, dim_h),
    torch.nn.Linear(dim_h, dim_z),
    nn.Softplus()
  )
)
  
encoder2 = NormalNet(
  mu_net=nn.Sequential(
    nn.Linear(dim_methy, dim_h),
    nn.ReLU(),
    nn.Linear(dim_h,dim_z)
  ),


  # Learned standard deviation as a function of the input
  sigma_net=torch.nn.Sequential(
    nn.Linear(dim_methy, dim_h),
    nn.Linear(dim_h,dim_z),
    nn.Softplus()
  )
)

encoder3 = NormalNet(
  mu_net=nn.Sequential(
    nn.Linear(dim_mrna, dim_h),
    nn.ReLU(),
    nn.Linear(dim_h,dim_z)
  ),


  # Learned standard deviation as a function of the input
  sigma_net=torch.nn.Sequential(
    nn.Linear(dim_mrna, dim_h),
    nn.Linear(dim_h,dim_z),
    nn.Softplus()
  )
)

encoder4 = NormalNet(
  mu_net=nn.Sequential(
    nn.Linear(dim_mutation, dim_h),
    nn.ReLU(),
    nn.Linear(dim_h,dim_z)
  ),


  # Learned standard deviation as a function of the input
  sigma_net=torch.nn.Sequential(
    nn.Linear(dim_mutation, dim_h),
    nn.Linear(dim_h,dim_z),
    nn.Softplus()
  )
)


encoder_z = NormalNet(
  mu_net=nn.Sequential(
    nn.Linear(dim_drug + dim_methy + dim_mrna + dim_mutation, dim_z)
  ),


  # Learned standard deviation as a function of the input
  sigma_net=torch.nn.Sequential(
    # inference_net_base,
    torch.nn.Linear(dim_drug + dim_methy + dim_mrna + dim_mutation, dim_z),
    Lambda(torch.exp),
    Lambda(lambda x: x * stddev_multiple + 1e-3)
  )
)


def make_group_generator(output_dim):
  # Note that this Variable is NOT going to show up in `net.parameters()` and
  # therefore it is implicitly free from the ridge penalty/p(theta) prior.
  log_sigma = Variable(
    torch.log(1e-2 * torch.ones(output_dim)),
    requires_grad=True
  )
  return NormalNet(
    mu_net=nn.Sequential(
            nn.Tanh(),
            nn.Linear(output_dim, output_dim)),
    sigma_net=Lambda(
      lambda x, log_sigma: torch.exp(log_sigma.expand(x.size(0), -1)) + 1e-3,
      extra_args=(log_sigma,)
    )
  )

generative_net = BayesianGroupLassoGenerator(
  group_generators=[make_group_generator(dim) for dim in group_dims],
  group_dims=group_dims,
  dim_z=dim_z
)

def debug_z_by_group_matrix():
    
    # groups x dim_z
  fig, ax = plt.subplots(dpi=200)
  W_col_norms = torch.sqrt(
    torch.sum(torch.pow(generative_net.W.data, 2), dim=2)
  )
  W_col_norms_prop = W_col_norms / torch.max(W_col_norms, dim=0)[0]
  ax.imshow(W_col_norms_prop, aspect='equal',cmap = 'Blues')
  ax.set_xlabel('dimensions of z')
  ax.set_ylabel('group generative nets')
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
  ax.yaxis.set_ticks(np.arange(len(groups)))
  ax.yaxis.set_ticklabels(group_names)

def share_group_dependency():
    fig, ax = plt.subplots(dpi=200)
    wa_row = torch.sum(torch.pow(generative_net.Wa.data, 2), dim=1)
    wb_row = torch.sum(torch.pow(generative_net.Wb.data, 2), dim=1)
    wc_row = torch.sum(torch.pow(generative_net.Wc.data, 2), dim=1)
    wd_row = torch.sum(torch.pow(generative_net.Wd.data, 2), dim=1)
    W_col_norms = torch.stack((wa_row, wb_row, wc_row, wd_row),dim=0)
    #W_col_norms_prop = W_col_norms / torch.max(W_col_norms, dim=0)[0]
    ax.imshow(W_col_norms, aspect='equal', cmap = 'Blues')
    ax.set_xlabel('factor')
    ax.set_ylabel('group')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_ticks(np.arange(len(groups)))
    ax.yaxis.set_ticklabels(group_names)

def private_group_dependency():
    fig, ax = plt.subplots(dpi=200)
    wa_row = torch.sum(torch.pow(generative_net.Waa.data, 2), dim=1)
    wb_row = torch.sum(torch.pow(generative_net.Wbb.data, 2), dim=1)
    wc_row = torch.sum(torch.pow(generative_net.Wcc.data, 2), dim=1)
    wd_row = torch.sum(torch.pow(generative_net.Wdd.data, 2), dim=1)
    W_col_norms = torch.stack((wa_row, wb_row, wc_row, wd_row),dim=0)
    W_col_norms_prop = W_col_norms / torch.max(W_col_norms, dim=0)[0]
    ax.imshow(W_col_norms_prop, aspect='equal',cmap = 'Blues')
    ax.set_xlabel('factor')
    ax.set_ylabel('group')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_ticks(np.arange(len(groups)))
    ax.yaxis.set_ticklabels(group_names)

   
def debug_z_by_group_matrix_view1():
    
    # groups x dim_z
  fig, ax = plt.subplots()
  W_col_norms = torch.sqrt(
    torch.sum(torch.pow(generative_net.Wa.data, 2), dim=1)
  )
  ax.imshow(W_col_norms.numpy(), aspect='equal')
  ax.set_xlabel('dimensions of z')
  ax.set_ylabel('group generative nets')
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
 

 
prior_z = Normal(
  Variable(torch.zeros(batch_size, dim_z)),
  Variable(torch.ones(batch_size, dim_z))
)


lr = 1e-4
betas = (0.9,0.999)

lr_inferencenet = 1e-4
betas_inferencenet = (0.9,0.999)

lr_generativenet = 1e-4
betas_generativenet = (0.9,0.999)


optimizer = torch.optim.Adam([
  {'params': encoder_z.parameters(), 'lr': lr,'betas':betas_inferencenet},
  {'params': encoder1.parameters(), 'lr': lr, 'betas':betas_inferencenet},
  {'params': encoder2.parameters(), 'lr': lr,'betas':betas_inferencenet},
  {'params': encoder3.parameters(), 'lr': lr, 'betas':betas_inferencenet},
  {'params': encoder4.parameters(), 'lr': lr,'betas':betas_inferencenet}, 
  {'params': generative_net.group_generators_parameters(), 'lr': lr_generativenet,'betas': betas_generativenet},
  {'params': [gen.sigma_net.extra_args[0] for gen in generative_net.group_generators], 'lr': lr, 'betas':betas}
])

    
Ws_lr = 1e-4
optimizer_Ws = torch.optim.SGD([
  #{'params': [generative_net.W], 'lr': Ws_lr, 'momentum': 0},
  {'params': [generative_net.Wa], 'lr': Ws_lr, 'momentum': 0},
  {'params': [generative_net.Wb], 'lr': Ws_lr, 'momentum': 0},
  {'params': [generative_net.Wc], 'lr': Ws_lr, 'momentum': 0},
  {'params': [generative_net.Wd], 'lr': Ws_lr, 'momentum': 0},
  {'params': [generative_net.Waa], 'lr': Ws_lr, 'momentum': 0},
  {'params': [generative_net.Wbb], 'lr': Ws_lr, 'momentum': 0},
  {'params': [generative_net.Wcc], 'lr': Ws_lr, 'momentum': 0},
  {'params': [generative_net.Wdd], 'lr': Ws_lr, 'momentum': 0},
])

model = PCCA(
  encoder_z = encoder_z,
  encoder1 = encoder1,
  encoder2 = encoder2,
  encoder3 = encoder3,
  encoder4 = encoder4,
  generative_model = generative_net,
  prior_z = prior_z,
  prior_theta = NormalPriorTheta(prior_theta_scale),
  lam = lam,
  optimizers = [optimizer, optimizer_Ws]
)

plot_interval = 10
elbo_per_iter = []
num_epochs = 2000
for epoch in range(num_epochs):        
        
    for batch_idx, (data_drug,data_methy,data_mrna, data_mutation) in enumerate(train_loader):
        data_drug = Variable(data_drug).float()
        data_methy = Variable(data_methy).float()
        data_mrna = Variable(data_mrna).float()
        data_mutation = Variable(data_mutation).float()
        data = torch.cat((data_drug, data_methy, data_mrna, data_mutation),1)
        
        if epoch > 10:
            stddev_multiple = 2
        info = model.step(
                X=data,
                y1=data_drug,
                y2=data_methy,
                y3=data_mrna,
                y4=data_mutation,
                prox_step_size=Ws_lr * lam * lam_adjustment,
                mc_samples=mc_samples
                )
        
        elbo_per_iter.append(info['elbo'].data[0])
    
        if batch_idx % plot_interval == 0:
    #debug_z_by_group_matrix(8)

            plt.figure()
            plt.plot(elbo_per_iter)
            plt.xlabel('iteration')
            plt.ylabel('ELBO')
            plt.title('ELBO per iteration. lam = {}'.format(lam))
            plt.show()  
      
            print('epoch', epoch)
            print('  ELBO:', info['elbo'].data[0])
            print('    loglik_term      ', info['loglike'].data[0])
            print('    -KL(q(z) || p(z))', -info['z_kl'].data[0])
            print('    log p(W)         ', info['logprob_W'].data[0])
            print('    log p(Ws)         ', info['logprob_Ws'].data[0])
        
    
  
# recounstruct 
drug_train= Variable(drug_train).float()
methy_train = Variable(methy_train).float()
mrna_train = Variable(mrna_train).float()  
mutation_train = Variable(mutation_train).float()
all_train = Variable(all_train).float()

z1 = encoder1(drug_train).sample()
z2 = encoder2(methy_train).sample()
z3 = encoder3(mrna_train).sample()
z4 = encoder4(mutation_train).sample()            
            
zc = [z1, z2, z3, z4]
z = encoder_z(all_train).sample()
wa = generative_net.Wa
wb = generative_net.Wb
wc = generative_net.Wc
wd = generative_net.Wd
waa = generative_net.Waa
wbb = generative_net.Wbb
wcc = generative_net.Wcc
wdd = generative_net.Wdd

W = [wa,wb,wc,wd]
Ws = [waa,wbb,wcc,wdd]
fX = generative_net(z,zc,W, Ws).sample()

      

