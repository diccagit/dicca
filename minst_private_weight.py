#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is for Nosiy MNIST dataset.
"""

import torch
import torch.nn as nn
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sn
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from pri.distributions import Normal
from pri.model import BayesianGroupLassoGenerator, NormalNet
from pri.pcca import NormalPriorTheta, PCCA
from pri.utils import Lambda,load_data,ConcatDataset

batch_size = 128
dim_z = 30
dim_x = 784
dim_h = 1024
prior_theta_scale = 1
lam = 1
lam_adjustment = 1

mc_samples = 1
n_layers = 1

data1 = load_data('noisymnist_view1.gz')
data2 = load_data('noisymnist_view2.gz')

train_set_x1, train_set_y1 = data1[0]
valid_set_x1, valid_set_y1 = data1[1]
test_set_x1, test_set_y1 = data1[2]

train_set_x2, train_set_y2 = data2[0]
valid_set_x2, valid_set_y2 = data2[1]
test_set_x2, test_set_y2 = data2[2]


train_loader = torch.utils.data.DataLoader(
               ConcatDataset(
                       train_set_x1[:12800],
                       train_set_x2[:12800]),
                batch_size=batch_size, shuffle=True)


groups = [
  ['view1'],
  ['view2']
  ]

group_names = [g[0] for g in groups]



num_groups = len(groups)
stddev_multiple = 1
group_input_dim = 30
group_dims = [784,784]

encoder1 = NormalNet(
  mu_net=nn.Sequential(
    nn.Linear(dim_x, dim_h),
    nn.Linear(dim_h, dim_h),
    nn.ReLU(),
    nn.Linear(dim_h, dim_z)
  ),


  # Learned standard deviation as a function of the input
  sigma_net=torch.nn.Sequential(
    # inference_net_base,
    nn.Linear(dim_x, dim_h),
    torch.nn.Linear(dim_h, dim_z),
    nn.Softplus()
  )
)
  
encoder2 = NormalNet(
  mu_net=nn.Sequential(
    nn.Linear(dim_x, dim_h),
    nn.ReLU(),
    nn.Linear(dim_h,dim_z)
  ),


  # Learned standard deviation as a function of the input
  sigma_net=torch.nn.Sequential(
    # inference_net_base,nn.Linear(dim_x, dim_h),
    nn.Linear(dim_x, dim_h),
    nn.Linear(dim_h,dim_z),

    nn.Softplus()
  )
)


encoder_z = NormalNet(
  mu_net=nn.Sequential(
    nn.Linear(dim_x + dim_x, dim_z)
  ),


  # Learned standard deviation as a function of the input
  sigma_net=torch.nn.Sequential(
    # inference_net_base,
    torch.nn.Linear(dim_x + dim_x, dim_z),
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
            nn.Linear(group_input_dim, output_dim)),
    sigma_net=Lambda(
      lambda x, log_sigma: torch.exp(log_sigma.expand(x.size(0), -1)) + 1e-3,
      extra_args=(log_sigma,)
    )
  )

generative_net = BayesianGroupLassoGenerator(
  group_generators=[make_group_generator(dim) for dim in group_dims],
  group_input_dim=group_input_dim,
  dim_z=dim_z
)

def debug_z_by_group_matrix():
    
    # groups x dim_z
  fig, ax = plt.subplots()
  W_col_norms = torch.sqrt(
    torch.sum(torch.pow(generative_net.W.data, 2), dim=2)
  )
  W_col_norms_prop = W_col_norms / torch.max(W_col_norms, dim=0)[0]
  ax.imshow(W_col_norms_prop.numpy(), aspect='equal')
  ax.set_xlabel('dimensions of z')
  ax.set_ylabel('group generative nets')
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
  {'params': generative_net.group_generators_parameters(), 'lr': lr_generativenet,'betas': betas_generativenet},
  {'params': [gen.sigma_net.extra_args[0] for gen in generative_net.group_generators], 'lr': lr, 'betas':betas}
])

    
Ws_lr = 1e-4
optimizer_Ws = torch.optim.SGD([
  {'params': [generative_net.W], 'lr': Ws_lr, 'momentum': 0}
])

model = PCCA(
  encoder_z = encoder_z,
  encoder1 = encoder1,
  encoder2 = encoder2,
  generative_model = generative_net,
  prior_z = prior_z,
  prior_theta = NormalPriorTheta(prior_theta_scale),
  lam = lam,
  optimizers = [optimizer, optimizer_Ws]
)

plot_interval = 10
elbo_per_iter = []


def train(epoch):
    
    for batch_idx, (data1,data2) in enumerate(train_loader):
        data1 = Variable(data1).float()
        data2 = Variable(data2).float()
        data = torch.cat((data1,data2),1)
        
        if epoch > 10:
            info = model.step(
                X=data,
                y1=data1,
                y2=data2,
                prox_step_size=Ws_lr * lam * lam_adjustment,
                mc_samples=mc_samples
                )
        
        elbo_per_iter.append(info['elbo'].data[0])
    
        if epoch % plot_interval == 0:
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
        print('     -KL(q(z1) || p(z1))', info['z1_kl'].data[0])
        print('     -KL(q(z2) || p(z2))', info['z2_kl'].data[0] )
        print('    log p(theta)     ', info['logprob_theta'].data[0])
        print('    log p(W)         ', info['logprob_W'].data[0])
        #print('    log p(W1)         ', info['logprob_Wa'].data[0])
        #print('    log p(W2)         ', info['logprob_Wb'].data[0])

#for epoch in range(1, EPOCHS+1):
#    train(epoch)
num_epochs = 1000
plot_interval = 100
for epoch in range(num_epochs):        
        
    for batch_idx, (data1,data2) in enumerate(train_loader):
        data1 = Variable(data1).float()
        data2 = Variable(data2).float()
        data = torch.cat((data1,data2),1)
        
        if epoch > 10:
            stddev_multiple = 2

        info = model.step(
                X=data,
                y1=data1,
                y2=data2,
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
            print('     -KL(q(z1) || p(z1))', info['z1_kl'].data[0])
            print('     -KL(q(z2) || p(z2))', info['z2_kl'].data[0] )
            print('    log p(theta)     ', info['logprob_theta'].data[0])
            print('    log p(W)         ', info['logprob_W'].data[0])
            print('    log p(Wa)         ', info['logprob_Wa'].data[0])
            print('    log p(Wb)         ', info['logprob_Wb'].data[0])
        


  
# recounstruct 
    
z1 = encoder1(data1).sample()
z2 = encoder2(data2).sample()
zc = [z1, z2]
z = encoder_z(data).sample()
wa = generative_net.Wa
wb = generative_net.Wb
Wc = [wa,wb]
fX = generative_net(z,zc, Wc).sample()
# view1
plt.imshow(data1[2].view(28,28).data.squeeze().numpy(),cmap="gray")
plt.imshow(fX[2][:784].view(28,28).data.squeeze().numpy(),cmap='gray')

def save_img_and_reconstruction(ix):
  plt.figure()
  plt.imshow(data1[ix].view(28,28).data.squeeze().numpy(),cmap="gray")
  plt.savefig('minst_recon_lambda1/true_view1_{}.pdf'.format(ix), format='pdf')
  
  plt.imshow(data2[ix].view(28,28).data.squeeze().numpy(),cmap="gray")
  plt.savefig('minst_recon_lambda1/true_view2_{}.pdf'.format(ix), format='pdf')
  
  
  plt.figure()
  plt.imshow(fX[ix][:784].view(28,28).data.squeeze().numpy(),cmap='gray')
  plt.savefig('minst_recon_lambda1/view1_reconstruction_{}.pdf'.format(ix), format='pdf')
  plt.imshow(fX[ix][784:].view(28,28).data.squeeze().numpy(),cmap='gray')
  plt.savefig('minst_recon_lambda1/view2_reconstruction_{}.pdf'.format(ix), format='pdf')





# view2
plt.imshow(data2[1].view(28,28).data.squeeze().numpy(),cmap='gray')
plt.imshow(fX[1][784:].view(28,28).data.squeeze().numpy(),cmap='gray')

#

test_data1 = Variable(torch.from_numpy(test_set_x1)).float()
test_data2 = Variable(torch.from_numpy(test_set_x2)).float()
test_data = torch.cat((test_data1,test_data2),1)
z1 = encoder1(test_data1).sample()
z2 = encoder2(test_data2).sample()
zc = [z1, z2]
z = encoder_z(test_data).sample()
wa = generative_net.Wa
wb = generative_net.Wb
Wc = [wa,wb]
fX = generative_net(z,zc, Wc).sample()

generated = fX  

#plt.savefig('{}_reconstruction_full_sample.pdf'.format(ix), format='pdf')

latent = torch.mm(z.data, generative_net.W[1].data)
# MSE
 
# T-SNE 

tsne = TSNE(n_components=2, random_state=0)
pz = latent.numpy()
pz = pd.DataFrame(pz)
y2_test = pd.DataFrame(test_set_y2)
df = pz
df['label'] = y2_test
z_tsne = tsne.fit_transform(pz)

# plot 2 for view1
tsne_data = np.vstack((z_tsne.T, test_set_y2)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1','Dim_2','labels'))
#sn.set(style="white", rc={'figure.figsize':(7,5),'figure.dpi':200})
sn.FacetGrid(tsne_df, hue='labels', size=6).map(plt.scatter,'Dim_1','Dim_2').add_legend()
plt.show()


#pz2 = z2.data.numpy()
x2 = pd.DataFrame(test_set_x2)
y2_test = pd.DataFrame(test_set_y2)
dfx2 = x2
dfx2['label'] = y2_test
x2_tsne = tsne.fit_transform(x2)

# plot1 for  view2
fig = plt.figure(figsize=(6,5),dpi=200)
ax = fig.add_subplot(1,1,1, title='TSNE')
ax.scatter(x=x2_tsne[:,0],
           y=x2_tsne[:,1],
           c=dfx2['label'],
           cmap=plt.cm.get_cmap('Paired'))
plt.legend()
plt.show()  


# plot 2 for view 2
tsne_x2 = np.vstack((x2_tsne.T, test_set_y2)).T
tsne_x2 = pd.DataFrame(data=tsne_x2, columns=('Dim_1','Dim_2','labels'))
sn.set(style="white", rc={'figure.figsize':(7,5),'figure.dpi':200})
sn.FacetGrid(tsne_x2, hue='labels', size=6).map(plt.scatter,'Dim_1','Dim_2').add_legend()
plt.show() 


