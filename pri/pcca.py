#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import torch
from torch.autograd import Variable

from .distributions import Normal
from .utils import KL_Normals


class NormalPriorTheta(object):
  """A distribution that places a zero-mean Normal distribution on all of the
  `group_generators` in a BayesianGroupLassoGenerator."""

  def __init__(self, sigma):
    self.sigma = sigma

  def logprob(self, module):
    return sum(
      Normal(
        torch.zeros_like(param),
        self.sigma * torch.ones_like(param)
      ).logprob(param)
      for gen in module.group_generators
      for param in gen.parameters()
    )

class PCCA(object):
  def __init__(
      self,
      encoder_z,
      encoder1,
      encoder2,
      encoder3,
      encoder4,
      generative_model,
      prior_z,
      prior_theta,
      lam,
      optimizers
  ):
    self.encoder_z = encoder_z
    self.encoder1 = encoder1
    self.encoder2 = encoder2
    self.encoder3 = encoder3
    self.encoder4 = encoder4
    self.generative_model = generative_model
    self.prior_z = prior_z
    self.prior_theta = prior_theta
    self.lam = lam
    self.optimizers = optimizers

  def step(self, X, y1, y2, y3, y4,prox_step_size, mc_samples):
    
    batch_size = X.size(0)

    # [batch_size, dim_z]
    q_z = self.encoder_z(X)
    q_z1 = self.encoder1(y1)
    q_z2 = self.encoder2(y2)
    q_z3 = self.encoder3(y3)
    q_z4 = self.encoder4(y4)
    #print(q_z.size())
    # KL divergence is additive across independent joint distributions, so this
    # works appropriately.
    z_kl = KL_Normals(q_z, self.prior_z.expand_as(q_z)) / batch_size
    z1_kl = KL_Normals(q_z1, self.prior_z.expand_as(q_z1)) / batch_size
    z2_kl = KL_Normals(q_z2, self.prior_z.expand_as(q_z2)) / batch_size
    z3_kl = KL_Normals(q_z3, self.prior_z.expand_as(q_z3)) / batch_size
    z4_kl = KL_Normals(q_z4, self.prior_z.expand_as(q_z4)) / batch_size

    # [batch_size * mc_samples, dim_z]
    z_sample = torch.cat([q_z.sample() for _ in range(mc_samples)], dim=0)
    z1_sample = torch.cat([q_z1.sample() for _ in range(mc_samples)], dim=0)
    z2_sample = torch.cat([q_z2.sample() for _ in range(mc_samples)], dim=0)
    z3_sample = torch.cat([q_z3.sample() for _ in range(mc_samples)], dim=0)
    z4_sample = torch.cat([q_z4.sample() for _ in range(mc_samples)], dim=0)
    
    
    zc = [z1_sample, z2_sample, z3_sample, z4_sample]
    
    #W = self.generative_model.W
    Wa = self.generative_model.Wa
    Wb = self.generative_model.Wb
    Wc = self.generative_model.Wc
    Wd = self.generative_model.Wd
    #print(z_sample.size())
    Wc = [Wa,Wb,Wc,Wd]
    
    Xrep = Variable(X.data.repeat(mc_samples, 1))
    loglik_term = (
      self.generative_model(z_sample, zc, Wc).logprob(Xrep)
      / mc_samples
      / batch_size
    )

    # Prior over the weights of the group generative nets.
    logprob_theta = self.prior_theta.logprob(self.generative_model)
 
    # Prior over the first layer Ws in the generative model.
    logprob_W = -self.lam * self.generative_model.group_lasso_penalty()
    logprob_Wa = -self.lam * self.generative_model.group_lasso_penaltypa()
    logprob_Wb = -self.lam * self.generative_model.group_lasso_penaltypb()
    logprob_Wc = -self.lam * self.generative_model.group_lasso_penaltypc()
    logprob_Wd = -self.lam * self.generative_model.group_lasso_penaltypd()

    # Proximal gradient descent requires differentiating through only the
    # non-group lasso terms, hence the separation between the loss
    # (differentiated) and the ELBO (not differentiated).
    loss = -1.0 * (- z_kl - z1_kl - z2_kl  + loglik_term + logprob_theta)
    logprob_Ws = logprob_Wa + logprob_Wb + logprob_Wc + logprob_Wd
    elbo = -loss + logprob_W  + logprob_Ws

    for opt in self.optimizers:
      opt.zero_grad()
    loss.backward()
    for opt in self.optimizers:
      opt.step()
    if self.lam > 0:
      self.generative_model.proximal_step(prox_step_size)
      self.generative_model.proximal_stepWa(prox_step_size)
      self.generative_model.proximal_stepWb(prox_step_size)
      self.generative_model.proximal_stepWc(prox_step_size)
      self.generative_model.proximal_stepWd(prox_step_size)
      

    return {
      'q_z': q_z,
      'z_kl': z_kl,
      'z1_kl': z1_kl,
      'z2_kl': z2_kl,
      'z_sample': z_sample,
      'z1_sample': z1_sample,
      'z2_sample': z2_sample,
      'z3_sample': z3_sample,
      'z4_sample': z4_sample,
      'loglike': loglik_term,
      'logprob_theta': logprob_theta,
      'logprob_W': logprob_W,
      'logprob_Ws': logprob_Ws,
      'loss': loss,
      'elbo': elbo
    }