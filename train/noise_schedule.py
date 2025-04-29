import torch

def lin_beta_schedule(T, beta_start=1e-4, beta_end=0.02):

  beta_schedule = torch.linspace(beta_start,beta_end,T)
  return beta_schedule