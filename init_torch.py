import torch
import torch.nn as nn
import torch.nn.functional as F

def init_eig_magnitude(r_min=0.0, r_max=1.0):
    def init(shape):
        u = torch.rand(shape)
        nu_log = torch.log(-0.5 * torch.log(u * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        return nu_log
    return init

def init_eig_phase(max_phase):
    def init(shape):
        u = torch.rand(shape)
        theta_log = torch.log(u * max_phase)
        return theta_log
    return init

def init_gamma_log(diag_lambda):
    gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
    return gamma_log

