import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib
matplotlib.use('agg')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=False)
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default='nnr/')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
    
def loadCSVfile2():
    tmp = np.loadtxt("UK.csv", dtype=np.str, delimiter=",")
    total_len = 920
    timestep = 48
    samp_trajs = tmp[0:timestep]
    for num in range(1,total_len):
        samp_trajs = np.vstack((samp_trajs,tmp[num:num+timestep]))
    samp_trajs = np.reshape(samp_trajs, (total_len,timestep,3))
    samp_trajs = samp_trajs.astype(np.float)
    samp_ts = np.linspace(0.0, timestep-1 , num=timestep)
    return samp_trajs,samp_ts,total_len,timestep

class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, nhidden)
        self.fc4 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        out = self.elu(out)
        out = self.fc4(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=20, obs_dim=3, nhidden=25, nbatch=920):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        # self.i2o = nn.Linear(obs_dim + nhidden, latent_dim * 2)
        self.h2h = nn.Linear(nhidden, nhidden)
        self.i2o = nn.Linear(nhidden, latent_dim * 2)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        combined = torch.cat((x, h), 1)
        h1 = self.i2h(combined)
        h1 = self.relu(h1)
        h = self.h2h(h1)
        h = self.relu(h)
        out = self.i2o(h1)
        out = self.softmax(out)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

if __name__ == '__main__':
    latent_dim = 100
    nhidden = 125
    rnn_nhidden = 125
    obs_dim = 3
    pred_dim = 1
    '''
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')'''
    device = torch.device('cpu')

    samp_trajs,samp_ts,nbatch,t_len = loadCSVfile2()
    
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)
    
    # model
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nbatch).to(device)
    dec = Decoder(latent_dim, pred_dim, nhidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    
    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            samp_trajs = checkpoint['samp_trajs']
            samp_ts = checkpoint['samp_ts']
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:,t,:]               
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
            pred_x = dec(pred_z)

            # compute loss
            noise_std_ = torch.ones(pred_x.size()).to(device)
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(
                torch.reshape(samp_trajs[:,:,0],(nbatch,t_len,1)), pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            loss.backward()
            optimizer.step()
            
            print('Iter: {}, running and MSE = {}/-logpx:{}/KL:{}'.format(itr, loss.data,-logpx.sum(-1).data,analytic_kl.sum(-1).sum(-1).data))
            
        tmp=pred_x
        tmp=tmp.detach().numpy()
        tmp=np.reshape(tmp,(nbatch,t_len))
        # np.savetxt('pred.csv', tmp, delimiter = ',')
        tmp2=samp_trajs
        tmp2=tmp2.detach().numpy()
        tmp2=np.reshape(tmp2[:,:,0],(nbatch,t_len))
        np.savetxt('ref.csv', tmp2, delimiter = ',')
        np.savetxt('err.csv', tmp-tmp2, delimiter = ',')
            
    except KeyboardInterrupt:
        print('Interrupted!')
        
    if args.train_dir is not None:
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        torch.save({
            'func_state_dict': func.state_dict(),
            'rec_state_dict': rec.state_dict(),
            'dec_state_dict': dec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'samp_trajs': samp_trajs,
            'samp_ts': samp_ts,
        }, ckpt_path)
        print('Stored ckpt at {}'.format(ckpt_path))
    
    print('Training complete after {} iters.'.format(itr))
    