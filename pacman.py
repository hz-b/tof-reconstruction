import numpy as np
from numpy import arange, pi, linspace, cos, zeros, array, sin, einsum, where, append, mean, diff, digitize, sum
from scipy.optimize import minimize
import math
import time
import pickle
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch

class PacMan(torch.nn.Module):
    def __init__(self, kickmin=0, kickmax=100, kickstep=0.5, tof_count=16):
        super().__init__()
        self.tof_count = tof_count
        self.kickg = arange(kickmin,kickmax,kickstep)
        self.phaseg =  linspace(0,2*pi,80)
        self.basis = self.basis()
        self.padding = 0

    def forward(self, img, its_override=None):
        img_cpu = img.cpu()
        output_list = []
        histo_list = []
        res_list = []
        for i in img_cpu:
            image, histo, res = self.overall_pacman(i.reshape(-1, self.tof_count).numpy(), self.basis, self.kickg, its_override=its_override)
            output_list.append(torch.tensor(image, device=img.device))
            histo_list.append(torch.tensor(histo, device=img.device).T)
            res_list.append(torch.tensor(res, device=img.device))

        return torch.stack(
            output_list,
            dim=0,
        ), torch.stack(
            histo_list,
            dim=0,
        ), torch.stack(
            res_list,
            dim=0,
        )
        
    def basis(self, gaussianwdt = 1):
        def gauss(x, x0, xw): 
            return np.exp(-((x - x0) ** 2) / 2 / (xw / 2.35) ** 2)
                
        dangle = np.array([math.radians(22.5*i) for i in range(16)])
        ENERGY_STEPS = 60
        ea=np.array(range(ENERGY_STEPS))#L_ea
    
        keg = arange(0, 60, gaussianwdt) 
        
        ell_a=0.73
        ell_b=1
        ell_tilt=(90-22.5)/360*2*pi

        ef = lambda phase: (ell_a*ell_b)**2 / ((ell_a*cos(phase-ell_tilt))**2+(ell_b*sin(phase-ell_tilt))**2)
        sine = lambda ke, kick, phase: ke+kick*cos(dangle-phase)*ef(phase)
        A=1
        tilt = pi
        angdist = A*(0.5-0.5*cos(2*(dangle-tilt)))
        
        lw = gaussianwdt
        sim= lambda ke, kick, phase: \
           angdist*array([gauss(ea, en, lw) for en in sine(ke,kick, phase)]).T
        basis = zeros((len(self.kickg), len(keg), len(self.phaseg),len(ea),len(dangle)))
        
        print('basis:',basis.shape)
        
        for ic,kick in enumerate(tqdm(self.kickg, leave=False, desc="Generating basis")):
            for ik,ke in enumerate(keg):
                for ip,phase in enumerate(self.phaseg):
                    basis[ic,ik,ip] = sim(ke,kick,phase)
        return basis
    def redbmkr(self, kickvalue):
        kickg = self.kickg
        phaseg = self.phaseg
        basis = self.basis
        kickbins = append(kickg-mean(diff(kickg)),kickg[-1]+mean(diff(kickg)))
        kicklineind = digitize(kickvalue,kickbins)-1-1
        redb = basis[kicklineind,:,arange(phaseg.shape[0]),:,:]
        return redb
    
    def pacman(self, acge, redbasis, its=100):
        acg = acge.copy()
        for ii in arange(its):
            brcoef = einsum('jklm,lm->jk', redbasis, acg)
            idx = tuple(array(where(brcoef==brcoef.max()))[:,0])
            acg+=-redbasis[idx]
            acg[acg<0]=0
        return sum(acg**2)
        
    def costfunc(self, x, acg, its=100):
        return self.pacman(acg,self.redbmkr(x),its)

    def estimate_kick(self, dsim):
        bcoef = einsum('ijklm,lm->ijk', self.basis,dsim)
        idxfirst = array(where(bcoef==bcoef.max()))[:,0]
        return self.kickg[idxfirst[0]]
        
    def optimize_kick(self, dsim, initial_kick):
        res = minimize(self.costfunc,initial_kick,args=(dsim),method='Powell', tol=1e-7)
        return res['x']
    
    def pacman_rec(self, acge, redbasis, its_fac=1.0, eat_fac=0.05, its_override=None):
        acg = acge.copy()
        its = int(acg.sum() * its_fac)
        if its_override is not None:
            its = its_override
        print(its)
        out=[]
        for ii in arange(its):
            brcoef = einsum('jklm,lm->jk', redbasis, acg)
            idx = tuple(array(where(brcoef==brcoef.max()))[:,0])
            acg+=eat_fac*-redbasis[idx]
            out.append(idx)
            acg[acg<0]=0
            #if np.mean(acg)<6: # 0.1296:#this value needs to be optimized ##6
            #    print('breaking:',ii)
            #    break
        return out, acg
    
    def rec_to_image(self, rec, optimized_kick):
        image=np.zeros((60,16))
        histo=np.zeros((80,60))
        for idx in rec:
            image+=self.redbmkr(optimized_kick)[idx]
            histo[idx]+=1
        return image, histo
    
    def plot_pacman(dsim, image, histo, res):
        fig,ax=plt.subplots(1,4,figsize=(10,5),sharex=False,sharey=True)
        ax[0].imshow(dsim,aspect='auto')
        ax[1].imshow(image,aspect='auto')
        ax[2].imshow(np.transpose(histo),aspect='auto',cmap='nipy_spectral')
        ax[3].imshow(res,aspect='auto')
        ax[0].set_title('Raw')
        ax[1].set_title('Rec')
        ax[2].set_title('Spec')
        ax[3].set_title('Res')
        plt.tight_layout()
        plt.savefig('./raw_rec_spec.png')
        #zero padding is needed for raw!!!
    
    def overall_pacman(self, dsim, basis, kickg, its_override=None):
        initial_kick = self.estimate_kick(dsim)
        optimized_kick = self.optimize_kick(dsim, initial_kick)
        rec, res = self.pacman_rec(dsim,self.redbmkr(optimized_kick), its_override=its_override)
        image, histo = self.rec_to_image(rec, optimized_kick)
        return image, histo, res
