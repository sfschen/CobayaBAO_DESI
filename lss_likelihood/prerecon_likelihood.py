import numpy as np
import time

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.signal import argrelmax
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval

from scipy.special import spherical_jn
from scipy.integrate import simps

from spherical_bessel_transform import SphericalBesselTransform as SBT
from loginterp import loginterp
from linear_theory import*
from pnw_dst import pnw_dst

# Class to for a BAO analysis

class PreReconLikelihood(Likelihood):
    
    zfid: float
    OmM_fid: float
    rsdrag_fid: float
    
    template_fn: str
    template_nw_fn: str
    Rsmooth: float
    
    bao_sample_name: str
    bao_datfn: str
    covfn: str
    
    xmin: float
    xmax: float
    
    fourier: bool

    def initialize(self):
        """Sets up the class.""" 
        
        # Load data
        self.loadData()
        
        # If in config space set up Fourier transform
        if self.fourier == False:
            self.kint = np.logspace(-5, 2, 2000)
            self.sphr = SBT(self.kint,L=5,fourier=True,low_ring=False)
        
        # Compute necessary (cosmological) quantities for template fit
        self.Dz   = D_of_a(1/(1.+self.zfid), OmegaM=self.OmM_fid)
        self.fz   = f_of_a(1/(1.+self.zfid), OmegaM=self.OmM_fid)
        
        self.klin, self.plin = np.loadtxt(self.template_fn,unpack=True)
        self.knw, self.pnw =   np.loadtxt(self.template_nw_fn,unpack=True)
        self.pw = self.plin - self.pnw

        j0 = spherical_jn(0,self.klin*self.rsdrag_fid)
        Sk = np.exp(-0.5*(self.klin*self.Rsmooth)**2)

        sigma_bao = self.Dz**2 * simps( 2./3 * self.plin * (1-j0), x = self.klin) / (2*np.pi**2)
        
        self.Zel = {'R': self.Rsmooth,
               'fz':self.fz,\
               'klin': self.klin, 'pnw': self.Dz**2 * self.pnw, 'pw': self.Dz**2 * self.pw,\
               'sigmas': sigma_bao}
        
        print(self.Zel['sigmas'])
        

        #

    def get_requirements(self):
        
        req = {'apar': None,\
               'aperp': None}
        
        req_bao = {\
                   'B1_' + self.bao_sample_name: None,\
                   'F_' +  self.bao_sample_name: None,\
                   'M0_' + self.bao_sample_name: None,\
                   'M1_' + self.bao_sample_name: None,\
                   'M2_' + self.bao_sample_name: None,\
                   'M3_' + self.bao_sample_name: None,\
                   'M4_' + self.bao_sample_name: None,\
                   'Q0_' + self.bao_sample_name: None,\
                   'Q1_' + self.bao_sample_name: None,\
                   'Q2_' + self.bao_sample_name: None,\
                   'Q3_' + self.bao_sample_name: None,\
                   'Q4_' + self.bao_sample_name: None,\
                    }
        req = {**req, **req_bao}
            
        return(req)
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        self.thy  = self.bao_predict(self.bao_sample_name)
        self.thy_obs  =  self.bao_observe(self.thy)

        diff = self.dd - self.thy_obs
        
        chi2 = np.dot(diff,np.dot(self.cinv,diff))
        return(-0.5*chi2)
        #
        
    def loadData(self):
        """
        Loads the required data.
        
        This can either be the power spectrum or correlation function.
        
        In both cases we will call the data (x,yell): (k,Pell) or (r,xiell).
        
        """
        # First load the data

        bao_dat = np.loadtxt(self.bao_datfn)
        self.xdat = bao_dat[:,0]
        self.y0dat = bao_dat[:,1]
        self.y2dat = bao_dat[:,2]
        
        self.dd = np.concatenate( (self.y0dat, self.y2dat) )
        
        # Now load the covariance matrix.
        cov = np.loadtxt(self.covfn)
        
        # Make the errors on the entries beyond kmin, kmax infinite
        startii = 0 # this needs to shift by len(self.xdat) for each multipole
        
        xcut = (self.xdat > self.xmax) | (self.xdat < self.xmin)
            
        for i in np.nonzero(xcut)[0]:     # FS Monopole.
            ii = i + startii
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e25
            
        startii += self.xdat.size
    
        for i in np.nonzero(xcut)[0]:       # FS Quadrupole.
            ii = i + startii
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e25


        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)
        #print(self.sample_name, np.diag(self.cinv)[:10])
        
        # Finally load the window function matrix, if in Fourier space:
        if self.fourier:
            #self.matM = np.loadtxt(self.matMfn)
            self.matW = np.loadtxt(self.matWfn)
                
    def compute_bao_pkmu(self, mu_obs, bao_sample_name):
        '''
        Helper function to get P(k,mu) post-recon in RecIso.
        
        This is turned into Pkell and then Hankel transformed in the bao_predict funciton.
        '''
    
        pp = self.provider
        apar = pp.get_param('apar')
        aperp = pp.get_param('aperp')
        B1   = pp.get_param('B1_' + bao_sample_name)
        F   = pp.get_param('F_' + bao_sample_name)
        
        Zels = self.Zel
        
        f0 = Zels['fz']
        
        klin = Zels['klin']
        pnw = Zels['pnw']
        pw  = Zels['pw']
        
        #sigmadd, sigmass, sigmads_dd, sigmads_ss, sigmads_ds = Zels['sigmas']
        sigma_bao = Zels['sigmas']
        R = Zels['R']

        Sk = np.exp(-0.5*(klin*R)**2)
        
        # Our philosophy here is to take kobs = klin
        # Then we predict P(ktrue) on the klin grid, so that the final answer is
        # Pobs(kobs,mu_obs) ~ Ptrue(ktrue, mu_true) = interp( klin, Ptrue(klin, mu_true) )(ktrue)
        # (Up to a normalization that we drop.)
        # Interpolating this way avoids having to interpolate Pw and Pnw separately.
        
        F_AP = apar/aperp
        AP_fac = np.sqrt(1 + mu_obs**2 *(1./F_AP**2 - 1) )
        mu = mu_obs / F_AP / AP_fac
        ktrue = klin/aperp*AP_fac
        
        dampfac = np.exp( -0.5 * klin**2 * sigma_bao * (1 + f0*(2+f0)*mu**2) )
        ptrue =  ( 1 + B1 + F*mu**2 )**2 * (dampfac * pw + pnw)
        
        # Sum it all up and interpolate?
        pmodel = interp1d(klin, ptrue, kind='cubic', fill_value=0,bounds_error=False)(ktrue)
    
        return pmodel
    
    def bao_predict(self, bao_sample_name):
        
        pp = self.provider
        
        M0, M1, M2, M3, M4 = [pp.get_param(param_name + '_' + bao_sample_name) for param_name in ['M0','M1','M2','M3','M4']]
        Q0, Q1, Q2, Q3, Q4 = [pp.get_param(param_name + '_' + bao_sample_name) for param_name in ['Q0','Q1','Q2','Q3','Q4']]
        
        # Generate the sampling
        ngauss = 4
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        klin = self.Zel['klin']
        pknutable = np.zeros((len(nus),len(klin)))
    
        for ii, nu in enumerate(nus_calc):
            pknutable[ii,:] = self.compute_bao_pkmu(nu, bao_sample_name)
 
        pknutable[ngauss:,:] = np.flip(pknutable[0:ngauss],axis=0)
        
        p0 = 0.5 * np.sum((ws*L0)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[m0,m1,m2,m3,m4,m5]) / klin
        p2 = 2.5 * np.sum((ws*L2)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[q0,q1,q2,q3,q4,q5]) / klin
        p4 = 4.5 * np.sum((ws*L4)[:,None]*pknutable,axis=0)
        
        if self.fourier:
            p0 += polyval(klin,[M0,M1,M2,M3,M4])
            p2 += polyval(klin,[Q0,Q1,Q2,Q3,Q4])
            
            return np.array([klin,p0,p2,p4]).T
        
        else:
            # Fourier transform it
            damping = np.exp(-(self.kint/3)**2)
            #p0t = loginterp(klin, p0)(self.kint)
            #p2t = loginterp(klin, p2)(self.kint)
            
            p0t = interp1d(klin, p0, kind='cubic', bounds_error=False, fill_value=0)(self.kint)
            p2t = interp1d(klin, p2, kind='cubic', bounds_error=False, fill_value=0)(self.kint)
        
            p4t = 0 * self.kint
            
            rr0, xi0t = self.sphr.sph(0,p0t * damping)
            rr2, xi2t = self.sphr.sph(2,p2t * damping); xi2t *= -1
            #rr2, xi4t = sphr.sph(4,p4t)
            xi4t = 0 * rr0 # no hexadecapole to speed things up
        
            xi0t += polyval(100/rr0,[M0,M1,M2,M3,M4])
            xi2t += polyval(100/rr0,[Q0,Q1,Q2,Q3,Q4])
        
            return np.array([rr0,xi0t,xi2t,xi4t]).T
        
        
        
        
    def bao_observe(self,tt):
        """Apply the window function matrix to get the binned prediction."""
        
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        
        if self.fourier:
        
            kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005
            zeros = np.zeros_like(kv)
            thy0 = Spline(tt[:,0],tt[:,1],ext=3)(kv)
            thy2 = Spline(tt[:,0],tt[:,2],ext=3)(kv)
            thy4 = Spline(tt[:,0],tt[:,3],ext=3)(kv)
        
            # wide angle (none for now)
            #expanded_model = np.matmul(self.matM, thy )
            expanded_model = np.concatenate( (thy0,zeros,thy2,zeros,thy4) )
            # Convolve with window (true) −> (conv) see eq. 2.18
            convolved_model = np.matmul(self.matW, expanded_model )
        
            # Take out the monopole and quadrupole and tap on k in data that exceed
            # the window matrix entries (which end at 0.4)
            self.p0conv = np.concatenate( (convolved_model[:40],self.p0dat[40:]) )
            self.p2conv = np.concatenate( (convolved_model[80:120],self.p2dat[40:]) )
            
            return np.concatenate( (self.p0conv,self.p2conv) )
        
        else:
            # If config space just need to average over r
            dx = self.xdat[1] = self.xdat[0]
            
            thy0 = Spline(tt[:,0],tt[:,1],ext='extrapolate')
            thy2 = Spline(tt[:,0],tt[:,2],ext='extrapolate')
            thy4 = Spline(tt[:,0],tt[:,3],ext='extrapolate')
            
            tmp0 = np.zeros_like(self.xdat)
            tmp2 = np.zeros_like(self.xdat)
            tmp4 = np.zeros_like(self.xdat)
            
            for i in range(self.xdat.size):
                kl = self.xdat[i]-dx/2
                kr = self.xdat[i]+dx/2

                ss = np.linspace(kl, kr, 100)
                p0     = thy0(ss)
                tmp0[i]= np.trapz(ss**2*p0,x=ss)*3/(kr**3-kl**3)
                p2     = thy2(ss)
                tmp2[i]= np.trapz(ss**2*p2,x=ss)*3/(kr**3-kl**3)
                p4     = thy4(ss)
                tmp4[i]= np.trapz(ss**2*p4,x=ss)*3/(kr**3-kl**3)
                
            return np.concatenate( (tmp0, tmp2) )
    
        
    
    def bao_observe_ideal(self,tt):
        
        thy =                     Spline(tt[:,0],tt[:,1],ext=3)(self.kdat)
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(self.kdat)])
        #thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3],ext=3)(self.kdat)])
        
        return thy

    

