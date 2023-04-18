import numpy as np
import time
import yaml

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

class ZeldovichBAOLikelihood(Likelihood):
    
    zfid: float
    OmM_fid: float
    rsdrag_fid: float
    
    # optimize: turn this on when optimizng/running the minimizer so that the Jacobian factor isn't included
    # include_priors: this decides whether the marginalized parameter priors are included (should be yes)
    linear_param_dict_fn: str
    optimize: bool
    include_priors: bool
    
    template_fn: str
    template_nw_fn: str
    Rsmooth: float
    
    bao_sample_name: str
    bao_datfn: str
    covfn: str
    
    xmin: float
    xmax: float
    
    recon_mode: str
    fourier: bool
    ideal: bool
    window_kin_fn: str
    window_fn: str
    wide_angle_fn: str

    def initialize(self):
        """Sets up the class.""" 
        
        # Load the linear parameters of the theory model theta_a such that
        # P_th = P_{th,nl} + theta_a P^a for some templates P^a we will compute
        self.templates = None
        self.linear_param_dict = yaml.load(open(self.linear_param_dict_fn), Loader=yaml.SafeLoader)
        self.Nlin = len(self.linear_param_dict) 

        self.linear_param_names = list(self.linear_param_dict.keys())
        self.linear_param_means = {key: self.linear_param_dict[key]['mean'] for key in self.linear_param_dict.keys()}
        self.linear_param_stds  = np.array([self.linear_param_dict[key]['std'] for key in self.linear_param_dict.keys()])
        self.linear_param_recompute  = np.array([self.linear_param_dict[key]['recompute'] for key in self.linear_param_dict.keys()])
        self.linear_param_recompute = np.arange(self.Nlin)[self.linear_param_recompute]
        
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
        
        if self.recon_mode == 'linear':
                        
            self.Zel = {'R': self.Rsmooth,
               'fz':self.fz,\
               'klin': self.klin, 'pnw': self.Dz**2 * self.pnw, 'pw': self.Dz**2 * self.pw,}            
        
        if self.recon_mode == 'prerecon':
            
            Sigma_BAO  = self.Dz**2 * simps( 2./3 * self.plin * (1-j0), x = self.klin) / (2*np.pi**2)
            
            self.Zel = {'R': self.Rsmooth,
               'fz':self.fz,\
               'klin': self.klin, 'pnw': self.Dz**2 * self.pnw, 'pw': self.Dz**2 * self.pw,\
               'sigmas': Sigma_BAO}
        
        if self.recon_mode == 'recsym':
        
            Sk = np.exp(-0.5*(self.klin*self.Rsmooth)**2)
        
            sigmadd = self.Dz**2 * simps( 2./3 * self.plin * (1-Sk)**2 * (1-j0), x = self.klin) / (2*np.pi**2)
            sigmass = self.Dz**2 * simps( 2./3 * self.plin * (-Sk)**2 * (1-j0),  x = self.klin) / (2*np.pi**2)

            sigmads_dd = self.Dz**2 * simps( 2./3 * self.plin * (1-Sk)**2, x = self.klin) / (2*np.pi**2)
            sigmads_ss = self.Dz**2 * simps( 2./3 * self.plin * (-Sk)**2, x = self.klin) / (2*np.pi**2)
            sigmads_ds = -self.Dz**2 * simps(2./3 * self.plin * (1-Sk)*(-Sk)*j0, x = self.klin) / (2*np.pi**2) # this minus sign is because we subtract the cross term
        
            self.Zel = {'R': self.Rsmooth,
               'fz':self.fz,\
               'klin': self.klin, 'pnw': self.Dz**2 * self.pnw, 'pw': self.Dz**2 * self.pw,\
               'sigmas': (sigmadd, sigmass, sigmads_dd, sigmads_ss, sigmads_ds)}
        
        

        #

    def get_requirements(self):
        
        req = {'apar': None,\
               'aperp': None}
        
        req_amp = {\
                'A': None,\
                'f_fac': None,\
               }
        
        req_bao = {\
                   'B1_' + self.bao_sample_name: None,\
                   'F_' +  self.bao_sample_name: None,\
                   'SigmaFoG_' + self.bao_sample_name: None}
        
        req = {**req, **req_amp, **req_bao}
            
        return(req)
    
    def full_predict(self,thetas=None):
        
        thy = self.bao_predict(self.bao_sample_name, thetas=thetas)
        thy_obs = self.bao_observe(thy)
        
        return thy_obs
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        # Compute the theory prediction with lin. params. at prior mean
        #t1 = time.time()
        thy_obs_0 = self.full_predict()
        self.Delta = self.dd - thy_obs_0
        #t2 = time.time()
        
        # Now compute template
        # If template is constant for all the fits then only compute once
        
        if self.templates is not None:
            for ii in self.linear_param_recompute:
                param = self.linear_param_names[ii]
                thetas = self.linear_param_means.copy()
                thetas[param] += 1.0
                self.templates[ii,:] = self.full_predict(thetas=thetas) - thy_obs_0
                
        else:
            self.templates = np.zeros( (self.Nlin, len(self.dd)) )
            for ii, param in enumerate(self.linear_param_names):
                thetas = self.linear_param_means.copy()
                thetas[param] += 1.0
                self.templates[ii,:] = self.full_predict(thetas=thetas) - thy_obs_0
            
        #t3 = time.time()
        
        # Make dot products
        self.Va = np.dot(np.dot(self.templates, self.cinv), self.Delta)
        self.Lab = np.dot(np.dot(self.templates, self.cinv), self.templates.T) + self.include_priors * np.diag(1./self.linear_param_stds**2)
        #self.Va = np.einsum('ij,jk,k', self.templates, self.cinv, self.Delta)
        #self.Lab = np.einsum('ij,jk,lk', self.templates, self.cinv, self.templates) + np.diag(1./self.linear_param_stds**2)
        self.Lab_inv = np.linalg.inv(self.Lab)
        #t4 = time.time()
        
        # Compute the modified chi2
        lnL  = -0.5 * np.dot(self.Delta,np.dot(self.cinv,self.Delta)) # this is the "bare" lnL
        lnL +=  0.5 * np.dot(self.Va, np.dot(self.Lab_inv, self.Va)) # improvement in chi2 due to changing linear params
        if not self.optimize:
            lnL += - 0.5 * np.log( np.linalg.det(self.Lab) ) + 0.5 * self.Nlin * np.log(2*np.pi) # volume factor from the determinant
        
        #t5 = time.time()
        
        #print(t2-t1, t3-t2, t4-t3, t5-t4)
        
        return lnL

    def get_best_fit(self):
        try:
            self.p0_nl  = self.dd - self.Delta
            self.bf_thetas = np.einsum('ij,j', np.linalg.inv(self.Lab), self.Va)
            self.p0_lin = np.einsum('i,il', self.bf_thetas, self.templates)
            return self.p0_nl + self.p0_lin
        except:
            print("Make sure to first compute the posterior.")
        
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
        print(self.dd.shape)
        
        # Now load the covariance matrix.
        cov = np.loadtxt(self.covfn)
        print(cov.shape)
        
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
        if self.fourier and not self.ideal:
            self.kin  = np.loadtxt(self.window_kin_fn)
            self.matM = np.loadtxt(self.wide_angle_fn)
            self.matW = np.loadtxt(self.window_fn)
            
        if self.ideal or not self.fourier:
            self.binmat = None
                
    
    def compute_bao_pkmu(self, mu_obs, bao_sample_name):
        '''
        Helper function to get P(k,mu) post-recon in RecIso.
        
        This is turned into Pkell and then Hankel transformed in the bao_predict funciton.
        '''
    
        pp = self.provider
        apar = pp.get_param('apar')
        aperp = pp.get_param('aperp')
        
        A = pp.get_param('A')
        f_fac = pp.get_param('f_fac')
        
        B1   = pp.get_param('B1_' + bao_sample_name)
        F   = pp.get_param('F_' + bao_sample_name)
        SigmaFoG = pp.get_param('SigmaFoG_' + bao_sample_name)
        
        Zels = self.Zel
        
        f0 = Zels['fz'] * f_fac
        
        klin = Zels['klin']
        pnw = Zels['pnw']
        pw  = Zels['pw']
        
        if self.recon_mode == 'recsym':
        
            sigmadd, sigmass, sigmads_dd, sigmads_ss, sigmads_ds = Zels['sigmas']
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
        
            # First construct P_{dd,ss,ds} individually
        
            rsd_fac = 1 + f0*(2+f0)*mu**2
        
            dampfac_dd = np.exp( -0.5 * klin**2 * (A * sigmadd * rsd_fac + SigmaFoG * mu**2) )
            pdd = ( (1 + F*mu**2)*(1-Sk) + B1 )**2 * (dampfac_dd * pw + pnw)
            
            # then Pss
            dampfac_ss = np.exp( -0.5 * klin**2 * A * sigmass * rsd_fac )
            pss = Sk**2 * (1 + F*mu**2)**2 * (dampfac_ss * pw + pnw)
        
            # Finally Pds
            dampfac_ds = np.exp(-0.5 * klin**2 * A * (0.5*sigmads_dd+0.5*sigmads_ss+sigmads_ds) * rsd_fac )
            linfac = - Sk * (1+F*mu**2) * ( (1+F*mu**2)*(1-Sk) + B1 )
            pds = linfac * (dampfac_ds * pw + pnw)
        
            # Sum it all up and interpolate?
            ptrue = pdd + pss - 2*pds #* np.exp(- 0.5 * klin**2 * mu**2 * SigmaFoG**2)
            pmodel = interp1d(klin, ptrue, kind='cubic', fill_value=0,bounds_error=False)(ktrue)
            
        if self.recon_mode == 'prerecon':
            sigma_bao = Zels['sigmas']

            # Our philosophy here is to take kobs = klin
            # Then we predict P(ktrue) on the klin grid, so that the final answer is
            # Pobs(kobs,mu_obs) ~ Ptrue(ktrue, mu_true) = interp( klin, Ptrue(klin, mu_true) )(ktrue)
            # (Up to a normalization that we drop.)
            # Interpolating this way avoids having to interpolate Pw and Pnw separately.
        
            F_AP = apar/aperp
            AP_fac = np.sqrt(1 + mu_obs**2 *(1./F_AP**2 - 1) )
            mu = mu_obs / F_AP / AP_fac
            ktrue = klin/aperp*AP_fac
        
            # First construct P_{dd,ss,ds} individually
        
            rsd_fac = 1 + f0*(2+f0)*mu**2
        
            dampfac = np.exp( -0.5 * klin**2 * (A * sigma_bao * rsd_fac + SigmaFoG * mu**2) )
            pbao = ( 1 + F*mu**2 + B1 )**2 * (dampfac * pw + pnw)
            
            pmodel = interp1d(klin, pbao, kind='cubic', fill_value=0,bounds_error=False)(ktrue)
            
        if self.recon_mode == 'linear':

            # Our philosophy here is to take kobs = klin
            # Then we predict P(ktrue) on the klin grid, so that the final answer is
            # Pobs(kobs,mu_obs) ~ Ptrue(ktrue, mu_true) = interp( klin, Ptrue(klin, mu_true) )(ktrue)
            # (Up to a normalization that we drop.)
            # Interpolating this way avoids having to interpolate Pw and Pnw separately.
        
            F_AP = apar/aperp
            AP_fac = np.sqrt(1 + mu_obs**2 *(1./F_AP**2 - 1) )
            mu = mu_obs / F_AP / AP_fac
            ktrue = klin/aperp*AP_fac
        
            # This is just the Kaiser formula        
            pbao = ( 1 + F*mu**2 + B1 )**2 * (pw + pnw)
            
            pmodel = interp1d(klin, pbao, kind='cubic', fill_value=0,bounds_error=False)(ktrue)
    
        return pmodel
    
    def bao_predict(self, bao_sample_name, thetas=None):
        
        pp = self.provider
        
        if thetas is None:
            M0, M1, M2, M3, M4, M5, M6, M7 = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in ['M0','M1','M2','M3','M4','M5','M6','M7']]
            Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7 = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in ['Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7']]
        else:
            M0, M1, M2, M3, M4, M5, M6, M7 = [thetas[param_name + '_' + bao_sample_name] for param_name in ['M0','M1','M2','M3','M4','M5','M6','M7']]
            Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7 = [thetas[param_name + '_' + bao_sample_name] for param_name in ['Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7']]
        
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
            p0 += polyval(klin/0.1,[M0,M1,M2,M3,M4,M5,M6,M7]) / (klin/0.1)
            p2 += polyval(klin/0.1,[Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7]) / (klin/0.1)
            
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
        
            xi0t += polyval(100/rr0,[M0,M1,M2,M3,M4,M5,M6,M7]) * (rr0/100)
            xi2t += polyval(100/rr0,[Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7]) * (rr0/100)
        
            return np.array([rr0,xi0t,xi2t,xi4t]).T
        
        
        
        
    def bao_observe(self,tt):
        """Apply the window function matrix to get the binned prediction."""
        
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        
        if self.ideal or not self.fourier:
            return self.bao_observe_ideal(tt)
        
        elif self.fourier:
            thy0 = Spline(tt[:,0],tt[:,1],ext=1)(self.kin)
            thy2 = Spline(tt[:,0],tt[:,2],ext=1)(self.kin)
            thy4 = Spline(tt[:,0],tt[:,3],ext=1)(self.kin)
        
            # wide angle and window
            thy = np.concatenate( (thy0, thy2, thy4) )
            expanded_model = np.matmul(self.matM.T, thy )
            convolved_model = np.matmul(self.matW.T, expanded_model )
            
            self.conv = convolved_model
        
            # Read out only the monopole and quadrupole...
            Ndat = len(self.xdat)
            self.p0conv = convolved_model[:Ndat]
            self.p2conv = convolved_model[(2*Ndat):(3*Ndat)]
            
            return np.concatenate( (self.p0conv,self.p2conv) )
        
    
    def bao_observe_ideal(self,tt, matrix=True):
        
        if matrix:
            # If no binning matrix for this sample yet, make it.
            if self.binmat is None:  
                
                rdat = self.xdat
                dr = rdat[1] - rdat[0]
                
                rth = tt[:,0]
                Nvec = len(rth)

                bin_mat = np.zeros( (len(rdat), Nvec) )

                for ii in range(Nvec):
                    # Define basis vector
                    xivec = np.zeros_like(rth); xivec[ii] = 1
    
                    # Define the spline:
                    thy = Spline(rth, xivec, ext='const')
    
                    # Now compute binned basis vector:
                    tmp = np.zeros_like(rdat)
    
                    for i in range(rdat.size):
                        kl = rdat[i]-dr/2
                        kr = rdat[i]+dr/2

                        ss = np.linspace(kl, kr, 100)
                        p     = thy(ss)
                        tmp[i]= np.trapz(ss**2*p,x=ss)*3/(kr**3-kl**3)
        
                    bin_mat[:,ii] = tmp
                
                self.binmat = np.array(bin_mat)
            
            tmp0 = np.dot(self.binmat, tt[:,1])
            tmp2 = np.dot(self.binmat, tt[:,2])
            
        else:
        
            dx = self.xdat[1] = self.xdat[0]
            
            thy0 = Spline(tt[:,0],tt[:,1],ext='extrapolate')
            thy2 = Spline(tt[:,0],tt[:,2],ext='extrapolate')
            #thy4 = Spline(tt[:,0],tt[:,3],ext='extrapolate')
            
            tmp0 = np.zeros_like(self.xdat)
            tmp2 = np.zeros_like(self.xdat)
            #tmp4 = np.zeros_like(self.xdat)
            
            for i in range(self.xdat.size):
                kl = self.xdat[i]-dx/2
                kr = self.xdat[i]+dx/2

                ss = np.linspace(kl, kr, 100)
                p0     = thy0(ss)
                tmp0[i]= np.trapz(ss**2*p0,x=ss)*3/(kr**3-kl**3)
                p2     = thy2(ss)
                tmp2[i]= np.trapz(ss**2*p2,x=ss)*3/(kr**3-kl**3)
                #p4     = thy4(ss)
                #tmp4[i]= np.trapz(ss**2*p4,x=ss)*3/(kr**3-kl**3)
                
        return np.concatenate( (tmp0, tmp2) )


    

