#************************************************************
#basic cosmology tools built around astropy.cosmology
#uses Planck 2013 parameters
#************************************************************
import numpy as np
import copy
from astropy.cosmology import Planck13 as pl13
pi=np.pi
c=299792458. #speed of light
f21=1.42040575177e9 #21 cm frequency
DH=c/(pl13.H0.value*1e3)#hubble distance Mpc
DHcm=DH*3.08567758e24#hubble distance in cm
kb=1.38e-23
jy=1e-26
mH=1.6726231e-24
#pl13.Ob0=0.022068/(pl13.H0.value/1e2)**2.
print pl13.Ob0



def H0_s():
    return pl13.H0_s

def get_crit():
    return pl13.critical_density(0)

#in g cm^-3
def baryon_density(z):
    return get_crit()*(1+z)**3.*pl13.Ob0

#in cm^-3
def baryon_n(z):
    return baryon_density(z)/mH

def f2z(f):
    return f21/f-1

def z2f(z):
    return f21/(z+1)

def de(z):
    return 1./np.sqrt(pl13.Om0*(1+z)**3.+1.-pl13.Om0)

def dr2df(dr,z):
    return dr/(DH*(1+z)**2.*de(z))*f21

def df2dr(df,z):
    return -df*(DH*(1+z)**2.*de(z))/f21

def cMpc2deg(d,z):
    return pl13.arcsec_per_kpc_comoving(z).value*1e3/3600.*d
def cMpc2rad(d,z):
    return np.radians(cMpc2deg(d,z))
    

def u2kperp(u,z):
    return u*2.*pi/pl13.comoving_distance(z).value

def eta2kpara(eta,z):
    return (2.*pi*f21)/(DH*(1.+z)**2.*de(z))*eta

def kpara2eta(kpara,z):
    return DH*(1.+z)**2.*de(z)/(2.*pi*f21)*kpara

def kperp2u(kperp,z):
    return kperp/(2.*pi)*pl13.comoving_distance(z).value
 

#convert from temperature to surface brightness
def t2i(f,t):
    return 2*(f/c)**2.*kb*t/jy

def X(f):
    z=f21/f-1
    return pl13.comoving_distance(z).value
def Y(f):
    z=f21/f-1
    return DH*(1+z)*(1+z)*de(z)/f21

#convert from temperature pixel (K) to flux (Jy)
#opix is solid pixel angle (in sr)
def t2s(opix,f,t):
    return opix*t2i(f,t)

#convert from surface to temperature to surface brightness in jy
def i2t(f,i):
    return i*(c/f)**2/(2*kb)*jy

#convert temperature data cube to flux data cube
def t2s_cube(opix,faxis,tcube):
    _,fcube,_=np.meshgrid(range(tcube.shape[1]),faxis,range(tcube.shape[2]))
    return t2s(opix,fcube,tcube)

def t2i_cube(faxis,tcube):
    _,fcube,_=np.meshgrid(range(tcube.shape[1]),faxis,range(tcube.shape[2]))
    return t2i(fcube,tcube)

def maxdelay(u,maxtheta,z):
    return np.sin(np.radians(maxtheta))*(X(z)/DH/(1+z)/de(z))*u/z2f(z)
def maxkpara(kperp,maxtheta,z):
    return eta2kpara(maxdelay(kperp2u(kperp,z),maxtheta,z),z)
    

#convert from redshift to observed frequency
def z2f(z):
    return f21/(1.+z)

#convert from delta z to a delta f
def dz2df(dz,z):
    return -f21*dz/(1+z)**2.
    

#convert a flux cube into a temperature cube
#freqaxis is list of frequency values (Hz)
#dcube is the data cube to convert
#opix: solid angle of pixel
#freqaxisnum: axis that corresponds to frequency in cube
def flux2temp(freqaxis,dcube,opix):
    nf=dcube.shape[0]
    output=copy.deepcopy(dcube)
    for mm in range(nf):
        kfactor=jy*((c/freqaxis[mm])**2.)/(2.*kb)/np.abs(opix)
        output[mm,:,:]=output[mm,:,:]*kfactor
    return output
        
def wedge(z):
    return X(z2f(z))/de(z)/(1+z)/DH
