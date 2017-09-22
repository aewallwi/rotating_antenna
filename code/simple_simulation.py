import numpy as np
import scipy.special as sp
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.signal as signal
import cosmology
from numba import jit
from joblib import Parallel, delayed
import multiprocessing
n_cpu=multiprocessing.cpu_count()
c=3e8

pi=np.pi
KAPPA=3200#6c dn/ds params hard coded
BETA=2.51
GAMMA=1.75
S0=.88
#airy beam
def airy_beam(theta,f,dant):
    output=np.ones(len(theta))
    output[theta==0]=1.
    output[theta>0.]=(2.*sp.j1(theta[theta>0.]*pi*dant*f/c)/(pi*theta[theta>0.]*dant*f/c))**2.
    return output

#smin is the minimum flux
#use 6C source counts
#draw alpha from distribution with mean 0.5 and std 0.5
def power_law(alpha,xmin,xmax,x):
    return (1-alpha)/(xmax**(-alpha+1)-xmin**(-alpha+1))*x**-alpha
#draw power law distribution between xmin and xmax with P(x) ~ x^-alpha
#with size samples
def draw_power_law(xmin,xmax,alpha,size=1):
    y=np.random.rand(size)
    return ((xmax**(-alpha+1)-xmin**(-alpha+1))*y+xmin**(-alpha+1))**(1/(-alpha+1))
def drawSources(area,smin=0,smax=np.inf):
    kappa=3200#6c dn/ds params hard coded
    beta=2.51
    gamma=1.75
    scut=.88
    if(smin>scut):
        coeff=(kappa*scut**beta*smin**(1.-beta))/(beta-1.)
        n1=int(np.random.poisson(int(area*coeff)))
        n2=0
        fluxes=draw_power_law(smin,smax,beta,size=n1)
    if(smin<scut):
        coeff1=(kappa*scut**gamma*(scut**(1.-gamma))-smin**(1-gamma))/(1-gamma)
        coeff2=kappa*scut/(beta-1.)
        n1=int(np.random.poisson(int(area*coeff1)))
        n2=int(np.random.poisson(int(area*coeff2)))
        fluxes=np.hstack([draw_power_law(scut,smax,beta,size=n2),draw_power_law(smin,scut,gamma,size=n1)])
    spectral_inds=np.random.normal(loc=-0.5,scale=0.5,size=n1+n2)
    return fluxes,spectral_inds
#lAngleList=np.array([0])
def drawRandomSources(sMin):    
    flux,spIndices=drawSources(2*np.pi,sMin)#draw sources greater than 1 Jy from 2 pi sr
    nsrcs=len(flux)
    print 'number of sources=%d'%(nsrcs)
    phiList=np.random.rand(int(nsrcs))*np.pi*2.
    thetaList=np.arccos(np.random.rand(int(nsrcs))-1)
    lAngleList=np.sin(thetaList)*np.cos(phiList)
    mAngleList=np.sin(thetaList)*np.sin(phiList)
    return np.array([lAngleList,mAngleList,flux,spIndices]).T

#    computeVisList(lAngles,mAngles,uList,vList,fluxList,output)
def drawSources(area,smin):
    kappa=3200#6c dn/ds params hard coded
    beta=2.51
    gamma=1.75
    scut=.88
    if(smin>scut):
        coeff=(kappa*scut**beta*smin**(1.-beta))/(beta-1.)
        n1=int(np.random.poisson(int(area*coeff)))
        n2=0
        fluxes=draw_power_law(smin,np.inf,gamma,size=n1)
    if(smin<scut):
        coeff1=(kappa*scut**gamma*scut**(1.-gamma))/(gamma-1.)
        coeff2=kappa*(scut-scut**beta*smin**(1.-beta))/(1.-beta)
        n1=int(np.random.poisson(int(area*coeff1)))
        n2=int(np.random.poisson(int(area*coeff2)))
        fluxes=np.hstack([draw_power_law(scut,np.inf,gamma,size=n1),draw_power_law(smin,scut,beta,size=n2)])
    spectral_inds=np.random.normal(loc=0.5,scale=0.5,size=n1+n2)
    return fluxes,spectral_inds
#lAngleList=np.array([0])
def drawRandomSources(sMin):    
    flux,spIndices=drawSources(2*np.pi,sMin)#draw sources greater than 1 Jy from 2 pi sr
    nsrcs=len(flux)
    print 'number of sources=%d'%(nsrcs)
    phiList=np.random.rand(int(nsrcs))*np.pi*2.
    thetaList=np.arccos(np.random.rand(int(nsrcs))-1)
    lAngleList=np.sin(thetaList)*np.cos(phiList)
    mAngleList=np.sin(thetaList)*np.sin(phiList)
    return np.array([lAngleList,mAngleList,flux,spIndices]).T


def unfoldCube(xGrid,yGrid,dataCube):
    nx=dataCube.shape[1]
    ny=dataCube.shape[2]
    output=np.zeros((dataCube.shape[0],nx*ny),dtype=complex)
    outputAxis=np.zeros(nx*ny)
    xyi=0
    for x in range(nx):
        for y in range(ny):
            output[:,xyi]=dataCube[:,x,y]
            outputAxis[xyi]=np.sqrt(xGrid[x,y]**2.+yGrid[x,y]**2.)
            xyi+=1
    return outputAxis,output

def gridDelay(uAIn,tAIn,ftVis,nCells,uMax):
    dGrid=np.zeros((ftVis.shape[0]/2,nCells))
    counts=np.zeros(nCells).astype(int)
    nu=ftVis.shape[1]
    nt=ftVis.shape[0]
    for uNum in range(nu):
         binNum=np.round(uAIn[uNum]/(uMax/nCells))
         if(binNum<nCells and not(np.any(np.isnan(ftVis[:,uNum]))) and not(np.any(np.isinf(ftVis[:,uNum])))):
             counts[binNum]+=1
             dGrid[:,binNum]+=np.abs(ftVis[nt/2:,uNum])**2.
             dGrid[:,binNum]+=np.abs(np.flipud(ftVis[:nt/2,uNum]))**2.
    for bNum in range(nCells):
        if(counts[bNum]>0):
            dGrid[:,bNum]/=counts[bNum]
    uAxis=(np.arange(0,nCells)+.5)*uMax/nCells
    tAxis=tAIn[nt/2:]
    return uAxis,tAxis,dGrid

@jit(nopython=True)
def gridDishKernel(nx,ny,dx,dy,posX,posY,data,rDish,griddedData,nData,nKernel):
    for mm in range(nData):
        #get data point location
        pX=posX[mm]
        pY=posY[mm]
        zX=int(np.round(pX/dx)+nx/2)
        zY=int(np.round(pY/dy)+ny/2)

        #rewrite this part as a loop over the grid items
        for ii in range(nKernel):
            for jj in range(nKernel):
                xInd=zX-nKernel/2+ii
                yInd=zY-nKernel/2+jj
                r=np.sqrt((dx*float(xInd-nx/2)-pX)**2.+(dy*float(yInd-ny/2)-pY)**2)
                if(r<=2.*rDish and xInd>=0 and xInd<nx and yInd>=0 and yInd<ny):
                    griddedData[xInd,yInd]+=2./(np.pi)*(np.arccos(.5*r/rDish)-(.5*r/rDish)*np.sqrt(1-(.5*r/rDish)**2.))*data[mm]
    return griddedData

@jit(nopython=True)
def computeVisList(lList,mList,uList,vList,sList,output):
    '''
    generate set of visibilities with u,v coordinates from list of sources with fluxes and l,m coordinates
    Assumes coplanar array
    Args:
    lList: list of e-w angle cosines of sources
    mList: list of n-s angle cosines of sources
    uList: list of e-w baseline separations in wavelengths
    vList: list of n-s baseline separations in wavelengths
    output: 
    Returns:
    array in which computed visibilities are stored
    '''
    assert(len(uList)==len(vList))
    assert(len(output)==len(uList))
    assert(len(lList)==len(mList))
    assert(len(sList)==len(mList))
    output[:]=0.
    for uvNum in range(len(uList)):
        for lmNum in range(len(lList)):
            output[uvNum]+=sList[lmNum]*np.exp(2j*pi*(uList[uvNum]*lList[lmNum]+vList[uvNum]*mList[lmNum]))
    return output



def hexShell(nside,d):
    points=[]
    if(nside>1):
        for m in range(nside):
            x=(.5+m/(2.*(nside-1.)))*d*nside
            y=np.sqrt(3.)*(x-.5*d*nside)-np.sqrt(3.)*d*nside/2.
            points.append([x,y])
            points.append([x-d*nside*3./2.,y+np.sqrt(3.)*d*nside/2.])
        for m in range(nside-1):
            x=(.5+m/(2.*(nside-1.)))*d*nside
            y=-np.sqrt(3.)*(x-.5*d*nside)+np.sqrt(3.)*d*nside/2.
            points.append([x,y])
            points.append([x-d*nside*3./2.+d*nside/(nside-1.)/2.,y-np.sqrt(3.)*d*nside/2.-d*nside/(nside-1.)*np.sqrt(3.)/2.])
        for m in range(1,nside-1):
            x=(-.5+m/(1.*nside-1.))*d*nside
            y=np.sqrt(3.)/2.*d*nside
            points.append([x,y])
            points.append([x,y-d*nside*np.sqrt(3.)])
    elif(nside==0):
        return np.array([[0,0]])
    elif(nside==1):
        return d*np.array([[-.5,np.sqrt(3.)/2.],[.5,np.sqrt(3.)/2.],[-1.,0.],[1.,0.],[-.5,-np.sqrt(3.)/2.],[.5,-np.sqrt(3.)/2.]])
    return np.array(points)
def hexPack(nshell,d):
    output=hexShell(0,d)
    for m in range(2,nshell+1):
        output=np.vstack([output,hexShell(m,(1.-1./m)*d)])
    return output

'''
simple simulation class allows me to create hex packed interferometer arrays, generate source lists, visibilities etc...
'''

class HexDishArray():
    def __init__(self,nSide,dAnt,dAntPos,nDays=1,df=100e3,f0=150e6,b=20e6,parallel=True):
        self.dAntPos=dAntPos
        self.positions=[hexPack(nSide,dAntPos)]
        self.perturbation=0.
        self.nDays=nDays
        self.f0=f0
        self.df=df
        self.band=b
        self.dAnt=dAnt
        self.nAnt=len(self.positions[0])
        self.nVis=((self.nAnt-1)*self.nAnt)/2
        self.nChan=int(b/df)
        self.fAxis=f0+df*np.arange(-self.nChan/2,self.nChan/2)
        self.srcList=np.array([[0,0,0,0]])#power law sources are stored as l,m,s150,alpha
        self.separations=np.zeros((self.nVis,2))
        self.setGridResolution(0.5)#initialize data cube parameters to be 0.5 wavelengths
        self.visCube=np.zeros((self.nChan,self.nVis),dtype=np.complex)
        self.dataCube=np.zeros((self.nChan,self.gridSize,self.gridSize),dtype=np.complex)
        self.sampleCube=np.zeros_like(self.dataCube)
        self.parallel=parallel
 
    def _computePairSep(self,dayNum=0):
        visNum=0
        for i in range(self.nAnt):
           for j in range(i):
               self.separations[visNum]=self.positions[dayNum][i]-self.positions[dayNum][j]
               visNum+=1           
    def _computeVisibilities(self,dayNum=0):
        self._computePairSep(dayNum)
        #make embarrasingly parallel
        if(not(self.parallel)):
            for cNum in range(self.nChan):
                wL=c/self.fAxis[cNum]
                gains=airy_beam(np.linalg.norm(self.srcList[:,:2],axis=1),self.fAxis[cNum],self.dAnt)
                self.visCube[cNum,:]=computeVisList(self.srcList[:,0],self.srcList[:,1],self.separations[:,0]/wL,self.separations[:,1]/wL,gains*self.srcList[:,2]*(self.fAxis[cNum]/150e6)**self.srcList[:,3],self.visCube[cNum,:])
        else:
            #parallelize over channels
            self.visCube=np.array(Parallel(n_jobs=n_cpu/2)(delayed(computeVisList)(self.srcList[:,0],self.srcList[:,1],self.separations[:,0]*self.fAxis[i]/c,self.separations[:,1]*self.fAxis[i]/c,airy_beam(np.linalg.norm(self.srcList[:,:2],axis=1),self.fAxis[i],self.dAnt)*self.srcList[:,2]*(self.fAxis[i]/150e6)**self.srcList[:,3],self.visCube[i,:]) for i in range(self.nChan)))

#    def _computeVisibilities(self,dayNum=0):
#        self._computePairSep(dayNum)
#        for cNum in range(self.nChan):
#            wL=c/self.fAxis[cNum]
#            gains=airy_beam(np.sqrt(self.srcList[:,0]**2.+self.srcList[:,1]**2.),self.fAxis[cNum],self.dAnt)
#            self.visCube[cNum,:]=computeVisList(self.srcList[:,0],self.srcList[:,1],self.separations[:,0]/wL,self.separations[:,1]/wL,gains*self.srcList[:,2]*(self.fAxis[cNum]/150e6)**self.srcList[:,3],self.visCube[cNum,:])
    def _addGridVisibilities(self):
        for cNum in range(self.nChan):
            wL=c/self.fAxis[cNum]
            rAnt=self.dAnt/2./wL
            nKernel=int(np.ceil(8*rAnt/self.gridSpacing))
            self.dataCube[cNum,:,:]+=gridDishKernel(self.gridSize,self.gridSize,self.gridSpacing,self.gridSpacing,self.separations[:,0]/wL,self.separations[:,1]/wL,self.visCube[cNum,:],rAnt,np.zeros(self.dataCube[cNum,:,:].shape,dtype=complex),self.nVis,nKernel)
            self.sampleCube[cNum,:,:]+=gridDishKernel(self.gridSize,self.gridSize,self.gridSpacing,self.gridSpacing,self.separations[:,0]/wL,self.separations[:,1]/wL,np.ones(self.nVis,dtype=complex),rAnt,np.zeros(self.dataCube[cNum,:,:].shape,dtype=complex),self.nVis,nKernel)
    def _resetGrid(self):
        self.dataCube[:]=0.
        self.sampleCube[:]=0.
    def _addDay(self,dNum):
        self._computeVisibilities(dNum)
        self._addGridVisibilities()
    '''
    Public Methods
    '''
    def gridDays(self,startDay=0,endDay=1):
        self._resetGrid()
        for dayNum in range(startDay,endDay):
            self._addDay(dayNum)
    def setGridResolution(self,dGrid):
        self.gridSpacing=dGrid
        maxSep=0.
        #generate global maximum uv spacing
        self._computePairSep(0)
        maxUV=np.linalg.norm(self.separations,axis=1).max()/c*self.fAxis.max()
        self.maxUV=maxUV
        self.gridSize=2*int(maxUV/dGrid)
        self.dataCube=np.zeros((self.nChan,self.gridSize,self.gridSize),dtype=np.complex)
        self.sampleCube=np.zeros((self.nChan,self.gridSize,self.gridSize),dtype=np.complex)
    def perturbPositions(self,dAp,nDays=1):
        self.nDays=nDays
        if(len(self.positions)>1):
            del self.positions[1:]
        self.perturbation=dAp
        for dayN in range(self.nDays):
            self.positions.append(self.positions[0]-dAp*np.ones(self.positions[0].shape)+2.*dAp*np.random.rand(*(self.positions[0].shape)))
    def setSourceList(self,srcList):
        self.srcList=srcList
    def getDataCube(self):
        return self.dataCube
    def getSampleCube(self):
        return self.sampleCube
    def getUniformData(self):
        return self.dataCube/self.sampleCube
    def getFourierTransformCube(self):
        windowCube=np.zeros_like(self.dataCube)
        for i in range(windowCube.shape[1]):
            for j in range(windowCube.shape[2]):
                windowCube[:,i,j]=signal.blackmanharris(self.nChan)
                windowCube[:,i,j]/=np.sqrt(np.mean(windowCube[:,i,j]**2.))
        ftTemp=self.dataCube*windowCube/self.sampleCube
        ftTemp[np.logical_or(np.isnan(ftTemp),np.isinf(ftTemp))]=0.
        return fft.fftshift(fft.fft(fft.fftshift(ftTemp,axes=[0]),axis=0),axes=[0])
    def getDelayTransformVisibilities(self,dNum):
        self._computeVisibilities(dNum)
        windowGrid=np.zeros_like(self.visCube)
        for vNum in range(self.nVis):
            windowGrid[:,vNum]=signal.blackmanharris(self.nChan)
            windowGrid[:,vNum]/=np.sqrt(np.mean(windowGrid[:,vNum]**2.))
        return fft.fftshift(fft.fft(fft.fftshift(self.visCube*windowGrid,axes=[0]),axis=0),axes=[0])
    
    def getDVisGrid(self,nCells=20,dNum=0):
        visCube=self.getDelayTransformVisibilities(dNum)
        uvAxis=np.linalg.norm(self.separations*self.fAxis[self.nChan/2]/c,axis=1)
        tAxis=np.arange(-self.nChan/2,self.nChan/2)/self.band
        return gridDelay(uvAxis,tAxis,visCube,nCells,self.maxUV)
        
    def getDCubeGrid(self,nCells=20):
        dataCube=self.getFourierTransformCube()
        uA=self.gridSpacing*np.arange(-self.gridSize/2,self.gridSize/2)
        uGrid,vGrid=np.meshgrid(uA,uA)
        uvAxis,dataCube=unfoldCube(uGrid,vGrid,dataCube)
        tAxis=np.arange(-self.nChan/2,self.nChan/2)/self.band
        return gridDelay(uvAxis,tAxis,dataCube,nCells,self.maxUV)
    
    def getCosmologyDCubeGrid(self,nCells=20):
        uAxis,tAxis,dGrid=self.getDCubeGrid(nCells)
        #convert dGrid to power spectrum
        sigma=c/self.dAnt/self.f0*.45
        dGrid=1e6*self.df*self.df*dGrid*cosmology.X(self.f0)**2.*cosmology.Y(self.f0)/(self.band*pi*sigma**2.)*cosmology.i2t(self.f0,1)**2.
        return cosmology.u2kperp(uAxis,cosmology.f2z(self.f0)),cosmology.eta2kpara(tAxis,cosmology.f2z(self.f0)),dGrid
    
    def getCosmologyDVisGrid(self,nCells=20):
        uAxis,tAxis,dGrid=self.getDVisGrid(nCells)
        #convert dGrid to cosmology power spectrum
        sigma=c/self.dAnt/self.f0*.45
        dGrid=1e6*self.df*self.df*dGrid*cosmology.X(self.f0)**2.*cosmology.Y(self.f0)/(self.band*pi*sigma**2.)*cosmology.i2t(self.f0,1)**2.
        return cosmology.u2kperp(uAxis,cosmology.f2z(self.f0)),cosmology.eta2kpara(tAxis,cosmology.f2z(self.f0)),dGrid

    def getNaturallyWeightedImage(self,nCells=20):
        return np.real(fft.fftshift(fft.fft2(fft.fftshift(self.dataCube,axes=[1,2]),axes=[1,2]),axes=[1,2]))
    def getUniformWeightedImage(self,nCells=20):
        tempRatio=self.dataCube/self.sampleCube
        tempRatio[np.logical_or(np.isnan(tempRatio),np.isinf(tempRatio))]=0.
        return np.real(fft.fftshift(fft.fft2(fft.fftshift(tempRatio,axes=[1,2]),axes=[1,2]),axes=[1,2]))
                

        
        
                
        
        

                                            
    

        


    
