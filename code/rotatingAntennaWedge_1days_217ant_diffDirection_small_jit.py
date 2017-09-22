#import matplotlib as mpl
#mpl.use('Agg')
import sys
sys.settrace
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.signal as signal
import cosmology
from numba import jit
#import time
c=3e8
pi=np.pi

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

def gridPack(nx,ny,d):
    output=np.zeros((nx,ny))
    jj=0
    for mm in range(nx):
        for nn in range(ny):
            output[jj,0]=mm*d
            output[jj,1]=nn*d
def perturbPositions(aP,dAp):
    return aP-dAp+2.*dAp*np.random.rand(*(dAp.shape))

def dishKernel(r,R):
    if(len(r)>1):
        output=np.zeros(r.shape)
        selection=r<=R
        output[selection]+=2./(np.pi)*(np.arccos(r[selection]/R)-(r[selection]/R)*np.sqrt(1-(r[selection]/R)**2.))
        return output
    else:
        return 2./(np.pi)*(np.arccos(r/R)-(r/R)*np.sqrt(1-(r/R)**2.))

def grid(nx,ny,dx,dy,posX,posY,data,kernelFunction,kernelSize=None):
    if(kernelSize is None):
        kernelSize=max(nx,ny)
    griddedData=np.zeros((nx,ny))
    nData=len(data)
    xGrid,yGrid=np.meshgrid(np.array(range(nx))*dx,np.array(range(ny))*dy)
    xGrid-=ny*dy/2.
    yGrid-=nx*dx/2.
    #@jit
    #def gridLoop():
    for mm in range(nData):
        #identify points within kernel
        dX=posX[mm]-xGrid
        dY=posY[mm]-yGrid
        pointsToAdd=np.logical_and(np.abs(dX/dx)<=kernelSize/2,np.abs(dY/dy)<=kernelSize/2.)
        griddedData[pointsToAdd]+=kernelFunction(dX[pointsToAdd],dY[pointsToAdd])*data[mm]
        dX=-posX[mm]-xGrid
        dY=-posY[mm]-yGrid
        pointsToAdd=np.logical_and(np.abs(dX/dx)<=kernelSize/2,np.abs(dY/dy)<=kernelSize/2.)
        griddedData[pointsToAdd]+=kernelFunction(dX[pointsToAdd],dY[pointsToAdd])*data[mm]
    #gridLoop()




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
                if(r<=rDish and xInd>=0 and xInd<nx and yInd>=0 and yInd<ny):
                    griddedData[xInd,yInd]+=2./(np.pi)*(np.arccos(r/rDish)-(r/rDish)*np.sqrt(1-(r/rDish)**2.))*data[mm]

    return griddedData

def airyDisk(l,m,waveLength,d):
    x=2.*pi/waveLength*d*np.sqrt(l**2.+m**2.)
    return (2.*sp.jn(1,x)/x)**2.
@jit(nopython=True)
def computeVisList(lList,mList,uList,vList,sList,output):
    output[:]=0.
    for uvNum in range(len(uList)):
        for lmNum in range(len(lList)):
            output[uvNum]+=sList[lmNum]*np.exp(2j*pi*(uList[uvNum]*lList[lmNum]+vList[uvNum]*mList[lmNum]))
    return output
#generate array
dAnt=1.
dAntPos=3.
antPos=hexPack(9,dAntPos)
nDish=len(antPos)


def delayGrid(dataCube,du,uMax,nCells):
    dGrid=np.zeros((dataCube.shape[-1]/2,nCells),dtype=complex)
    counts=np.zeros(nCells).astype(int)
    for uNum in range(dataCube.shape[0]):
        for vNum in range(dataCube.shape[1]):
            uVal=du*(uNum-dataCube.shape[0]/2)
            vVal=du*(vNum-dataCube.shape[1]/2)
            uvVal=np.sqrt(uVal**2.+vVal**2.)
            binNum=np.round(uvVal/(uMax/nCells))
            if(binNum<nCells and not(np.any(np.isnan(dataCube[uNum,vNum,:])))):
                counts[binNum]+=1
                dGrid[:,binNum]+=dataCube[uNum,vNum,dataCube.shape[-1]/2:]
                dGrid[:,binNum]+=np.flipud(dataCube[uNum,vNum,:dataCube.shape[-1]/2])
    for bNum in range(nCells):
        if(counts[bNum]>0):
            dGrid[:,bNum]/=counts[bNum]
    return dGrid



nVis=nDish*(nDish-1)/2
antDiff=np.zeros((nVis,2))
antDiffP=np.zeros_like(antDiff)
dIndex=0
nChan=200
f0=150e6
df=100e3
fAxis=f0+df*np.arange(-nChan/2,nChan/2)


for iDish in range(nDish):
    for jDish in range(iDish):
        antDiff[dIndex,0]=antPos[iDish,0]-antPos[jDish,0]
        antDiff[dIndex,1]=antPos[iDish,1]-antPos[jDish,1]
        dIndex+=1

uvAmp=np.sqrt(np.sum((antDiff/(c/fAxis.max()))**2.,axis=1))
maxUV=uvAmp.max()
gridSize=int(np.ceil(4.*maxUV))
if(np.mod(gridSize,2)==1):
    gridSize+=1



uvCube=np.zeros((gridSize,gridSize,nChan),dtype=complex)
sampleCube=np.zeros((gridSize,gridSize,nChan),dtype=complex)

lAngleList=np.array([0.9])
mAngleList=np.array([0.])
fluxList=np.array([1])

nDays=1

visList=np.zeros((nVis),dtype=complex)
for dNum in range(nDays):
    antPosP=perturbPositions(antPos,np.ones(antPos.shape)*dAntPos*np.sqrt(2.))
    np.save('antPos/antPos_nAnt_%d_dAnt_%d_dAntPos_%d_day_%d.npy'%(nDish,dAnt,dAntPos,dNum),antPosP)
    antPosP=np.load('antPos/antPos_nAnt_%d_dAnt_%d_dAntPos_%d_day_%d.npy'%(nDish,dAnt,dAntPos,dNum))

    dIndex=0
    print 'day=%d'%(dNum)
    for iDish in range(nDish):
        for jDish in range(iDish):
            antDiffP[dIndex,0]=antPosP[iDish,0]-antPosP[jDish,0]
            antDiffP[dIndex,1]=antPosP[iDish,1]-antPosP[jDish,1]
            dIndex+=1
    for fNum in range(nChan):
        wL=c/fAxis[fNum]
        kernelFunc=lambda dx,dy: dishKernel(np.sqrt(dx**2.+dy**2.),dAnt/2./wL)
        #t=time.Timer('visList=computeVisList(lAngleList,mAngleList,antDiffP[:,0]/wL,antDiffP[:,1]/wL,fluxList,lambda l,m:airyDisk(l,m,wL,dAnt))')
        #print 'computing vis: %e'%(t.time(1))
        #start=time.time()
        gainList=airyDisk(lAngleList,mAngleList,wL,dAnt)
        computeVisList(lAngleList,mAngleList,antDiffP[:,0]/wL,antDiffP[:,1]/wL,fluxList*gainList,visList)
        #end=time.time()
        #print 'compute vis %e'%(1e6*(end-start))
        #t=time.Timer('uvCube[:,:,fNum]+=grid(gridSize,gridSize,0.5,0.5,antDiffP[:,0]/wL,antDiffP[:,1]/wL,visList,kernelFunc,2.*dAnt/wL)')
        #print 'gridding data %e'%(t.time(1))
        #start=time.time()
        #print int(2.*dAnt/wL)
        dGrid=0.5
        nKernel=int(np.ceil(4.*dAnt/wL/dGrid))
        uvCube[:,:,fNum]+=gridDishKernel(gridSize,gridSize,dGrid,dGrid,antDiffP[:,0]/wL,antDiffP[:,1]/wL,visList,dAnt/2./wL,uvCube[:,:,fNum].squeeze(),len(visList),nKernel)
        #end=time.time()
        #print 'grid %e'%(1e6*(end-start))
        sampleCube[:,:,fNum]+=gridDishKernel(gridSize,gridSize,dGrid,dGrid,antDiffP[:,0]/wL,antDiffP[:,1]/wL,np.ones(len(visList)),dAnt/2./wL,sampleCube[:,:,fNum].squeeze(),len(visList),nKernel)
#np.savez('dataCubes_17ant_highres.npz',sampleCube=sampleCube,uvCube=uvCube,dGrid=dGrid,fAxis=fAxis)

nKperpBins=30
windowCube=np.zeros_like(uvCube)
for mm in range(uvCube.shape[0]):
    for nn in range(uvCube.shape[1]):
        windowCube[mm,nn,:]=signal.blackmanharris(nChan)
        windowCube[mm,nn,:]/=np.sqrt(np.mean(windowCube[mm,nn,:]**2.))
uvCubeDelayTransform=fft.fftshift(fft.fft(fft.fftshift(uvCube/sampleCube*windowCube,axes=[2]),axis=2),axes=[2])
wedgeGrid=delayGrid(np.abs(uvCubeDelayTransform)**2.,0.5,maxUV,nKperpBins)
nFactor=np.sqrt(np.mean(np.abs(wedgeGrid[0,np.invert(np.isnan(wedgeGrid[0,:]))])**2.))
wedgeGrid/=nFactor
#plt.plot(wedgeGrid[:,10])
#plt.yscale('log')
#plt.show()
uAxis=np.arange(0,nKperpBins)*maxUV/nKperpBins
kPerpAxis=cosmology.u2kperp(uAxis,cosmology.f2z(f0))/.68
tauAxis=np.arange(nChan/2)/(nChan*df)
kParAxis=cosmology.eta2kpara(tauAxis,cosmology.f2z(f0))/.68
wedgeLine=kPerpAxis*cosmology.wedge(cosmology.f2z(f0))


kPerpGrid,kParGrid=np.meshgrid(kPerpAxis,kParAxis)

#print wedge
imWedgeGrid=np.log10(np.abs(wedgeGrid)).astype(np.float64)
imWedgeGrid[np.logical_or(np.isnan(imWedgeGrid),np.isinf(imWedgeGrid))]=0.
#imWedgeGridCopy=np.zeros(imWedgeGrid.shape)
#for mm in range(imWedgeGrid.shape[0]):
#    for nn in range(imWedgeGrid.shape[1]):
#        imWedgeGridCopy[mm,nn]=float(imWedgeGrid[mm,nn])
        #print imWedgeGridCopy[mm,nn]
#print imWedgeGrid.shape

#np.savez('wedgeGrid_37ant_higherRes.npz',gridVis=imWedgeGrid,kPerp=kPerpGrid,kPara=kParGrid)
#plt.plot(imWedgeGridCopy[:,0])

plt.pcolor(kPerpGrid,kParGrid,imWedgeGrid,vmin=-12,vmax=0,cmap='inferno')
#plt.imshow(np.log10(np.abs(np.flipud(wedgeGrid))),cmap='inferno',extent=[kPerpAxis[1],kPerpAxis.max(),kParAxis[1],kParAxis.max()],interpolation='nearest')
plt.plot(kPerpAxis,wedgeLine,lw=5,color='w')
plt.plot(kPerpAxis,wedgeLine,lw=2,color='k',ls='--')
plt.ylim(5e-2,2.5e0)
plt.xlim(1e-3,1e-1)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('$k_{||}$ ($h$Mpc$^{-1}$)')
plt.xlabel('$k_\perp$ ($h$Mpc$^{-1}$)')
plt.colorbar()
plt.gca().set_aspect('auto')
#plt.show()

#print cosmology.eta2kpara(1/(nChan*df),cosmology.f2z(f0))/.68
plt.savefig('wedgeSingleSource_1day_217Ant.png')
#plt.close()
#plt.close()

