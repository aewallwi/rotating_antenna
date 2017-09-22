import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.signal as signal
import cosmology
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
    xGrid,yGrid=np.meshgrid(np.arange(nx)*dx,np.arange(ny)*dy)
    xGrid-=ny*dy/2.
    yGrid-=nx*dx/2.
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



    return griddedData

def airyDisk(l,m,waveLength,d):
    x=2.*pi/waveLength*d*np.sqrt(l**2.+m**2.)
    return (2.*sp.jn(1,x)/x)**2.


def computeVisList(lList,mList,uList,vList,sList,beamFunc):
    gains=beamFunc(lList,mList)
    visList=np.zeros(len(uList))
    for uvNum in range(len(uList)):
        for lmNum in range(len(lList)):
            visList[uvNum]+=gains[lmNum]*sList[lmNum]*np.exp(2j*pi*(uList[uvNum]*lList[lmNum]+vList[uvNum]*mList[lmNum]))
    return visList
#generate array
dAnt=2.
dAntPos=5.
antPos=hexPack(4,dAntPos)
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
                dGrid[:,binNum]+=np.flipud(dataCube[uNum,vNum,dataCube.shape[-1]/2:].squeeze())
                dGrid[:,binNum]+=dataCube[uNum,vNum,:dataCube.shape[-1]/2]
    for bNum in range(nCells):
        dGrid[:,bNum]/=counts[bNum]
    return dGrid



nVis=nDish*(nDish-1)/2
antDiff=np.zeros((nVis,2))
antDiffP=np.zeros_like(antDiff)
dIndex=0
nChan=100
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
print uvAmp
gridSize=int(np.ceil(4.*maxUV))
if(np.mod(gridSize,2)==1):
    gridSize+=1



uvCube=np.zeros((gridSize,gridSize,nChan),dtype=complex)
sampleCube=np.zeros((gridSize,gridSize,nChan),dtype=complex)

lAngleList=np.array([0])
mAngleList=np.array([0.8])
fluxList=np.array([1])

nDays=1


for dNum in range(nDays):
    antPosP=perturbPositions(antPos,np.ones(antPos.shape)*2.)
    dIndex=0
    for iDish in range(nDish):
        for jDish in range(iDish):
            antDiffP[dIndex,0]=antPosP[iDish,0]-antPosP[jDish,0]
            antDiffP[dIndex,1]=antPosP[iDish,1]-antPosP[jDish,1]
            dIndex+=1
    for fNum in range(nChan):
        wL=c/fAxis[fNum]
        kernelFunc=lambda dx,dy: dishKernel(np.sqrt(dx**2.+dy**2.),dAnt/2./wL)
        visList=computeVisList(lAngleList,mAngleList,antDiffP[:,0]/wL,antDiffP[:,1]/wL,fluxList,lambda l,m:airyDisk(l,m,wL,dAnt))
        uvCube[:,:,fNum]+=grid(gridSize,gridSize,0.5,0.5,antDiffP[:,0]/wL,antDiffP[:,1]/wL,visList,kernelFunc,2.*dAnt/wL)
        sampleCube[:,:,fNum]+=grid(gridSize,gridSize,0.5,0.5,antDiffP[:,0]/wL,antDiffP[:,1]/wL,np.ones(len(visList)),kernelFunc,2.*dAnt/wL)

nKperpBins=50
windowCube=np.zeros_like(uvCube)
for mm in range(uvCube.shape[0]):
    for nn in range(uvCube.shape[1]):
        windowCube[mm,nn,:]=signal.blackmanharris(nChan)
        windowCube[mm,nn,:]/=np.sqrt(np.mean(windowCube[mm,nn,:]**2.))
uvCubeDelayTransform=fft.fftshift(fft.fft(fft.fftshift(uvCube/sampleCube*windowCube,axes=[2]),axis=2),axes=[2])
wedgeGrid=delayGrid(np.abs(uvCubeDelayTransform)**2.,0.5,maxUV,nKperpBins)


uAxis=np.arange(0,nKperpBins)*maxUV/nKperpBins
kPerpAxis=cosmology.u2kperp(uAxis,cosmology.f2z(f0))/.68
tauAxis=np.arange(1,nChan/2+1)/(nChan*df)
kParAxis=cosmology.eta2kpara(tauAxis,cosmology.f2z(f0))/.68
wedgeLine=kPerpAxis*cosmology.wedge(cosmology.f2z(f0))
print kParAxis.max()
print kParAxis.min()
print kPerpGrid.shape
print wedgeGrid.shape
print maxUV

kPerpGrid,kParGrid=np.meshgrid(kPerpAxis,kParAxis)


plt.pcolor(kPerpGrid,kParGrid,np.flipud(np.log10(np.abs(wedgeGrid))),vmin=-13,vmax=0,cmap='inferno')
plt.plot(kPerpAxis,wedgeLine,lw=5,color='w')
plt.plot(kPerpAxis,wedgeLine,lw=2,color='k',ls='--')
plt.ylim(5e-2,2.5e0)
plt.xlim(1e-3,1e-1)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$k_{||}$ ($h$Mpc$^{-1}$)')
plt.ylabel('$k_\perp$ ($h$Mpc$^{-1}$)')
plt.colorbar()
plt.show()
print cosmology.eta2kpara(1/(nChan*df),cosmology.f2z(f0))/.68
#plt.savefig('wedgeSingleSource_1day.png')
#plt.close()
