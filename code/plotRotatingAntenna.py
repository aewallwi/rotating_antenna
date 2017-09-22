import numpy as np
import matplotlib.pyplot as plt
import sys
args=sys.argv
inputFile,output=sys.argv[-2],sys.argv[-1]
data=np.load(inputFile)
kPerpGrid=data['kPerp']
kParGrid=data['kPara']
imWedgeGrid=data['gridVis']



plt.pcolor(kPerpGrid,kParGrid,imWedgeGrid,vmin=-12,vmax=0,cmap='inferno')
#plt.imshow(np.log10(np.abs(np.flipud(wedgeGrid))),cmap='inferno',extent=[kPerpAxis[1],kPerpAxis.max(),kParAxis[1],kParAxis.max()],interpolation='nearest')
#plt.plot(kPerpAxis,wedgeLine,lw=5,color='w')
#plt.plot(kPerpAxis,wedgeLine,lw=2,color='k',ls='--')
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
plt.savefig(output+'.png')
