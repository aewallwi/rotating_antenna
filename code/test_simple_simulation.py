import simple_simulation
import numpy as np
import matplotlib.pyplot as plt
import cosmology
simpleCatalog=np.array([[0.,.9,1.,0.]])
simulationDefault=simple_simulation.HexDishArray(nSide=9,dAnt=2.,dAntPos=3.,nDays=1)
simulationDefault.setGridResolution(.25)
simulationDefault.setSourceList(simpleCatalog)
simulationDefault.gridDays()
kperp_c,kpara_c,cubeGrid=simulationDefault.getCosmologyDCubeGrid()
kperp_dt,kpara_dt,cubeDGrid=simulationDefault.getCosmologyDVisGrid()
