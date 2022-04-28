import xarray as xr
import numpy as np

ROOT = '/home/mlb542/Moist_QG/inputs/'
test = '0'
ds = xr.Dataset()

#basic parameters
ds['Lx_int']=18
ds['Ly_int']=ds['Lx_int']
ds['Lx']=np.pi*ds['Lx_int']
ds['Ly']=np.pi*ds['Ly_int']

ds['Nx']=256
ds['Ny']=256

ds['nlayers'] = 2

#coordinates
ds.coords['layers']= [n for n in range(ds['nlayers'].values)]

#dry parameters
ds['damping'] = (['layers'],np.array([0.0,0.16]))
ds['beta'] = 0.8
ds['RWN'] = (['layers'],np.array([1.0,1.0]))
ds['mean_velocity'] = (['layers'],np.array([0.5,-0.5]))
ds['dissipation_coeff'] = 10**(-7)

#moist parameters
ds['latent_heating'] = 0.9
ds['CC'] = 0.0
ds['moisture'] = (['layers'],np.array([False, True],dtype=bool))
ds['rain'] = True
ds['evaporation'] = 500.0
ds['init_moisture'] = -50.0

#checking consistency of moist parameters
if sum(ds['moisture'].values) ==0:
	ds['rain'] = False
	ds['latent_heating']=0.0
	ds['CC']=0.0

#time parameters
nstep = 1000
ds['dt']=1.0/nstep
ds['Nstep'] = 400*nstep
ds['Nout'] = nstep
ds['precip_relaxation'] = .15#200.0*ds['dt'].values

#ds['dissipation_coeff'] = ds['dissipation_coeff'].values*((1-ds['latent_heating'].values)/(1+ds['latent_heating'].values*ds['CC'].values))**3

d = ds.to_netcdf(ROOT+'MQG_params_'+test+'.nc',engine='h5netcdf',invalid_netcdf=True)
