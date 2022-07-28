import xarray as xr
import numpy as np

ROOT = '/scratch/mlb542/MQG_output/inputs/'
test = '0'
ds = xr.Dataset()

#filetag
name='MQG_sat_sweep'

#domain size parameters
Lx_int=18
Ly_int=Lx_int
ds['Lx']=np.pi*Lx_int
ds['Ly']=np.pi*Ly_int

#resolution
ds['Nx']=256
ds['Ny']=256

ds['nlayers'] = 2#the code is not set up to handle more than 2, but I can dream

#coordinates
ds.coords['layers']= [n for n in range(ds['nlayers'].values)]

#dry parameters
ds['damping'] = (['layers'],np.array([0.0,0.16]))#Ekman damping
ds['beta'] = 1.25#inverse dry criticality
ds['RWN'] = (['layers'],np.array([1.0,1.0]))#inverse Rossby radius
ds['mean_velocity'] = (['layers'],np.array([0.5,-0.5]))#vertical shear
ds['dissipation_coeff'] = 10**(-7)#higher order numerical dissipation, currently set to del8

#moist parameters
ds['LH'] = 0.7#moisture stratification
ds['CC'] = 2.0#Clausius-Clayperon coefficient
ds['moisture'] = (['layers'],np.array([False, True],dtype=bool))#which layers have moisture? from top to bottom
ds['rain'] = True#is it raining? If no, moisture acts as a passive tracer
ds['evaporation'] = 1000.0#evaporation rate
ds['init_moisture'] = -50.0#initial domain averaged value of moisture

#checking consistency of moist parameters; if there is no moisture, then there can be no rain and latent heating, etc are removed
if sum(ds['moisture'].values) ==0:
        ds['rain'] = False
        ds['latent_heating']=0.0
        ds['CC']=0.0

#time parameters
nstep = 2000
ds['dt']=1.0/nstep#timestep size
ds['Nstep'] = 400*nstep#total run length
ds['Nout'] = nstep//10#output frequency
ds['Nsave'] = nstep*200#write-to-file frequency
ds['precip_relaxation'] = 5*ds['dt'].values#precipitation relaxation timescale

#naming the file
ds['name']=name+'_L='+str(Lx_int)+'_dx='+str(ds['Nx'].values)+'_dt='+str(ds['dt'].values)+'_critinv='+str(ds['beta'].values)+'_LH='+str(ds['LH'].values)+'_CC='+str(ds['CC'].values)+'_E='+str(ds['evaporation'].values)+'_tau='+str(ds['precip_relaxation'].values)

#ds['dissipation_coeff'] = ds['dissipation_coeff'].values*((1-ds['latent_heating'].values)/(1+ds['latent_heating'].values*ds['CC'].values))**3

d = ds.to_netcdf(ROOT+'MQG_params_'+test+'.nc',engine='h5netcdf',invalid_netcdf=True)
