import xarray as xr
import numpy as np

ROOT = '/scratch/mlb542/MQG_output/inputs/'
test = '0'
ds = xr.Dataset()

#filetag
name='MQG_sat_sweep'

for LH in [0.7]:
    for CC in [2.0]:
        for beta in [1.25]:
            for E in [1000.0]:
                
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
                ds['beta'] = beta
                ds['RWN'] = (['layers'],np.array([1.0,1.0]))
                ds['mean_velocity'] = (['layers'],np.array([0.5,-0.5]))
                ds['dissipation_coeff'] = 10**(-7)

                #moist parameters
                ds['latent_heating'] = LH
                ds['CC'] = CC
                ds['moisture'] = (['layers'],np.array([False, True],dtype=bool))
                ds['rain'] = True
                ds['evaporation'] = E
                ds['init_moisture'] = 0.0

                #checking consistency of moist parameters
                if sum(ds['moisture'].values) ==0:
                    ds['rain'] = False
                    ds['latent_heating']=0.0
                    ds['CC']=0.0

                #time parameters
                nstep = 2000# steps per lambda/U
                ds['dt']=1.0/nstep
                ds['Nstep'] = nstep*400#=400 lambda/U total runtimes
                ds['Nout'] = nstep//10#=outputs every .1 lambda/U
                ds['Nsave'] = nstep*200#=200 lambda/U save file length
                ds['precip_relaxation'] = 5*ds['dt'].values
                
                #update filename
                ds['name']=name+'_L='+str(ds['Lx_int'].values)+'_dx='+str(ds['Nx'].values)+'_dt='+str(ds['dt'].values)+'_critinv='+str(beta)+'_LH='+str(LH)+'_CC='+str(CC)+'_E='+str(ds['evaporation'].values)+'_tau='+str(ds['precip_relaxation'].values)
                d = ds.to_netcdf(ROOT+'MQG_params_'+str(test)+'.nc',engine='h5netcdf',invalid_netcdf=True)
                test = test+1
