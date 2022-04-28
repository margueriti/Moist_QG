import numpy as np
import Moist_QG
import xarray as xr
import sys

paramfile = sys.argv[1]
ds = xr.open_dataset(paramfile,engine="h5netcdf")

Nx = int(ds['Nx'].values)
MQG = Moist_QG.Moist_QG(Nx =int(ds['Nx'].values), Ny = int(ds['Ny'].values), length=np.array([ds['Lx'].values,ds['Ly'].values]))
MQG.param.set_dry_parameters(damping=ds['damping'].values,beta = ds['beta'].values,RWN=ds['RWN'].values,dissipation_coeff=ds['dissipation_coeff'].values)
MQG.param.set_moist_parameters(moisture=ds['moisture'].values,rain=ds['rain'].values,latent_heating=ds['latent_heating'].values,CC=ds['CC'].values,evaporation=ds['evaporation'].values)
MQG.param.set_time_parameters(dt = ds['dt'].values, Nstep = ds['Nstep'].values, Nout = ds['Nout'].values, tau = ds['precip_relaxation'].values)
MQG.initial_conditions(init_moisture = ds['init_moisture'],noise = 0.001*ds['Nx'].values *ds['Ny'].values,  init_krange=16, init_lrange=16)

print (ds['precip_relaxation'].values,MQG.param.tau)
print ((1+ds['CC'].values*ds['latent_heating'].values)/(1-ds['latent_heating'].values))
step = Moist_QG.Timestepping_AB3IF(MQG)
step.timestepping()

ds.coords['kx'] = MQG.param.freqsx[:MQG.param.Nk]
ds.coords['ky'] = MQG.param.freqsy[:MQG.param.Nl]
ds.coords['time'] = MQG.param.time
ds.coords['total_layers'] = [n for n in range(MQG.param.total_layers)]
ds.coords['moist_layers'] = [n for n in range(np.sum(MQG.param.moisture))]
ds.coords['x'] = np.linspace(MQG.param.xymin[0], MQG.param.xymax[0], MQG.param.Nx, endpoint=False)
ds.coords['y'] = np.linspace(MQG.param.xymin[1], MQG.param.xymax[1], MQG.param.Ny, endpoint=False)

ds['psi_hat'] = (['time', 'layers', 'ky', 'kx'], MQG.save.psi_hat)
ds['pv_hat'] = (['time', 'total_layers', 'ky', 'kx'], MQG.save.pv_hat)
ds['tendency']=(['total_layers','ky','kx'],MQG.var.current_tendency)
ds['old']=(['total_layers','ky','kx'],MQG.var.old_tendency)
ds['oldold']=(['total_layers','ky','kx'],MQG.var.oldold_tendency)

#parameters (done inefficiently)
ds['nu'] = MQG.param.nu
ds['num'] = MQG.param.num
ds['gamma'] = MQG.param.gamma

if MQG.param.rain:
    ds['precip']=(['time', 'y', 'x'],MQG.save.precip)
if any(MQG.param.moisture):
    ds['water_content_hat'] = (['time', 'moist_layers', 'ky', 'kx'], MQG.save.water_content_hat)

savefile = '/scratch/mlb542/MQG_output/MQG_AB3IF_data_L='+str(ds['Lx_int'].values)+'_Nx='+str(ds['Nx'].values)+'_beta='+str(ds['beta'].values)+'_rain='+str(ds['rain'].values)+'_LH='+str(ds['latent_heating'].values)+'_CC='+str(ds['CC'].values)+'_E='+str(ds['evaporation'].values)+'_tau='+str(ds['precip_relaxation'].values)+'_dt='+str(ds['dt'].values)+'.nc'

ds.to_netcdf(savefile,engine="h5netcdf", invalid_netcdf=True)
