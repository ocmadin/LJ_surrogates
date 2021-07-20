from LJ_surrogates.LJ_surrogates.build_surrogates import collate_physical_property_data
import json
import matplotlib.pyplot as plt
import numpy as np
path = '/media/owenmadin/storage/alcohol_alkane/modified_8_runs_7_15'

smirks_types_to_change = ['[#6X4:1]', '[#1:1]-[#6X4]', '[#8X2H1+0:1]', '[#1:1]-[#8]']

properties_all = collate_physical_property_data(path, smirks_types_to_change)


y = properties_all[15][list(properties_all[15].keys())[0]][2]
x = np.linspace(0,np.shape(y)[0]-1,num=np.shape(y)[0])
plt.scatter(x,y[:,0],ls='None',marker='x',label='Simulated Values')
plt.axhline(0.717,ls='--',color='k',label='Experimental Value')
plt.title(f'Density \n {list(properties_all[15].keys())[0]}')
plt.legend()
plt.show()

y = properties_all[30][list(properties_all[30].keys())[0]][2]
x = np.linspace(0,np.shape(y)[0]-1,num=np.shape(y)[0])
plt.scatter(x,y[:,0],ls='None',marker='x',label='Simulated Values')
plt.axhline(0.5012,ls='--',color='k',label='Experimental Value')
plt.title(f'Enthalpy of Mixing \n {list(properties_all[30].keys())[0]}')
plt.legend()
plt.show()