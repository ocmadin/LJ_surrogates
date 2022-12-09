import pandas as pd
import numpy as np
import sklearn

df1 = pd.read_csv('df_1.csv')
df2 = pd.read_csv('df_2.csv')
df2 = df2.drop([14, 18])
df3 = pd.read_csv('df_3.csv')
df3 = df3.drop([11, 13, 14, 16])
df4 = pd.read_csv('df_4.csv')
df4 = df4.drop([11, 13])
df5 = pd.read_csv('df_5.csv')
df5 = df5.drop([14, 15])

df = df1.merge(df2, how='outer')
df = df.merge(df3, how='outer')
df = df.merge(df4, how='outer')
df = df.merge(df5, how='outer')


def flatten_parameters(forcefield):
    flat_parameters = []
    flat_parameter_names = []
    for key in forcefield.keys():
        flat_parameters.append(forcefield[key][0]._value)
        flat_parameters.append(forcefield[key][1]._value)
        flat_parameter_names.append(key + '_epsilon')
        flat_parameter_names.append(key + '_rmin_half')
    flat_parameters = np.asarray(flat_parameters)
    return flat_parameters


import os
from LJ_surrogates.surrogates.collate_data import get_force_field_parameters
from openff.toolkit.typing.engines.smirnoff import ForceField

ff_params = []
smirks = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
topdir = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/force-balance-pure-only'
for dir in os.listdir(topdir):
    ff = ForceField(os.path.join(topdir, dir, 'openff-1.0.0.offxml'), allow_cosmetic_attributes=True)
    ff_params.append(flatten_parameters(get_force_field_parameters(ff, smirks)))

objectives_fb = np.asarray([0.180, 0.088, 0.080, 0.0955, 0.091, 0.09, 0.095, 0.086, 0.091, 0.089, 0.089, 0.092])

fb_df = pd.DataFrame(np.asarray(ff_params), columns=df.columns[1:-1])

fb_df['objective'] = objectives_fb

df = df.merge(fb_df, how='outer')

from sklearn.preprocessing import StandardScaler

features = df.columns[1:-1]
target = df.columns[-1]

X = df.loc[:, features].values
y = df.loc[:, target].values

x = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA

pca = PCA(2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

import matplotlib.pyplot as plt

# plt.scatter(principalDf.values[9:24, 0], principalDf.values[9:24, 1], c=df['objective'].values[9:24], cmap='plasma', marker='x', label='Run 1')
plt.plot(principalDf.values[9:25, 0], principalDf.values[9:25, 1], alpha=0.5, lw=2,
         label=f'1 (objective={round(df["objective"].values[24], 3)})')
plt.scatter(principalDf.values[24, 0], principalDf.values[24, 1], marker='x', s=100, lw=3, color='tab:blue')
# plt.scatter(principalDf.values[34:41, 0], principalDf.values[34:41, 1], c=df['objective'].values[34:41], cmap='plasma', marker='x', label='Run 2')
plt.plot(principalDf.values[34:43, 0], principalDf.values[34:43, 1], alpha=0.5, lw=2,
         label=f'2 (objective={round(df["objective"].values[42], 3)})')
plt.scatter(principalDf.values[42, 0], principalDf.values[42, 1], marker='x', s=100, lw=3, color='tab:orange')
# plt.scatter(principalDf.values[51:65, 0], principalDf.values[51:65, 1], c=df['objective'].values[51:65], cmap='plasma', marker='x', label='Run 3')
plt.plot(principalDf.values[52:63, 0], principalDf.values[52:63, 1], alpha=0.5, lw=2,
         label=f'3 (objective={round(df["objective"].values[62], 3)})')
plt.scatter(principalDf.values[62, 0], principalDf.values[62, 1], marker='x', s=100, lw=3, color='tab:green')
# plt.scatter(principalDf.values[65:71, 0], principalDf.values[65:71, 1], c=df['objective'].values[65:71], cmap='plasma', marker='x', label='Run 4')
plt.plot(principalDf.values[72:77, 0], principalDf.values[72:77, 1], alpha=0.5, lw=2,
         label=f'4 (objective={round(df["objective"].values[76], 3)})')
plt.scatter(principalDf.values[9, 0], principalDf.values[9, 1], color='k', marker='x', s=100, lw=3,
            label='Initial Value')
plt.scatter(principalDf.values[76, 0], principalDf.values[76, 1], marker='x', s=100, lw=3, color='tab:red')
# plt.scatter(principalDf.values[81:89, 0], principalDf.values[81:89, 1], c=df['objective'].values[81:89], cmap='plasma', marker='x', label='Run 5')
plt.plot(principalDf.values[86:92, 0], principalDf.values[86:92, 1], alpha=0.5, lw=2,
         label=f'5 (objective={round(df["objective"].values[91], 3)})')
plt.scatter(principalDf.values[91, 0], principalDf.values[91, 1], marker='x', s=100, lw=3, color='tab:purple')

plt.plot(principalDf.values[92:, 0], principalDf.values[92:, 1], alpha=0.5, lw=2,
         label=f'FB (objective={round(df["objective"].values[103], 3)})')
plt.scatter(principalDf.values[103, 0], principalDf.values[103, 1], marker='x', s=100, lw=3, color='tab:brown')
# plt.plot(principalDf.values[52:65, 0], principalDf.values[52:65, 1], marker='x', label='Run 3')
# plt.plot(principalDf.values[66:71, 0], principalDf.values[66:71, 1], marker='x', label='Run 4')
# plt.plot(principalDf.values[82:89, 0], principalDf.values[82:89, 1], marker='x', label='Run 5')
plt.title('2-D PCA of optimization trajectories (50% of variance)')
plt.xlabel('PC1')
plt.xlabel('PC2')
plt.legend()
plt.savefig('pca_10.png')
plt.cla()

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
labels = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
# plot [#1:1-6X4]
ax[0, 0].plot(df.values[9:24, 1], df.values[9:24, 2])
ax[0, 0].plot(df.values[34:43, 1], df.values[34:43, 2])
ax[0, 0].plot(df.values[52:63, 1], df.values[52:63, 2])
ax[0, 0].plot(df.values[72:77, 1], df.values[72:77, 2])
ax[0, 0].plot(df.values[86:92, 1], df.values[86:92, 2])
ax[0, 0].plot(df.values[92:, 1], df.values[92:, 2])
ax[0, 0].scatter(df.values[23, 1], df.values[23, 2], marker='x', s=100, lw=3, color='tab:blue')
ax[0, 0].scatter(df.values[42, 1], df.values[42, 2], marker='x', s=100, lw=3, color='tab:orange')
ax[0, 0].scatter(df.values[62, 1], df.values[62, 2], marker='x', s=100, lw=3, color='tab:green')
ax[0, 0].scatter(df.values[76, 1], df.values[76, 2], marker='x', s=100, lw=3, color='tab:red')
ax[0, 0].scatter(df.values[91, 1], df.values[91, 2], marker='x', s=100, lw=3, color='tab:purple')
ax[0, 0].scatter(df.values[-1, 1], df.values[-1, 2], marker='x', s=100, lw=3, color='tab:brown')
ax[0, 0].set_title(labels[0], fontsize=16)
ax[0, 0].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[0, 0].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
ax[0, 0].locator_params(axis='both', nbins=6)
# plot [#6:1]
ax[0, 1].plot(df.values[9:24, 3], df.values[9:24, 4])
ax[0, 1].plot(df.values[34:43, 3], df.values[34:43, 4])
ax[0, 1].plot(df.values[52:63, 3], df.values[52:63, 4])
ax[0, 1].plot(df.values[72:77, 3], df.values[72:77, 4])
ax[0, 1].plot(df.values[86:92, 3], df.values[86:92, 4])
ax[0, 1].plot(df.values[92:, 3], df.values[92:, 4])
ax[0, 1].scatter(df.values[23, 3], df.values[23, 4], marker='x', s=100, lw=3, color='tab:blue')
ax[0, 1].scatter(df.values[42, 3], df.values[42, 4], marker='x', s=100, lw=3, color='tab:orange')
ax[0, 1].scatter(df.values[62, 3], df.values[62, 4], marker='x', s=100, lw=3, color='tab:green')
ax[0, 1].scatter(df.values[76, 3], df.values[76, 4], marker='x', s=100, lw=3, color='tab:red')
ax[0, 1].scatter(df.values[91, 3], df.values[91, 4], marker='x', s=100, lw=3, color='tab:purple')
ax[0, 1].scatter(df.values[-1, 3], df.values[-1, 4], marker='x', s=100, lw=3, color='tab:brown')
ax[0, 1].set_title(labels[1], fontsize=16)
ax[0, 1].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[0, 1].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
ax[0, 1].locator_params(axis='both', nbins=6)

# plot [#6X4:1]
ax[0, 2].plot(df.values[9:24, 5], df.values[9:24, 6])
ax[0, 2].plot(df.values[34:43, 5], df.values[34:43, 6])
ax[0, 2].plot(df.values[52:63, 5], df.values[52:63, 6])
ax[0, 2].plot(df.values[72:77, 5], df.values[72:77, 6])
ax[0, 2].plot(df.values[86:92, 5], df.values[86:92, 6])
ax[0, 2].plot(df.values[92:, 5], df.values[92:, 6])
ax[0, 2].scatter(df.values[23, 5], df.values[23, 6], marker='x', s=100, lw=3, color='tab:blue')
ax[0, 2].scatter(df.values[42, 5], df.values[42, 6], marker='x', s=100, lw=3, color='tab:orange')
ax[0, 2].scatter(df.values[62, 5], df.values[62, 6], marker='x', s=100, lw=3, color='tab:green')
ax[0, 2].scatter(df.values[76, 5], df.values[76, 6], marker='x', s=100, lw=3, color='tab:red')
ax[0, 2].scatter(df.values[91, 5], df.values[91, 6], marker='x', s=100, lw=3, color='tab:purple')
ax[0, 2].scatter(df.values[-1, 5], df.values[-1, 6], marker='x', s=100, lw=3, color='tab:brown')
ax[0, 2].set_title(labels[2], fontsize=16)
ax[0, 2].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[0, 2].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
ax[0, 2].locator_params(axis='both', nbins=6)

# plot [#8:1]
ax[1, 0].plot(df.values[9:24, 7], df.values[9:24, 8])
ax[1, 0].plot(df.values[34:43, 7], df.values[34:43, 8])
ax[1, 0].plot(df.values[52:63, 7], df.values[52:63, 8])
ax[1, 0].plot(df.values[72:77, 7], df.values[72:77, 8])
ax[1, 0].plot(df.values[86:92, 7], df.values[86:92, 8])
ax[1, 0].plot(df.values[92:, 7], df.values[92:, 8])
ax[1, 0].scatter(df.values[23, 7], df.values[23, 8], marker='x', s=100, lw=3, color='tab:blue')
ax[1, 0].scatter(df.values[42, 7], df.values[42, 8], marker='x', s=100, lw=3, color='tab:orange')
ax[1, 0].scatter(df.values[62, 7], df.values[62, 8], marker='x', s=100, lw=3, color='tab:green')
ax[1, 0].scatter(df.values[76, 7], df.values[76, 8], marker='x', s=100, lw=3, color='tab:red')
ax[1, 0].scatter(df.values[91, 7], df.values[91, 8], marker='x', s=100, lw=3, color='tab:purple')
ax[1, 0].scatter(df.values[-1, 7], df.values[-1, 8], marker='x', s=100, lw=3, color='tab:brown')
ax[1, 0].set_title(labels[3], fontsize=16)
ax[1, 0].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[1, 0].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
ax[1, 0].locator_params(axis='both', nbins=6)

# plot [#8X2H0+0:1]
ax[1, 1].plot(df.values[9:24, 9], df.values[9:24, 10])
ax[1, 1].plot(df.values[34:43, 9], df.values[34:43, 10])
ax[1, 1].plot(df.values[52:63, 9], df.values[52:63, 10])
ax[1, 1].plot(df.values[72:77, 9], df.values[72:77, 10])
ax[1, 1].plot(df.values[86:92, 9], df.values[86:92, 10])
ax[1, 1].plot(df.values[92:, 9], df.values[92:, 10])
ax[1, 1].scatter(df.values[23, 9], df.values[23, 10], marker='x', s=100, lw=3, color='tab:blue')
ax[1, 1].scatter(df.values[42, 9], df.values[42, 10], marker='x', s=100, lw=3, color='tab:orange')
ax[1, 1].scatter(df.values[62, 9], df.values[62, 10], marker='x', s=100, lw=3, color='tab:green')
ax[1, 1].scatter(df.values[76, 9], df.values[76, 10], marker='x', s=100, lw=3, color='tab:red')
ax[1, 1].scatter(df.values[91, 9], df.values[91, 10], marker='x', s=100, lw=3, color='tab:purple')
ax[1, 1].scatter(df.values[-1, 9], df.values[-1, 10], marker='x', s=100, lw=3, color='tab:brown')
ax[1, 1].set_title(labels[4], fontsize=16)
ax[1, 1].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[1, 1].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
ax[1, 1].locator_params(axis='both', nbins=6)

# plot [#8X2H1+0:1]
ax[1, 2].plot(df.values[9:24, 11], df.values[9:24, 12], label='MF Run 1')
ax[1, 2].plot(df.values[34:43, 11], df.values[34:43, 12], label='MF Run 2')
ax[1, 2].plot(df.values[52:63, 11], df.values[52:63, 12], label='MF Run 3')
ax[1, 2].plot(df.values[72:77, 11], df.values[72:77, 12], label='MF Run 4')
ax[1, 2].plot(df.values[86:92, 11], df.values[86:92, 12], label='MF Run 5')
ax[1, 2].plot(df.values[92:, 11], df.values[92:, 12], label='Simulation L-BFGS-B')
ax[1, 2].scatter(df.values[23, 11], df.values[23, 12], marker='x', s=100, lw=3, color='tab:blue')
ax[1, 2].scatter(df.values[42, 11], df.values[42, 12], marker='x', s=100, lw=3, color='tab:orange')
ax[1, 2].scatter(df.values[62, 11], df.values[62, 12], marker='x', s=100, lw=3, color='tab:green')
ax[1, 2].scatter(df.values[76, 11], df.values[76, 12], marker='x', s=100, lw=3, color='tab:red')
ax[1, 2].scatter(df.values[91, 11], df.values[91, 12], marker='x', s=100, lw=3, color='tab:purple')
ax[1, 2].scatter(df.values[-1, 11], df.values[-1, 12], marker='x', s=100, lw=3, color='tab:brown')
ax[1, 2].set_title(labels[5], fontsize=16)
ax[1, 2].legend()
ax[1, 2].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[1, 2].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
ax[1, 2].locator_params(axis='both', nbins=6)

plt.tight_layout()
plt.savefig('parameter_trajectory_10.png', dpi=300)
fig.show()

df1 = pd.read_csv('df_1_5_initial.csv')
df1 = df1.drop([6, 7, 8, 16])
df2 = pd.read_csv('df_2_5_initial.csv')
df2 = df2.drop([5, 6, 7, 8, 10, 11])
df3 = pd.read_csv('df_3_5_initial.csv')
df3 = df3.drop([7])
df4 = pd.read_csv('df_4_5_initial.csv')
df4 = df4.drop([5, 6, 9, 11])
df4['objective'][4] = 0.15955
df5 = pd.read_csv('df_5_5_initial.csv')
df5 = df5.drop([5, 6, 7, 10, 11, 13, 14, 15, 18])

df = df1.merge(df2, how='outer')
df = df.merge(df3, how='outer')
df = df.merge(df4, how='outer')
df = df.merge(df5, how='outer')


def flatten_parameters(forcefield):
    flat_parameters = []
    flat_parameter_names = []
    for key in forcefield.keys():
        flat_parameters.append(forcefield[key][0]._value)
        flat_parameters.append(forcefield[key][1]._value)
        flat_parameter_names.append(key + '_epsilon')
        flat_parameter_names.append(key + '_rmin_half')
    flat_parameters = np.asarray(flat_parameters)
    return flat_parameters


ff_params = []
smirks = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
topdir = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/force-balance-pure-only'
for dir in os.listdir(topdir):
    ff = ForceField(os.path.join(topdir, dir, 'openff-1.0.0.offxml'), allow_cosmetic_attributes=True)
    ff_params.append(flatten_parameters(get_force_field_parameters(ff, smirks)))

objectives_fb = np.asarray([0.180, 0.088, 0.080, 0.0955, 0.091, 0.09, 0.095, 0.086, 0.091, 0.089, 0.089, 0.092])

fb_df = pd.DataFrame(np.asarray(ff_params), columns=df.columns[1:-1])

fb_df['objective'] = objectives_fb

df = df.merge(fb_df, how='outer')

from sklearn.preprocessing import StandardScaler

features = df.columns[1:-1]
target = df.columns[-1]

X = df.loc[:, features].values
y = df.loc[:, target].values

x = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA

pca = PCA(2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
df.to_csv('5-objs.csv')
import matplotlib.pyplot as plt

# plt.scatter(principalDf.values[9:24, 0], principalDf.values[9:24, 1], c=df['objective'].values[9:24], cmap='plasma', marker='x', label='Run 1')
plt.plot(principalDf.values[4:16, 0], principalDf.values[4:16, 1], alpha=0.5, lw=2,
         label=f'1 (objective={round(df["objective"].values[15], 3)})')
plt.scatter(principalDf.values[15, 0], principalDf.values[15, 1], marker='x', s=100, lw=3, color='tab:blue')
# plt.scatter(principalDf.values[34:41, 0], principalDf.values[34:41, 1], c=df['objective'].values[34:41], cmap='plasma', marker='x', label='Run 2')
plt.plot(principalDf.values[20:27, 0], principalDf.values[20:27, 1], alpha=0.5, lw=2,
         label=f'2 (objective={round(df["objective"].values[26], 3)})')
plt.scatter(principalDf.values[26, 0], principalDf.values[26, 1], marker='x', s=100, lw=3, color='tab:orange')
# plt.scatter(principalDf.values[51:65, 0], principalDf.values[51:65, 1], c=df['objective'].values[51:65], cmap='plasma', marker='x', label='Run 3')
plt.plot(principalDf.values[31:46, 0], principalDf.values[31:46, 1], alpha=0.5, lw=2,
         label=f'3 (objective={round(df["objective"].values[45], 3)})')
plt.scatter(principalDf.values[45, 0], principalDf.values[45, 1], marker='x', s=100, lw=3, color='tab:green')
# plt.scatter(principalDf.values[65:71, 0], principalDf.values[65:71, 1], c=df['objective'].values[65:71], cmap='plasma', marker='x', label='Run 4')
plt.plot(principalDf.values[50:57, 0], principalDf.values[50:57, 1], alpha=0.5, lw=2,
         label=f'4 (objective={round(df["objective"].values[56], 3)})')
plt.scatter(principalDf.values[4, 0], principalDf.values[4, 1], color='k', marker='x', s=100, lw=3,
            label='Initial Value')
plt.scatter(principalDf.values[56, 0], principalDf.values[56, 1], marker='x', s=100, lw=3, color='tab:red')
plt.scatter(principalDf.values[61:68, 0], principalDf.values[61:68, 1], cmap='plasma', marker='x', label='Run 5')
plt.plot(principalDf.values[61:68, 0], principalDf.values[61:68, 1], alpha=0.5, lw=2,
         label=f'5 (objective={round(df["objective"].values[67], 3)})', color='tab:purple')
plt.scatter(principalDf.values[67, 0], principalDf.values[67, 1], marker='x', s=100, lw=3, color='tab:purple')

plt.plot(principalDf.values[68:, 0], principalDf.values[68:, 1], alpha=0.5, lw=2,
         label=f'FB (objective={round(df["objective"].values[-1], 3)})')
plt.scatter(principalDf.values[-1, 0], principalDf.values[-1, 1], marker='x', s=100, lw=3, color='tab:brown')
# plt.plot(principalDf.values[52:65, 0], principalDf.values[52:65, 1], marker='x', label='Run 3')
# plt.plot(principalDf.values[66:71, 0], principalDf.values[66:71, 1], marker='x', label='Run 4')
# plt.plot(principalDf.values[82:89, 0], principalDf.values[82:89, 1], marker='x', label='Run 5')
plt.title('2-D PCA of optimization trajectories (50% of variance)')
plt.xlabel('PC1')
plt.xlabel('PC2')
plt.legend()
plt.savefig('pca_5.png')

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
labels = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
# plot [#1:1-6X4]
ax[0, 0].plot(df.values[4:16, 1], df.values[4:16, 2])
ax[0, 0].plot(df.values[20:27, 1], df.values[20:27, 2])
ax[0, 0].plot(df.values[31:46, 1], df.values[31:46, 2])
ax[0, 0].plot(df.values[50:57, 1], df.values[50:57, 2])
ax[0, 0].plot(df.values[61:68, 1], df.values[61:68, 2])
ax[0, 0].plot(df.values[68:, 1], df.values[68:, 2])
ax[0, 0].scatter(df.values[15, 1], df.values[15, 2], marker='x', s=100, lw=3, color='tab:blue')
ax[0, 0].scatter(df.values[26, 1], df.values[26, 2], marker='x', s=100, lw=3, color='tab:orange')
ax[0, 0].scatter(df.values[45, 1], df.values[45, 2], marker='x', s=100, lw=3, color='tab:green')
ax[0, 0].scatter(df.values[56, 1], df.values[56, 2], marker='x', s=100, lw=3, color='tab:red')
ax[0, 0].scatter(df.values[67, 1], df.values[67, 2], marker='x', s=100, lw=3, color='tab:purple')
ax[0, 0].scatter(df.values[-1, 1], df.values[-1, 2], marker='x', s=100, lw=3, color='tab:brown')
ax[0, 0].set_title(labels[0], fontsize=16)
ax[0, 0].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[0, 0].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
ax[0, 0].locator_params(axis='both', nbins=6)
# plot [#6:1]
ax[0, 1].plot(df.values[4:16, 3], df.values[4:16, 4])
ax[0, 1].plot(df.values[20:27, 3], df.values[20:27, 4])
ax[0, 1].plot(df.values[31:46, 3], df.values[31:46, 4])
ax[0, 1].plot(df.values[50:57, 3], df.values[50:57, 4])
ax[0, 1].plot(df.values[61:68, 3], df.values[61:68, 4])
ax[0, 1].plot(df.values[68:, 3], df.values[68:, 4])
ax[0, 1].scatter(df.values[15, 3], df.values[15, 4], marker='x', s=100, lw=3, color='tab:blue')
ax[0, 1].scatter(df.values[26, 3], df.values[26, 4], marker='x', s=100, lw=3, color='tab:orange')
ax[0, 1].scatter(df.values[45, 3], df.values[45, 4], marker='x', s=100, lw=3, color='tab:green')
ax[0, 1].scatter(df.values[56, 3], df.values[56, 4], marker='x', s=100, lw=3, color='tab:red')
ax[0, 1].scatter(df.values[67, 3], df.values[67, 4], marker='x', s=100, lw=3, color='tab:purple')
ax[0, 1].scatter(df.values[-1, 3], df.values[-1, 4], marker='x', s=100, lw=3, color='tab:brown')
ax[0, 1].set_title(labels[1], fontsize=16)
ax[0, 1].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[0, 1].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
ax[0, 1].locator_params(axis='both', nbins=6)

# plot [#6X4:1]
ax[0, 2].plot(df.values[4:16, 5], df.values[4:16, 6])
ax[0, 2].plot(df.values[20:27, 5], df.values[20:27, 6])
ax[0, 2].plot(df.values[31:46, 5], df.values[31:46, 6])
ax[0, 2].plot(df.values[50:57, 5], df.values[50:57, 6])
ax[0, 2].plot(df.values[61:68, 5], df.values[61:68, 6])
ax[0, 2].plot(df.values[68:, 5], df.values[68:, 6])
ax[0, 2].scatter(df.values[15, 5], df.values[15, 6], marker='x', s=100, lw=3, color='tab:blue')
ax[0, 2].scatter(df.values[26, 5], df.values[26, 6], marker='x', s=100, lw=3, color='tab:orange')
ax[0, 2].scatter(df.values[45, 5], df.values[45, 6], marker='x', s=100, lw=3, color='tab:green')
ax[0, 2].scatter(df.values[56, 5], df.values[56, 6], marker='x', s=100, lw=3, color='tab:red')
ax[0, 2].scatter(df.values[67, 5], df.values[67, 6], marker='x', s=100, lw=3, color='tab:purple')
ax[0, 2].scatter(df.values[-1, 5], df.values[-1, 6], marker='x', s=100, lw=3, color='tab:brown')
ax[0, 2].set_title(labels[2], fontsize=16)
ax[0, 2].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[0, 2].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
ax[0, 2].locator_params(axis='both', nbins=6)

# plot [#8:1]
ax[1, 0].plot(df.values[4:16, 7], df.values[4:16, 8])
ax[1, 0].plot(df.values[20:27, 7], df.values[20:27, 8])
ax[1, 0].plot(df.values[31:46, 7], df.values[31:46, 8])
ax[1, 0].plot(df.values[50:57, 7], df.values[50:57, 8])
ax[1, 0].plot(df.values[61:68, 7], df.values[61:68, 8])
ax[1, 0].plot(df.values[68:, 7], df.values[68:, 8])
ax[1, 0].scatter(df.values[15, 7], df.values[15, 8], marker='x', s=100, lw=3, color='tab:blue')
ax[1, 0].scatter(df.values[26, 7], df.values[26, 8], marker='x', s=100, lw=3, color='tab:orange')
ax[1, 0].scatter(df.values[45, 7], df.values[45, 8], marker='x', s=100, lw=3, color='tab:green')
ax[1, 0].scatter(df.values[56, 7], df.values[56, 8], marker='x', s=100, lw=3, color='tab:red')
ax[1, 0].scatter(df.values[67, 7], df.values[67, 8], marker='x', s=100, lw=3, color='tab:purple')
ax[1, 0].scatter(df.values[-1, 7], df.values[-1, 8], marker='x', s=100, lw=3, color='tab:brown')
ax[1, 0].set_title(labels[3], fontsize=16)
ax[1, 0].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[1, 0].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
ax[1, 0].locator_params(axis='both', nbins=6)
# plot [#8X2H0+0:1]
ax[1, 1].plot(df.values[4:16, 9], df.values[4:16, 10])
ax[1, 1].plot(df.values[20:27, 9], df.values[20:27, 10])
ax[1, 1].plot(df.values[31:46, 9], df.values[31:46, 10])
ax[1, 1].plot(df.values[50:57, 9], df.values[50:57, 10])
ax[1, 1].plot(df.values[61:68, 9], df.values[61:68, 10])
ax[1, 1].plot(df.values[68:, 9], df.values[68:, 10])
ax[1, 1].scatter(df.values[15, 9], df.values[15, 10], marker='x', s=100, lw=3, color='tab:blue')
ax[1, 1].scatter(df.values[26, 9], df.values[26, 10], marker='x', s=100, lw=3, color='tab:orange')
ax[1, 1].scatter(df.values[45, 9], df.values[45, 10], marker='x', s=100, lw=3, color='tab:green')
ax[1, 1].scatter(df.values[56, 9], df.values[56, 10], marker='x', s=100, lw=3, color='tab:red')
ax[1, 1].scatter(df.values[67, 9], df.values[67, 10], marker='x', s=100, lw=3, color='tab:purple')
ax[1, 1].scatter(df.values[-1, 9], df.values[-1, 10], marker='x', s=100, lw=3, color='tab:brown')
ax[1, 1].set_title(labels[4], fontsize=16)
ax[1, 1].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[1, 1].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
ax[1, 1].locator_params(axis='both', nbins=6)
# plot [#8X2H1+0:1]
ax[1, 2].plot(df.values[4:16, 11], df.values[4:16, 12], label='MF Run 1')
ax[1, 2].plot(df.values[20:27, 11], df.values[20:27, 12], label='MF Run 2')
ax[1, 2].plot(df.values[31:46, 11], df.values[31:46, 12], label='MF Run 3')
ax[1, 2].plot(df.values[50:57, 11], df.values[50:57, 12], label='MF Run 4')
ax[1, 2].plot(df.values[61:68, 11], df.values[61:68, 12], label='MF Run 5')
ax[1, 2].plot(df.values[68:, 11], df.values[68:, 12], label='Simulation L-BFGS-B')
ax[1, 2].scatter(df.values[15, 11], df.values[15, 12], marker='x', s=100, lw=3, color='tab:blue')
ax[1, 2].scatter(df.values[26, 11], df.values[26, 12], marker='x', s=100, lw=3, color='tab:orange')
ax[1, 2].scatter(df.values[45, 11], df.values[45, 12], marker='x', s=100, lw=3, color='tab:green')
ax[1, 2].scatter(df.values[56, 11], df.values[56, 12], marker='x', s=100, lw=3, color='tab:red')
ax[1, 2].scatter(df.values[67, 11], df.values[67, 12], marker='x', s=100, lw=3, color='tab:purple')
ax[1, 2].scatter(df.values[-1, 11], df.values[-1, 12], marker='x', s=100, lw=3, color='tab:brown')
ax[1, 2].set_title(labels[5], fontsize=16)
ax[1, 2].locator_params(axis='both', nbins=6)
ax[1, 2].legend()
ax[1, 2].set_xlabel(r'$\epsilon$, kcal/mol', fontsize=14)
ax[1, 2].set_ylabel(r'$R_{min/2}$, $\AA$', fontsize=14)
plt.tight_layout()
plt.savefig('parameter_trajectory_5.png', dpi=300)
fig.show()
