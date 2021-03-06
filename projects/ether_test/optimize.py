from LJ_surrogates.sampling.integrated_optimization import SurrogateDESearchOptimizer

optimizer = SurrogateDESearchOptimizer('openff-1.0.0.offxml','test-set-collection.json', port=8005)

# param_range = [[0.5, 1.5], [0.95, 1.05], [0.9, 1.1], [0.95, 1.05], [0.9, 1.1], [0.95, 1.05], [0.95, 1.05],
#                [0.95, 1.05],
#                [0.95, 1.05], [0.95, 1.05], [0.95, 1.05], [0.95, 1.05]]
# smirks = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
smirks = ['[#8X2H0+0:1]']
param_range = [[0.8, 1.2], [0.8, 1.2]]
max_simulations = 20

initial_samples = 20

optimizer.optimize(param_range=param_range,smirks=smirks, max_simulations=max_simulations, initial_samples=initial_samples,n_workers=20, use_cached_data=False)
