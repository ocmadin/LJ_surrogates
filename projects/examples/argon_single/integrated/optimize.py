from LJ_surrogates.sampling.integrated_optimization import TestOptimizer

optimizer = TestOptimizer('force-field.offxml','test-set-collection.json', port=8005)

param_range = [[0.5, 1.5], [0.5, 1.5]]
smirks = ['[#18:1]']

optimizer.optimize(param_range=param_range,smirks=smirks)
