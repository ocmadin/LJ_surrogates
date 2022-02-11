from LJ_surrogates.sampling.integrated_optimization import IntegratedOptimizer

optimizer = IntegratedOptimizer('force-field.offxml','test-set-collection.json',port=8003)

optimizer.optimize()