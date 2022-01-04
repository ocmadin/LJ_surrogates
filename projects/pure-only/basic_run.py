from LJ_surrogates.run_simulations import estimate_forcefield_properties
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField

forcefield_file = 'force-field.offxml'

forcefield = ForceField(forcefield_file)

property_dataset = PhysicalPropertyDataSet.from_json('test-set-collection.json')

estimate_forcefield_properties(property_dataset,forcefield)