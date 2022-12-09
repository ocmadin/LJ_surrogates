import mdtraj
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#
# low_energy_pdb = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/efa0e5a8482743cd8b138a591cdbe32d/1968a6338087426ca21539c942bcc1a7_equilibration_simulation_component_1/input.pdb'
# low_energy_dcd = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/efa0e5a8482743cd8b138a591cdbe32d/1968a6338087426ca21539c942bcc1a7_equilibration_simulation_component_1/trajectory.dcd'
# low_energy_csv = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/efa0e5a8482743cd8b138a591cdbe32d/1968a6338087426ca21539c942bcc1a7_equilibration_simulation_component_1/openmm_statistics.csv'
# high_energy_pdb = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/8b5ee67d6df8478e878c87248d4cc201/1968a6338087426ca21539c942bcc1a7_equilibration_simulation_component_1/input.pdb'
# high_energy_dcd = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/8b5ee67d6df8478e878c87248d4cc201/1968a6338087426ca21539c942bcc1a7_equilibration_simulation_component_1/trajectory.dcd'
# high_energy_csv = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/8b5ee67d6df8478e878c87248d4cc201/1968a6338087426ca21539c942bcc1a7_equilibration_simulation_component_1/openmm_statistics.csv'

low_energy_pdb = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/efa0e5a8482743cd8b138a591cdbe32d/1968a6338087426ca21539c942bcc1a7_conditional_group_component_1/1968a6338087426ca21539c942bcc1a7_production_simulation_component_1/input.pdb'
low_energy_dcd = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/efa0e5a8482743cd8b138a591cdbe32d/1968a6338087426ca21539c942bcc1a7_conditional_group_component_1/1968a6338087426ca21539c942bcc1a7_production_simulation_component_1/trajectory.dcd'
low_energy_csv = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/efa0e5a8482743cd8b138a591cdbe32d/1968a6338087426ca21539c942bcc1a7_conditional_group_component_1/1968a6338087426ca21539c942bcc1a7_production_simulation_component_1/openmm_statistics.csv'
high_energy_pdb = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/8b5ee67d6df8478e878c87248d4cc201/1968a6338087426ca21539c942bcc1a7_conditional_group_component_1/1968a6338087426ca21539c942bcc1a7_production_simulation_component_1/input.pdb'
high_energy_dcd = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/8b5ee67d6df8478e878c87248d4cc201/1968a6338087426ca21539c942bcc1a7_conditional_group_component_1/1968a6338087426ca21539c942bcc1a7_production_simulation_component_1/trajectory.dcd'
high_energy_csv = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/8b5ee67d6df8478e878c87248d4cc201/1968a6338087426ca21539c942bcc1a7_conditional_group_component_1/1968a6338087426ca21539c942bcc1a7_production_simulation_component_1/openmm_statistics.csv'


# packmol_output_5 = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/f41462077a03455b9e17335ab3aee5c4/1968a6338087426ca21539c942bcc1a7_energy_minimisation_mixture/minimised.pdb'
# packmol_output_4 = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/f28d03eae52c49c2a5500cb0f5b1c542/1968a6338087426ca21539c942bcc1a7_energy_minimisation_mixture/minimised.pdb'
# packmol_output_3 = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/efa0e5a8482743cd8b138a591cdbe32d/1968a6338087426ca21539c942bcc1a7_energy_minimisation_mixture/minimised.pdb'
# packmol_output_2 = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/8925297fb885456da1c9c05cf6b24962/1968a6338087426ca21539c942bcc1a7_energy_minimisation_mixture/minimised.pdb'
# packmol_output_1 = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test/working_directory/SimulationLayer/8b5ee67d6df8478e878c87248d4cc201/1968a6338087426ca21539c942bcc1a7_energy_minimisation_mixture/minimised.pdb'

packmol_output_5 = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test-2-0-0/phase-separation-test-2-0-0-traj/SimulationLayer/fdd27955c7e9422fb9a2fad8b005e191/1968a6338087426ca21539c942bcc1a7_energy_minimisation_component_1/minimised.pdb'
packmol_output_4 = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test-2-0-0/phase-separation-test-2-0-0-traj/SimulationLayer/cf6d6422536b48d18bd1add9f2f3dca4/1968a6338087426ca21539c942bcc1a7_energy_minimisation_component_1/minimised.pdb'
packmol_output_3 = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test-2-0-0/phase-separation-test-2-0-0-traj/SimulationLayer/980f971a6a2f4bf092dde9f588de13f3/1968a6338087426ca21539c942bcc1a7_energy_minimisation_component_1/minimised.pdb'
packmol_output_2 = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test-2-0-0/phase-separation-test-2-0-0-traj/SimulationLayer/80ace7919e22477ab4c6f131bf5bc264/1968a6338087426ca21539c942bcc1a7_energy_minimisation_component_1/minimised.pdb'
packmol_output_1 = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/phase-separation-test-2-0-0/phase-separation-test-2-0-0-traj/SimulationLayer/9e05210483e6434b8c061e8b18937480/1968a6338087426ca21539c942bcc1a7_energy_minimisation_component_1/minimised.pdb'

traj = mdtraj.load([packmol_output_1, packmol_output_2, packmol_output_3, packmol_output_4, packmol_output_5])

low_energy_stats = pd.read_csv(low_energy_csv)
high_energy_stats = pd.read_csv(high_energy_csv)

packmol_low_energy = mdtraj.load(packmol_output_1)
packmol_high_energy = mdtraj.load(packmol_output_3)

low_energy = mdtraj.load_dcd(low_energy_dcd, top=low_energy_pdb)
high_energy = mdtraj.load_dcd(high_energy_dcd, top=high_energy_pdb)

oh_pairs_low = []
oh_pairs_high = []
oh_pairs_off_low = []
oh_pairs_off_high = []
co_pairs_low = []
co_pairs_high = []
cc_pairs_off_low = []
cc_pairs_off_high = []
angle_low = []
angle_high = []
torsion_low = []
torsion_high = []
packmol_torsion_low = []
packmol_torsion_high = []
for i in range(1000):
    co_pairs_low.append(low_energy.top.select_pairs(f"name H12 and resid {i}", f"name H14 and resid {i}"))
    co_pairs_high.append(high_energy.top.select_pairs(f"name H12 and resid {i}", f"name H14 and resid {i}"))
    oh_pairs_off_low.append(low_energy.top.select_pairs(f"name O1 and resid {i}", f"name H14 and resid != {i}"))
    oh_pairs_off_high.append(high_energy.top.select_pairs(f"name O1 and resid {i}", f"name H14 and resid != {i}"))
    cc_pairs_off_low.append(low_energy.top.select_pairs(f"name C6 and resid {i}", f"name C6 and resid != {i}"))
    cc_pairs_off_high.append(high_energy.top.select_pairs(f"name C6 and resid {i}", f"name C6 and resid != {i}"))
    oh_pairs_low.append(low_energy.top.select_pairs(f"name O1 and resid {i}", f"name H14 and resid {i}"))
    oh_pairs_high.append(high_energy.top.select_pairs(f"name O1 and resid {i}", f"name H14 and resid {i}"))
    angle_high.append(np.concatenate(
        [high_energy.top.select(f"name C6 and resid {i}"), high_energy.top.select(f"name O1 and resid {i}"),
         high_energy.top.select(f"name H14 and resid {i}")]))
    angle_low.append(np.concatenate(
        [low_energy.top.select(f"name C6 and resid {i}"), low_energy.top.select(f"name O1 and resid {i}"),
         low_energy.top.select(f"name H14 and resid {i}")]))
    torsion_low.append(np.concatenate(
        [low_energy.top.select(f"name C2 and resid {i}"), low_energy.top.select(f"name C3 and resid {i}"),
         low_energy.top.select(f"name C4 and resid {i}"), low_energy.top.select(f"name C5 and resid {i}")]))
    torsion_high.append(np.concatenate(
        [high_energy.top.select(f"name C2 and resid {i}"), high_energy.top.select(f"name C3 and resid {i}"),
         high_energy.top.select(f"name C4 and resid {i}"), high_energy.top.select(f"name C5 and resid {i}")]))
    packmol_torsion_low.append(np.concatenate(
        [packmol_low_energy.top.select(f"name C4 and resid {i}"),
         packmol_low_energy.top.select(f"name C5 and resid {i}"),
         packmol_low_energy.top.select(f"name C6 and resid {i}"),
         packmol_low_energy.top.select(f"name O1 and resid {i}")]))
    packmol_torsion_high.append(np.concatenate(
        [packmol_high_energy.top.select(f"name C4 and resid {i}"),
         packmol_high_energy.top.select(f"name C5 and resid {i}"),
         packmol_high_energy.top.select(f"name C6 and resid {i}"),
         packmol_high_energy.top.select(f"name O1 and resid {i}")]))

oh_pairs_off_low = np.asarray(oh_pairs_off_low).reshape(999000, 2)
oh_pairs_off_high = np.asarray(oh_pairs_off_high).reshape(999000, 2)
cc_pairs_off_low = np.asarray(cc_pairs_off_low).reshape(999000, 2)
cc_pairs_off_high = np.asarray(cc_pairs_off_high).reshape(999000, 2)
oh_pairs_low = np.asarray(oh_pairs_low).squeeze(1)
oh_pairs_high = np.asarray(oh_pairs_high).squeeze(1)
co_pairs_low = np.asarray(co_pairs_low).squeeze(1)
co_pairs_high = np.asarray(co_pairs_high).squeeze(1)
angle_low = np.asarray(angle_low)
angle_high = np.asarray(angle_high)
torsion_low = np.asarray(torsion_low)
torsion_high = np.asarray(torsion_high)
packmol_torsion_low = np.asarray(packmol_torsion_low)
packmol_torsion_high = np.asarray(packmol_torsion_high)

# rdf_low_off = mdtraj.compute_rdf(low_energy, oh_pairs_off_low, r_range=(0, 1), n_bins=500)
# rdf_high_off = mdtraj.compute_rdf(high_energy, oh_pairs_off_high, r_range=(0, 1), n_bins=500)
#
# rdf_low = mdtraj.compute_rdf(low_energy, oh_pairs_low, r_range=(0.095, 0.1), n_bins=500)
# rdf_high = mdtraj.compute_rdf(high_energy, oh_pairs_high, r_range=(0.095, 0.1), n_bins=500)
#
# rdf_co_low = mdtraj.compute_rdf(low_energy, co_pairs_low, r_range=(0, 1), n_bins=500)
# rdf_co_high = mdtraj.compute_rdf(high_energy, co_pairs_high, r_range=(0, 1), n_bins=500)
#
# rdf_cc_low = mdtraj.compute_rdf(low_energy, cc_pairs_off_low, r_range=(0, 1), n_bins=500)
# rdf_cc_high = mdtraj.compute_rdf(high_energy, cc_pairs_off_high, r_range=(0, 1), n_bins=500)

# plt.plot(*rdf_low, label='Low Energy')
# plt.plot(*rdf_high, label='High Energy')
# plt.title('Intra-molecule O-H RDF')
# plt.legend()
# plt.show()
#
# plt.plot(*rdf_low_off, label='Low Energy')
# plt.plot(*rdf_high_off, label='High Energy')
# plt.title('Inter-molecule O-H RDF')
# plt.legend()
# plt.show()
#
# plt.plot(*rdf_co_low, label='Low Energy')
# plt.plot(*rdf_co_high, label='High Energy')
# plt.title('Intra-molecule C-O RDF')
# plt.legend()
# plt.show()
#
# plt.plot(*rdf_cc_low, label='Low Energy')
# plt.plot(*rdf_cc_high, label='High Energy')
# plt.title('Inter-molecule C-C RDF')
# plt.legend()
# plt.show()

import openmm
from simtk.openmm.app import PDBFile

pdb = PDBFile(high_energy_pdb)

torsion_values_low = mdtraj.compute_dihedrals(low_energy, torsion_low)
torsion_values_high = mdtraj.compute_dihedrals(high_energy, torsion_high)

packmol_torsion_values_low = mdtraj.compute_dihedrals(packmol_low_energy, packmol_torsion_low)
packmol_torsion_values_high = mdtraj.compute_dihedrals(packmol_high_energy, packmol_torsion_high)

os.makedirs('torsion-plots', exist_ok=True)
# for i in range(torsion_values_low.shape[0]):
#     plt.hist(torsion_values_high[i],alpha=0.5,label='High',bins=50)
#     plt.hist(torsion_values_low[i],alpha=0.5,label='Low',bins=50)
#     plt.legend()
#     plt.xlabel('Angle (rads)')
#     plt.savefig(os.path.join('torsion-plots',f'H-C-O-H_{i}.png'), dpi=300)
#     plt.cla()

for i in range(packmol_torsion_values_low.shape[0]):
    plt.hist(packmol_torsion_values_high[i], alpha=0.5, label='High', bins=100)
    plt.hist(packmol_torsion_values_low[i], alpha=0.5, label='Low', bins=100)
    plt.legend()
    plt.xlabel('Angle (rads)')
    plt.show()
    plt.cla()

plt.hist(torsion_values_high[:,661], alpha = 0.5, label='High', bins=50)
plt.hist(torsion_values_low[:,661], alpha = 0.5, label='Low', bins=50)
plt.legend()
plt.show()