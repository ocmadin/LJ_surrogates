import json
import os
import pandas
import copy
from nonbonded.library.utilities.checkmol import components_to_categories
from nonbonded.library.utilities.environments import ChemicalEnvironment
from nonbonded.library.models.datasets import Component
import re
import numpy as np


class SimulatedDataset:
    def __init__(self, dataset_collection, estimated_data_set):
        self.dataset_collection = dataset_collection
        self.estimated_data_set = estimated_data_set
        self.chemical_environments = [
            ChemicalEnvironment.Alkane,
            ChemicalEnvironment.Alkene,
            ChemicalEnvironment.Alcohol,
            ChemicalEnvironment.CarbonylHydrate,
            ChemicalEnvironment.Hemiacetal,
            ChemicalEnvironment.Acetal,
            ChemicalEnvironment.Hemiaminal,
            ChemicalEnvironment.Aminal,
            ChemicalEnvironment.Thioacetal,
            ChemicalEnvironment.CarboxylicAcidEster,
            ChemicalEnvironment.Ether,
            ChemicalEnvironment.Aldehyde,
            ChemicalEnvironment.Ketone,
            ChemicalEnvironment.Aromatic,
            ChemicalEnvironment.CarboxylicAcidPrimaryAmide,
            ChemicalEnvironment.CarboxylicAcidSecondaryAmide,
            ChemicalEnvironment.CarboxylicAcidTertiaryAmide,
            ChemicalEnvironment.PrimaryAmine,
            ChemicalEnvironment.SecondaryAmine,
            ChemicalEnvironment.TertiaryAmine,
            ChemicalEnvironment.Cyanate,
            ChemicalEnvironment.Isocyanate,
            ChemicalEnvironment.Heterocycle,
            ChemicalEnvironment.AlkylFluoride,
            ChemicalEnvironment.ArylFluoride,
            ChemicalEnvironment.AlkylChloride,
            ChemicalEnvironment.ArylChloride,
            ChemicalEnvironment.AlkylBromide,
            ChemicalEnvironment.ArylBromide,
            ChemicalEnvironment.Aqueous,
            ChemicalEnvironment.CarboxylicAcid
        ]

    def build_dataframe(self):
        df = pandas.DataFrame(
            columns=['Property Type', 'N Components', 'Component 1', 'Amount 1', 'Component 2', 'Amount 2',
                     'Temperature (K)', 'Pressure (kPa)',
                     'Reference Value',
                     'Reference Uncertainty',
                     'Estimated Value', 'Estimated Uncertainty', 'Category 1', 'Category 2'])

        with open(self.dataset_collection) as f:
            dataset_collection = json.load(f)
        with open(self.estimated_data_set) as f:
            estimated_data_set = json.load(f)

        for entry in dataset_collection['properties']:
            if len(entry['substance']['components']) > 1:
                df.loc[entry['id']] = pandas.Series({'Property Type': entry['@type'],
                                                     'N Components': len(entry['substance']['components']),
                                                     'Component 1': entry['substance']['components'][0]['smiles'],
                                                     'Amount 1': entry['substance']['amounts'][
                                                         entry['substance']['components'][0]['smiles'] + '{solv}'][0][
                                                         'value'],
                                                     'Component 2': entry['substance']['components'][1]['smiles'],
                                                     'Amount 2': entry['substance']['amounts'][
                                                         entry['substance']['components'][1]['smiles'] + '{solv}'][0][
                                                         'value'],
                                                     'Temperature (K)': entry['thermodynamic_state']['temperature'][
                                                         'value'],
                                                     'Pressure (kPa)': entry['thermodynamic_state']['pressure'][
                                                         'value'],
                                                     'Reference Value': entry['value']['value']})
            else:
                df.loc[entry['id']] = pandas.Series({'Property Type': entry['@type'],
                                                     'N Components': len(entry['substance']['components']),
                                                     'Component 1': entry['substance']['components'][0]['smiles'],
                                                     'Amount 1': entry['substance']['amounts'][
                                                         entry['substance']['components'][0]['smiles'] + '{solv}'][0][
                                                         'value'],
                                                     'Temperature (K)': entry['thermodynamic_state']['temperature'][
                                                         'value'],
                                                     'Pressure (kPa)': entry['thermodynamic_state']['pressure'][
                                                         'value'],
                                                     'Reference Value': entry['value']['value']})
            if 'uncertainty' in entry.keys():
                df.at[entry['id'], 'Reference Uncertainty'] = entry['uncertainty']['value']

            for estimated_entry in estimated_data_set['estimated_properties']['properties']:
                if entry['id'] == estimated_entry['id']:
                    df.at[entry['id'], 'Estimated Value'] = estimated_entry['value']['value']
                    df.at[entry['id'], 'Estimated Uncertainty'] = estimated_entry['uncertainty']['value']

            components = []
            for component in entry['substance']['components']:
                components.append(Component(smiles=component['smiles'],
                                            mole_fraction=
                                            entry['substance']['amounts'][component['smiles'] + '{solv}'][0][
                                                'value']))
            categories = components_to_categories(components, self.chemical_environments)

            for i, category in enumerate(categories):
                categories[i] = re.sub('[~,>,<]', lambda x: '+', category)

            df['Category 1'].loc[entry['id']] = categories[0]

            if len(categories) > 1:
                df['Category 2'].loc[entry['id']] = categories[1]

        self.df = df

    def split_property_dataframes(self):
        self.enthalpy_of_vaporization = self.df[
            self.df['Property Type'] == 'openff.evaluator.properties.enthalpy.EnthalpyOfVaporization']
        self.enthalpy_of_mixing = self.df[
            self.df['Property Type'] == 'openff.evaluator.properties.enthalpy.EnthalpyOfMixing']
        self.binary_density = self.df[self.df['Property Type'] == 'openff.evaluator.properties.density.Density']
        self.binary_density = self.binary_density[self.binary_density['N Components'] == 2]
        self.pure_density = self.df[self.df['Property Type'] == 'openff.evaluator.properties.density.Density']
        self.pure_density = self.pure_density[self.pure_density['N Components'] == 1]

    def calculate_rmses(self):
        dataframes = [self.enthalpy_of_vaporization, self.enthalpy_of_mixing, self.pure_density, self.binary_density]

        rmses = []
        for dataframe in dataframes:
            if dataframe.shape[0] > 0:
                rmse = np.sqrt(np.mean(np.square(dataframe['Reference Value'] - dataframe['Estimated Value'])))
            else:
                rmse = np.nan
            rmses.append(rmse)
        return rmses

    def calculate_rmse_by_category(self, dataframe):
        categories = set(dataframe['Category 1'].dropna()).union(set(dataframe['Category 2'].dropna()))

        rmses = {}
        for category in categories:
            cols = ['Category 1', 'Category 2']
            cat_df = copy.deepcopy(dataframe)
            cat_df["multiple"] = (dataframe[cols] == category).any(axis="columns")

            cat_df = cat_df[cat_df['multiple'] == True]
            rmses[category] = np.sqrt(np.mean(np.square(cat_df['Reference Value'] - cat_df['Estimated Value'])))

        return rmses

    def plot_parity(self, dataframe, property, category):
        rmse = np.sqrt(np.mean(np.square(dataframe['Reference Value'] - dataframe['Estimated Value'])))
        os.makedirs(os.path.join('plots', 'parity', property),exist_ok=True)
        mse = np.mean(dataframe['Estimated Value'] - dataframe['Reference Value'])
        max = np.max(np.concatenate((dataframe['Reference Value'].values, dataframe['Estimated Value'].values),
                                    axis=None))
        min = np.min(
            np.concatenate((dataframe['Reference Value'].values, dataframe['Estimated Value'].values), axis=None))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.errorbar(dataframe['Reference Value'], dataframe['Estimated Value'],
                       yerr=dataframe['Estimated Uncertainty'], xerr=dataframe['Reference Uncertainty'], ls='none',
                       marker='.')
        ax.set_ylabel('Simulation Value', fontsize=14)
        ax.set_xlabel('Experimental Value', fontsize=14)
        ax.plot([min, max], [min, max], color='k')
        fig.suptitle(
            f'{property} \n {category} parity plot \n RMSE={round(rmse,2)}, Mean signed error={round(mse, 2)}, n={dataframe.shape[0]}',
            fontsize=14)
        fig.tight_layout()
        plt.savefig(os.path.join('plots', 'parity', property, category + '.png'))
        plt.close(fig)

    def plot_parity_comparison(self, dataframes, labels, property, category):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(len(dataframes)*5, 5))
        for i, dataframe in enumerate(dataframes):
            rmse = np.sqrt(np.mean(np.square(dataframe['Reference Value'] - dataframe['Estimated Value'])))
            os.makedirs(os.path.join('plots', 'parity-comparison', property),exist_ok=True)
            mse = np.mean(dataframe['Estimated Value'] - dataframe['Reference Value'])
            max = np.max(np.concatenate((dataframe['Reference Value'].values, dataframe['Estimated Value'].values),
                                        axis=None))
            min = np.min(
                np.concatenate((dataframe['Reference Value'].values, dataframe['Estimated Value'].values), axis=None))

            ax[i].errorbar(dataframe['Reference Value'], dataframe['Estimated Value'],
                           yerr=dataframe['Estimated Uncertainty'], xerr=dataframe['Reference Uncertainty'], ls='none',
                           marker='.')
            ax[i].set_ylabel('Simulation Value', fontsize=14)
            ax[i].set_xlabel('Experimental Value', fontsize=14)
            ax[i].plot([min, max], [min, max], color='k')
            ax[i].set_title(
                f'{labels[i]} \n RMSE={round(rmse,2)}, Mean signed error={round(mse, 2)}, n={dataframe.shape[0]}',
                fontsize=14)
        fig.suptitle(f'{property} {category} parity plot', fontsize=14)
        fig.tight_layout()
        plt.savefig(os.path.join('plots', 'parity-comparison', property, category + '.png'))
        plt.close(fig)


    def plot_category_parities(self, dataframe, property):

        self.plot_parity(dataframe, property, 'All')

        categories = set(dataframe['Category 1'].dropna()).union(set(dataframe['Category 2'].dropna()))

        for category in categories:
            cols = ['Category 1', 'Category 2']
            cat_df = copy.deepcopy(dataframe)
            cat_df["multiple"] = (dataframe[cols] == category).any(axis="columns")

            cat_df = cat_df[cat_df['multiple'] == True]
            self.plot_parity(cat_df, property, category)

    def plot_all_parity(self):
        if self.enthalpy_of_mixing.shape[0] > 0:
            self.plot_category_parities(self.enthalpy_of_mixing,'Enthalpy Of Mixing')
        if self.enthalpy_of_vaporization.shape[0] > 0:
            self.plot_category_parities(self.enthalpy_of_vaporization,'Enthalpy Of Vaporization')
        if self.pure_density.shape[0] > 0:
            self.plot_category_parities(self.pure_density,'Pure Density')
        if self.binary_density.shape[0] > 0:
            self.plot_category_parities(self.binary_density,'Binary Density')