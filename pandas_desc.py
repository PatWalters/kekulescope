#!/usr/bin/env python

import sys
import os
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm


def molecule_supplier_from_name(input_file_name):
    ext = os.path.splitext(input_file_name)[-1]
    if ext == ".smi":
        suppl = Chem.SmilesMolSupplier(input_file_name, titleLine=False)
    elif ext == ".sdf":
        suppl = Chem.SDMolSupplier(input_file_name)
    else:
        print("%s is not a valid molecule extension" % ext)
        sys.exit(0)
    return suppl


class PandasDescriptors:
    def __init__(self, fp_type_list, num_fp_bits=1024):
        self.num_fp_bits = num_fp_bits
        self.fp_function_list = []
        self.fp_type_list = fp_type_list
        self.fp_dict = {}
        self.des_names = [name[0] for name in Descriptors._descList]
        des_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self.des_names)
        self.fp_dict['descriptors'] = [lambda m: des_calculator.CalcDescriptors(m), -1]
        self.fp_dict['morgan2'] = [lambda m: rdmd.GetMorganFingerprintAsBitVect(m, 2, nBits=self.num_fp_bits),
                                   self.num_fp_bits]
        self.fp_dict['morgan3'] = [lambda m: rdmd.GetMorganFingerprintAsBitVect(m, 3, nBits=self.num_fp_bits),
                                   self.num_fp_bits]
        self.fp_dict['ap'] = [lambda m: rdmd.GetHashedAtomPairFingerprintAsBitVect(m, nBits=self.num_fp_bits),
                              self.num_fp_bits]
        self.fp_dict['rdk5'] = [lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=self.num_fp_bits, nBitsPerHash=2),
                                self.num_fp_bits]
        self.fp_names = []
        for fp_type in fp_type_list:
            if self.fp_dict.get(fp_type):
                self.fp_function_list.append(self.fp_dict[fp_type])
                if fp_type == "descriptors":
                    self.fp_names += self.des_names
                else:
                    self.fp_names += self.get_names(self.num_fp_bits)
            else:
                print("invalid fingerprint type: %s" % fp_type)
                sys.exit(1)

    def set_fp_bits(self, num_bits):
        self.num_fp_bits = num_bits

    def get_fp_types(self):
        return [x for x in self.fp_dict.keys() if x != "ref"]

    def get_names(self, num_bits):
        if num_bits > 0:
            name_list = ["%s_%d" % ("B", i) for i in range(0, num_bits)]
        else:
            name_list = self.des_names
        return name_list

    def get_fp(self, mol):
        fp_list = []
        for fp_function in self.fp_function_list:
            fp_list.append(fp_function[0](mol))
        return fp_list

    def get_descriptors(self, mol):
        master_fp = np.array([], dtype=np.int)
        for fp_function, num_fp_bits in self.fp_function_list:
            fp = fp_function(mol)
            arr = np.zeros((1,), np.int)
            if num_fp_bits > 0:
                DataStructs.ConvertToNumpyArray(fp, arr)
            else:
                arr = np.array(fp)
            master_fp = np.append(master_fp, arr)
        return master_fp

    def dataframe_from_list(self, fp_list, name_list, scale_data):
        df = pd.DataFrame(fp_list)
        if scale_data:
            min_max_scaler = preprocessing.MinMaxScaler()
            df = pd.DataFrame(min_max_scaler.fit_transform(df))
        df.columns = self.fp_names
        df.insert(0, "Name", name_list)
        return df

    def from_molecule_file(self, input_file_name, name_field="_Name", activity_field=None, scale_data=False):
        suppl = molecule_supplier_from_name(input_file_name)
        name_list = []
        fp_list = []
        activity_list = []
        for mol in tqdm(suppl):
            name_list.append(mol.GetProp(name_field))
            fp = self.get_descriptors(mol)
            fp_list.append(fp)
            if activity_field:
                activity_list.append(float(mol.GetProp(activity_field)))
        output_df = self.dataframe_from_list(fp_list, name_list, scale_data)
        if activity_field:
            output_df.insert(1, activity_field, activity_list)
        return output_df

    def from_dataframe(self, input_df, smiles_column='SMILES', name_column='Name', scale_data=False):
        name_list = []
        fp_list = []
        for smiles, name in input_df[[smiles_column, name_column]].values:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                name_list.append(name)
                fp = self.get_descriptors(mol)
                fp_list.append(fp)
        fp_df = self.dataframe_from_list(fp_list, name_list, scale_data)
        new_df = input_df.merge(fp_df, left_on=name_column, right_on="Name")
        return new_df


def test(infile_name, outfile_name, name_field="_Name", activity_field=None):
    pd_desc = PandasDescriptors(['morgan2', 'descriptors'])
    df = pd_desc.from_molecule_file(infile_name, name_field=name_field, activity_field=activity_field)
    df.to_csv(outfile_name, index=False)


if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2], name_field="ChEMBL_ID", activity_field="pIC50")
