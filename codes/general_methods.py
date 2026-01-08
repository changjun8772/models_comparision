import numpy as np
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
import pickle
import pandas as pd


def Load_model(filepath: str) -> None:
    # 加载RFC模型
    with open(filepath, 'rb') as fr:
        model = pickle.load(fr)
        return model


# 进行SMILES标准化的方法
def get_canoncialsmiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return None
    except:
        return None

# ECFP
def GetFPS(smiles_list: list) -> np.array:
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    fp = np.array(
        [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols]
    )
    return fp


def get_valid_smiles(smiles_list):
    valid_smiles = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                valid_smiles.append(smi)
        except:
            pass
    return valid_smiles
