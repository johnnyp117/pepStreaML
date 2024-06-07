# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from io import StringIO
from typing import Union

import numpy as np
import biotite
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
from biotite.structure import filter

def pdb_file_to_atomarray(pdb_path: Union[str, StringIO], chain_id: str ="A") -> AtomArray:
    full_structure = PDBFile.read(pdb_path).get_structure(model=1)
    # chain_mask = filter(full_structure, f"chain_id == '{chain_id}'")
    full_structure = full_structure[full_structure.hetero == False] # can be replaced with biotite.structure.filter_amino_acids
    full_structure = full_structure[full_structure.element != 'H' ]
    # remove residues with non-numeric res_num using insertion codes
    full_structure = full_structure[full_structure.ins_code == '' ]
    chain_mask = full_structure.chain_id == chain_id
    return full_structure[chain_mask]

# def pdb_file_to_atomarray(pdb_path: Union[str, StringIO]) -> AtomArray:
#     return PDBFile.read(pdb_path).get_structure(model=1)


def get_atomarray_in_residue_range(atoms: AtomArray, start: int, end: int) -> AtomArray:
    return atoms[np.logical_and(atoms.res_id >= start, atoms.res_id < end)]

def get_atomarray_in_residue_list(atoms: AtomArray, residue_list: list) -> AtomArray:
    return atoms[np.isin(atoms.res_id, residue_list)]
