import os, sys, re
import torch
import esm
import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import logomaker
from optparse import OptionParser
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from biotite.database.rcsb import fetch
from biotite.structure import AtomArray
import biotite.structure.io as bsio
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from language import (
    MinimizeTargetDistance,
    ConstantSequenceSegment,
    VariableLengthSequenceSegment,
    FixedLengthSequenceSegment,
    MaximizeGlobularity,
    MaximizePLDDT,
    MaximizePTM,
    MinimizeSurfaceExposure,
    MaximizeSurfaceExposure,
    MinimizeCRmsd,
    MinimizeDRmsd,
    MinimizeSurfaceHydrophobics,
    ProgramNode,
    SymmetryRing,
    get_atomarray_in_residue_range,
    get_atomarray_in_residue_list,
    pdb_file_to_atomarray,
    sequence_from_atomarray,
)

from language import EsmFoldv1
from language import run_simulated_annealing
####
# pepstream core peptide prediction routine
# 
# This can run with only a sequence or using a structure. 
# If passing in a sequence the alphafold uniprot library will be pulled to acquire a structure
# The esm design language will be implemented such that this script can control most of the important 
#     parameters of interest to this design
# The exact design routine is optimizing closness to the target site along with an energy calculation for stability
#
####


folding_callback = EsmFoldv1()
folding_callback.load(device="cuda:0")
def get_alphafold_download_link(uniprot_id):
	link_pattern = 'https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v2.pdb'
	return link_pattern.format(uniprot_id)

def download_alphafold_prediction(uniprot_id):
	url = get_alphafold_download_link(uniprot_id)
	result = subprocess.run(['wget', url, '-O', uniprot_id+'.pdb'])
	return result   # Result will be 0 if operation was successful.

def parse_binding_site(binding_site_str):
    # Split by comma
    binding_sites_str = binding_site_str.split(',')

    binding_sites_indices = []
    for binding_site in binding_sites_str:
        # Split by hyphen and convert each to int
        start_end = list(map(int, binding_site.split('-')))
        if len(start_end) != 2:
            raise ValueError("Please provide start and end values for each binding site range.")
        binding_sites_indices.append(tuple(start_end))
    
    return binding_sites_indices

def predict_b_factor_mean(sequence, name):
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    # Optionally, uncomment to set a chunk size for axial attention.
    # model.set_chunk_size(128)

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(f"result_{str(name)}.pdb", "w") as f:
        f.write(output) 

    struct = bsio.load_structure(f"result_{name}.pdb", extra_fields=["b_factor"])
    mean_b_factor = struct.b_factor.mean()

    return mean_b_factor
#### --------------------------------------------
def symmetric_monomer(num_protomers: int) -> ProgramNode:
# fixed length only?
    protomer_sequence = FixedLengthSequenceSegment(50)
# This is needed for all methods to initalize the optimization for the new construct?
    def _make_protomer_node():
        # A new ProgramNode must be made for each new protomer,
        # but the sequence can (and should) be shared.
        return ProgramNode(
            sequence_segment=protomer_sequence
        )
    
# This is how symmetry is applied in this case a 3 fold symmetry of a monomer
#    N = 3
#    _node = ProgramNode(
#        children=[make_protomer_node() for _ in range(N)],
#    )
# This method is the actual run command for the design
    return ProgramNode(
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            SymmetryRing(),
            MinimizeSurfaceHydrophobics(),
        ],
        children=[
            _make_protomer_node()
            for _ in range(num_protomers)
        ],
    )

# I could just snip parts that are far away to make peptides? this takes a motif and builds around it
def symmetric_binding_il10(num_binding_sites, binding_site_atoms) -> ProgramNode: 

    binding_site_sequence: str = sequence_from_atomarray(binding_site_atoms)

    leader_amino_acid_sequence = FixedLengthSequenceSegment(45)
    #VariableLengthSequenceSegment()
    binding_site_sequence = ConstantSequenceSegment(binding_site_sequence)
    follower_amino_acid_sequence = FixedLengthSequenceSegment(45)

    def _binder_protomer_program() -> ProgramNode:
        return ProgramNode(
            children=[
                ProgramNode(sequence_segment=leader_amino_acid_sequence),
                ProgramNode(
                    sequence_segment=binding_site_sequence,
                    energy_function_terms=[
                        MaximizeSurfaceExposure(),
                        MinimizeCRmsd(template=binding_site_atoms),
                        MinimizeDRmsd(template=binding_site_atoms),
                    ],
                    energy_function_weights=[1.0, 10.0, 10.0],
                ),
                ProgramNode(sequence_segment=follower_amino_acid_sequence),
            ]
        )


    return ProgramNode(
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            SymmetryRing(),
            MinimizeSurfaceHydrophobics(),
        ],
        children=[_binder_protomer_program() for _ in range(num_binding_sites)],
    )

def symmetric_two_level_multimer(
    num_chains: int,
    num_protomers_per_chain: int,
    protomer_sequence_length: int = 50,
) -> ProgramNode:
    """
    Programs a homo-oligomeric protein with two-level symmetry.
    The number of chains in the multimer is specified by `num_chains`.
    A protomer sequence, with length `protomer_sequence_length`, is
    repeated `num_protomers_per_chain` times.
    For example, a two-chain protein with three protomers per chain
    would repeat the protomer six times.
    """

    # The basic repeated unit.
    protomer_sequence = FixedLengthSequenceSegment(protomer_sequence_length)
    def _make_protomer_node():
        return ProgramNode(sequence_segment=protomer_sequence)

    # Protomers are symmetrically combined into a chain.
    def _make_chain_node():
        return ProgramNode(
            energy_function_terms=[
                SymmetryRing(),
                MaximizeGlobularity()
            ],
            energy_function_weights=[1., 0.05,],
            children=[
                _make_protomer_node()
                for _ in range(num_protomers_per_chain)
            ],
        )

    # Chains are symmetrically combined into a multimer.
    return ProgramNode(
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            SymmetryRing(),
            MinimizeSurfaceHydrophobics(),
        ],
        children=[
            _make_chain_node()
            for _ in range(num_chains)
        ],
        children_are_different_chains=True,
    )

def asymmetric_binding_complex(structure_atoms, binding_site_atoms, min_length=30, max_length=70) -> ProgramNode:
    # The sequence of the binding site is fixed
    structure_site_sequence: str = sequence_from_atomarray(structure_atoms)
    structure_site_sequence = ConstantSequenceSegment(structure_site_sequence)
    # dis will make more than 1 peptwide (UwU)
    num_protomers_per_chain = 1
    # The other chain can vary in length
    rng = np.random.default_rng()

    # Define the binding site ProgramNode
    def _binding_site_node():
        return ProgramNode(
            sequence_segment=structure_site_sequence,
            energy_function_terms=[
                MaximizePLDDT()
#                MinimizeCRmsd(template=structure_atoms,backbone_only=True),
#                MinimizeDRmsd(template=structure_atoms,backbone_only=True),
            ],
            energy_function_weights=[1.0]
        )

    # Define the variable length chain ProgramNode
    def _variable_length_chain_node():
        variable_length_sequence = VariableLengthSequenceSegment(rng.integers(min_length,max_length))
        return ProgramNode(sequence_segment=variable_length_sequence)

    # Combine the two chains in a complex
    return ProgramNode(
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            MinimizeSurfaceHydrophobics(),
            MinimizeSurfaceExposure(),
            MaximizeGlobularity(),
            MinimizeTargetDistance(target=binding_site_atoms,backbone_only=True)
        ],
        energy_function_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 10.0],
        # energy_function_weights=[1.0], 
        children=[_binding_site_node()]+
            [_variable_length_chain_node()  for _ in range(num_protomers_per_chain)],
        children_are_different_chains=True
    )

def asymmetric_binding_footprint(binding_site_atoms_array, conserved_atoms, min_length=10, max_length=70) -> ProgramNode:

    num_protomers_per_chain = 1
    # The other chain can vary in length
    rng = np.random.default_rng()

    # Define the binding site ProgramNode
    def _binding_site_node(binding_site_atoms):
        # The sequence of the binding site is fixed
        binding_site_sequence: str = sequence_from_atomarray(binding_site_atoms)
        binding_site_sequence = ConstantSequenceSegment(binding_site_sequence)
        return ProgramNode(
            sequence_segment=binding_site_sequence,
            energy_function_terms=[
                MaximizePLDDT(),
                # MinimizeCRmsd(template=binding_site_atoms,backbone_only=True),
                # MinimizeDRmsd(template=binding_site_atoms,backbone_only=True),
            ],
            # energy_function_weights=[1.0, 1.0, 1.0]
            energy_function_weights=[1.0]
        )

    # Define the variable length chain ProgramNode
    def _variable_length_chain_node():
        variable_length_sequence = VariableLengthSequenceSegment(rng.integers(min_length,max_length))
        return ProgramNode(sequence_segment=variable_length_sequence)

    # Combine the two chains in a complex
    return ProgramNode(
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            MinimizeSurfaceHydrophobics(),
            MinimizeSurfaceExposure(),
            MaximizeGlobularity(),
            MinimizeTargetDistance(target=conserved_atoms) # changed from binding_site_atoms
        ],
        energy_function_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 10.0],
        children=[_binding_site_node(site) for site in binding_site_atoms_array]+
            [_variable_length_chain_node()  for _ in range(num_protomers_per_chain)],
        children_are_different_chains=True
    )
#### --------------------------------------------

def main():
    parser = OptionParser()
    parser.add_option("-u", "--uniprot", dest="uniprot", help="Uniprot ID")
    parser.add_option("-p", "--pdb", dest="pdb", help="pdb file path")
    parser.add_option("-c", "--chain", dest="chain", help="pdb file chain of interest", default="A")
    parser.add_option("-b", "--binding-site", dest="binding_site", type="string",
                  help="binding site indices")
    parser.add_option("-t", "--temp", dest="initial_temperature", default=1.0, type=float,
                      help="initial temperature for simulated annealing")
    parser.add_option("-a", "--rate", dest="annealing_rate", default=0.97, type=float,
                      help="annealing rate for simulated annealing")
    parser.add_option("-n", "--steps", dest="total_num_steps", default=10000, type=int,
                      help="total number of steps for simulated annealing")
    parser.add_option("-d", "--display", dest="display_progress", action="store_true", default=True,
                      help="whether to display progress or not")
    parser.add_option("-o", "--output", dest="output_name", type="string", default="result",
                      help="name to append to output fasta and pdb file from this script")
    parser.add_option("-m", "--mode", dest="mode", type="string", default="target",
                      help="whole or target of strucutre for design optimization")
    
    (options, args) = parser.parse_args()
    
    if options.binding_site:
        start, end = parse_binding_site(options.binding_site)
    else:
        print("Please provide binding site indices.")
        return
    
    # binding_site_atoms: AtomArray = pdb_file_to_atomarray(fetch("1y6k", format="pdb"))
    
    # Depending on whether Uniprot ID or pdb file was provided
    if options.uniprot:
        # Fetch the sequence using the Uniprot ID
        # replace fetch_uniprot_sequence with the actual function that fetches the sequence using the Uniprot ID
        sequence = download_alphafold_prediction(options.uniprot) # fetch_uniprot_sequence(options.uniprot)
        # convert the sequence to atom array
    #    sequence_atoms = sequence_to_atomarray(sequence)
        structure_atoms: AtomArray = pdb_file_to_atomarray(options.uniprot+".pdb")
    elif options.pdb:
        # read pdb file and generate the necessary atom array for the sequence
        # replace pdb_file_to_atomarray with the actual function that reads the pdb file and generates the atom array
        # sequence_atoms = pdb_file_to_atomarray(options.pdb)
# the garbage method they made does not allow specifiying chain ID????, so should I override it here or there..... , options.chain        
        structure_atoms: AtomArray = pdb_file_to_atomarray(fetch(options.pdb, format="pdb"), options.chain )
    else:
        print("Please provide a Uniprot ID or a pdb file path.")
        return
## --------------------------------------------
    structure_atoms = get_atomarray_in_residue_range(structure_atoms, start=structure_atoms.res_id.min(), end=structure_atoms.res_id.max())
    binding_site_atoms = get_atomarray_in_residue_range(structure_atoms, start=start, end=end)

# Begin the prediction
    # program = symmetric_binding_il10(1, binding_site_atoms)
    if options.mode == "target":
        program = asymmetric_binding_footprint(binding_site_atoms, min_length=10, max_length=30)
    else:
        program = asymmetric_binding_complex(structure_atoms, binding_site_atoms, min_length=10, max_length=30)

    # optimized_program = run_simulated_annealing(
    # program=program,
    # initial_temperature=1.0,
    # annealing_rate=0.97,
    # total_num_steps=10_000,
    # folding_callback=folding_callback,
    # display_progress=True,
    # )
    
# Generate the sequence of the structure
    structure_sequence = sequence_from_atomarray(structure_atoms)
# Generate the sequence of the target
    target_sequence = sequence_from_atomarray(binding_site_atoms)
    
# Set up the program.
    inital_sequence, residue_indices = program.get_sequence_and_set_residue_index_ranges()
# Compute and print the energy function.
    energy_terms = program.get_energy_term_functions()
    folding_output = folding_callback.fold(inital_sequence, residue_indices)
    for name, weight, energy_fn in energy_terms:
        print(f"{name} = {weight:.1f} * {energy_fn(folding_output):.2f}")
    
    optimized_program = run_simulated_annealing(
    program=program,
    initial_temperature=options.initial_temperature,
    annealing_rate=options.annealing_rate,
    total_num_steps=options.total_num_steps,
    folding_callback=folding_callback,
    display_progress=options.display_progress,
    progress_verbose_print=True
    )

    optimized_sequence , residue_indices = optimized_program.get_sequence_and_set_residue_index_ranges()
    folding_output = folding_callback.fold(optimized_sequence, residue_indices)
    
    # print(folding_output)
    if options.mode == "target":
        print("Final sequence = {}".format(optimized_program.get_sequence_and_set_residue_index_ranges()[0][len(target_sequence):]))
        # Create the final sequence
        final_sequence = structure_sequence + ":" + optimized_sequence[len(target_sequence):]
    else:
        print("Final sequence = {}".format(optimized_program.get_sequence_and_set_residue_index_ranges()[0][len(structure_sequence):]))
        # Create the final sequence
        final_sequence = structure_sequence + ":" + optimized_sequence[len(structure_sequence):]
            
    print("Target sequence = {}".format(target_sequence))
    print("Final pLDDT = {}".format(folding_output.plddt))
    print("Final PTM = {}".format(folding_output.ptm))
    
    file = PDBFile()
    file.set_structure(folding_output.atoms)
    file.write(os.path.join(os.getcwd(), options.output_name+".pdb"))
    
# This is the original fold method from v1, this will NOT produce the designed complex as it lacks the index trick
    # mean_b_factor = predict_b_factor_mean(optimized_sequence,options.output_name)
    # print(f"The mean accuracy of the structure is estimated as {mean_b_factor:.2f}")


# Join the current working directory path and the output file name
    out_path = os.path.join(os.getcwd(), options.output_name + '.fasta')

# Open the output file for writing
    with open(out_path, 'w') as f:
        # Write the header
        f.write(">" + options.output_name + "\n")
        # Write the final sequence
        f.write(final_sequence + "\n")
    
if __name__ == "__main__":
    main()
