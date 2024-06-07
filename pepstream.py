#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon July  15 15:28:54 2023

@author: jp
"""
# Packages
import os, sys, re
import os.path
import time
import json
import pandas as pd
import numpy as np
from optparse import OptionParser
from functools import reduce
import operator

from biotite.database.rcsb import fetch
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

# dependancies 
import diversify_on_structure
import design_cmorf
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
# pepstream method implemented via ESM from Meta
# 
# This can run from either a uniprot ID or a PDBID
#
# The exact design routine is optimizing closness to the target site along with an energy calculation for stability
# As of today the footprint or target mode will used conserved residues inside your selection, which can be discontinous
#     the complex or whole mode actually ignores this in favour of the subtarget you do (or dont) specify. 
#TODO: Check uniprot import works, fix all these confusing variable names (esm_range)
####
def diversify_target_structure(conserved_positions, start, end, folding_callback, results_df, row_idx):
    """
    Diversifies target structure based on a specific sequence from results_df.
    
    - conserved_positions (list): List of conserved positions.
    - start (int): Start residue index.
    - end (int): End residue index.
    - folding_callback (obj): EsmFoldv1 callback object.
    - results_df (DataFrame): DataFrame containing alternative sequences.
    - row_idx (int): Row index for which sequence is to be considered after sorting by recovery.

    Returns:
    - ProgramNode: A ProgramNode representing diversified target structure.
    """
    
    # Sort results_df by 'recovery'
    sorted_df = results_df.sort_values(by='recovery', ascending=False)
    
    if row_idx < len(sorted_df):
        alternative_sequence = sorted_df.iloc[row_idx]['sequence']
        folding_output = folding_callback.fold(alternative_sequence, np.arange(183,183+len(alternative_sequence),1))
        binding_site_atoms = get_atomarray_in_residue_range(folding_output.atoms, start=start, end=end)
        program = design_cmorf.asymmetric_binding_complex(folding_output.atoms, binding_site_atoms, min_length=10, max_length=30)
        return program
    # picks a random row.
    else:
        row_idx = np.random.randint(0, len(sorted_df))
        alternative_sequence = sorted_df.iloc[row_idx]['sequence']
        folding_output = folding_callback.fold(alternative_sequence, np.arange(0,len(alternative_sequence),1))
        binding_site_atoms = get_atomarray_in_residue_range(folding_output.atoms, start=start, end=end)
        program = design_cmorf.asymmetric_binding_complex(folding_output.atoms, binding_site_atoms, min_length=10, max_length=30)
        return program

def main():
    start_time = time.time()  # record start time
    parser = OptionParser(usage="usage: python pepstream.py -i 1BTL -c A -b 1-48 -n 10 -t 0.1,0.5,0.1 -s 300 -o binder_H1_7 -z 5",
                          version="%prog v0.5")
    parser.add_option("-i", "--input", dest="input_structure", help="Input PDB structure file")
    parser.add_option("-u", "--uniprot", dest="uniprot", help="Uniprot ID")
    parser.add_option("-c", "--chain", dest="chain", help="Input PDB structure chain")
    parser.add_option("-o", "--output", dest="output_name", help="Output CSV file with generated sequences")
    parser.add_option("-n", "--num_samples", dest="num_samples", help="Number of generated sequences per temp", default=10)
    parser.add_option("-t", "--temperature", dest="temperature_range", help="Temperature range (start, end, step)")
    parser.add_option("-e", "--design_temp", dest="initial_temperature", default=1.0, type=float,
                      help="initial temperature for simulated annealing")
    parser.add_option("-b", "--binding-site", dest="binding_site", type="string", help="binding site indices")
    parser.add_option("-a", "--rate", dest="annealing_rate", default=0.97, type=float, help="annealing rate for simulated annealing")
    parser.add_option("-s", "--steps", dest="total_num_steps", default=10000, type=int, help="total number of steps for simulated annealing")
    parser.add_option("-d", "--display", dest="display_progress", action="store_true", default=True, help="whether to display progress or not")
    parser.add_option("-m", "--mode", dest="mode", type="string", default="target", help="whole or target of strucutre for design optimization")
    parser.add_option("-z", "--num_designs", dest="num_designs", default=2, type=int, 
                  help="Number of designs to generate")
    parser.add_option("--diversify", action="store_true", dest="diversify", default=False, help="Enable diversification of target structure")
    # save design energies to plot
    # save design energies to file
    (options, args) = parser.parse_args()
    print(options)
    diversify = options.diversify
    # if options.binding_site:
    #     start, end = design_cmorf.parse_binding_site(options.binding_site)
    # else:
    #     print("Please provide binding site indices.")
    #     return
    if options.binding_site:
        binding_site_ranges = design_cmorf.parse_binding_site(options.binding_site)
    else:
        print("Please provide binding site indices.")
        return
    
    if options.uniprot:
        pdb_file_path = options.uniprot + ".pdb"
        if not os.path.exists(pdb_file_path):
            sequence = design_cmorf.download_alphafold_prediction(options.uniprot)
            pdb_file_path = fetch(options.uniprot, format="pdb", target_path=os.getcwd())
        structure_atoms: AtomArray = pdb_file_to_atomarray(pdb_file_path)
    elif options.input_structure:
        # PDBFile.read(pdb_path)
        pdb_file_path = options.input_structure + ".pdb"
        if not os.path.exists(pdb_file_path):
            pdb_file_path = fetch(options.input_structure, format="pdb", target_path=os.getcwd())
        # pdb_file_path = fetch(options.input_structure, format="pdb", target_path=os.getcwd() )
        structure_atoms: AtomArray = pdb_file_to_atomarray(pdb_file_path, options.chain )
    else:
        print("Please provide a Uniprot ID or a pdb file path.")
        return

    structure_atoms = get_atomarray_in_residue_range(structure_atoms, start=structure_atoms.res_id.min(), end=structure_atoms.res_id.max())
    structure_sequence = sequence_from_atomarray(structure_atoms)
    # binding_site_atoms = get_atomarray_in_residue_range(structure_atoms, start=start, end=end)
    binding_site_atoms = []
    for start, end in binding_site_ranges:
        binding_site_atoms.append(get_atomarray_in_residue_range(structure_atoms, start=start, end=end))
    # Reduce the list of AtomArray slices to a single AtomArray
    # binding_site_atoms = reduce(operator.add, binding_site_atoms)

    output_directory = os.path.join(os.getcwd(), options.output_name)
    os.makedirs(output_directory, exist_ok=True)
    # Save the options to a file
    with open(os.path.join(output_directory, 'options.json'), 'w') as f:
        json.dump(options.__dict__, f, indent=4)
#%% Section 1: Run time ESM-IF diversify structure
    esm_inverse_folding = diversify_on_structure.ESMInverseFolding()
    csv_path = os.path.join(output_directory, options.output_name+"_IF.csv")
# Check if the file already exists
    if os.path.exists(csv_path):
# If the file exists, load the data into a DataFrame
        results_df = pd.read_csv(csv_path)
        print("We detected a previous run and found existing sequences.\n \
              To save your time and resources, we'll use these sequences for the current analysis.\n \
              If you want to generate new sequences, please delete or rename the existing file and rerun the program.")
    else:
        temperature_range = [float(t) for t in options.temperature_range.split(',')]
        temperature_range_ary = np.arange(temperature_range[0], temperature_range[1], temperature_range[2])
        print(os.path.basename(pdb_file_path))
        results_df = esm_inverse_folding.inv_folding(os.path.basename(pdb_file_path), options.chain, temperature_range_ary, options.num_samples, os.path.join(output_directory, options.output_name))

# Save the results to a CSV file
        results_df.to_csv(os.path.join(output_directory, options.output_name+"_IF.csv"), index=False)

    # conserved_positions = esm_inverse_folding.logo(results_df, options.output_name)
    conserved_positions = esm_inverse_folding.logo(results_df, os.path.join(output_directory, options.output_name))
    # esm_inverse_folding.plot(results_df, options.output_name)
    esm_inverse_folding.plot(results_df, os.path.join(output_directory, options.output_name))
# Filter out positions outside the range defined by start and end
    # conserved_positions = [pos for pos in conserved_positions if start <= pos <= end]
    conserved_positions_all = []
    offset = structure_atoms.res_id.min()  # calculate the offset
    for start, end in binding_site_ranges:
    # Filter out positions outside the range defined by start and end !original index of PDB!
        conserved_positions = [pos + offset for pos in conserved_positions if start <= pos + offset  <= end]
        conserved_positions_all.extend(conserved_positions)
    conserved_atoms = get_atomarray_in_residue_list(structure_atoms, conserved_positions_all) # conserved_positions

    
#%% Section 2: Run time design using ESMFold   

    # structure_sequence = sequence_from_atomarray(structure_atoms)
    target_sequence = [sequence_from_atomarray(atoms) for atoms in binding_site_atoms] # reduce?
    num_designs = options.num_designs
    folding_callback = EsmFoldv1()
    folding_callback.load(device="cuda:0") # dis makes it run on cuda, can add a check for cpu or cude?
    for i in range(num_designs):
## Position peptide, pre-step to annealing that looks for an inital sequence with some cut off to the target
## doesnt seem to work amazingly on the MHCI problem. pointless?
        num_iterations = 20  # Adjust to fit your needs
        best_dist = 100
        for j in range(num_iterations):
# at this point we have the sequences generated by inverse folding stored in results_df and conserved_positions
            if diversify:
                output_receptor_len = len(structure_sequence)
                print("Diversification enabled >:D !")
                program = diversify_target_structure(conserved_positions, start, end, folding_callback, results_df, j)
            # if options.mode == "target":
            else:
               # output_receptor_len = sum([len(seq) for seq in target_sequence]) 
                output_receptor_len = len(target_sequence)
                program = design_cmorf.asymmetric_binding_footprint(binding_site_atoms, conserved_atoms, min_length=10, max_length=30)
#             else:
# # add in conserved_atoms input to this method?
#                 program = design_cmorf.asymmetric_binding_complex(structure_atoms, binding_site_atoms, min_length=10, max_length=30)
            # if options.mode == "target":
            
            print("inital sequence = {}".format(program.get_sequence_and_set_residue_index_ranges()[0][output_receptor_len:]))
            # else:
            #     output_receptor_len = len(structure_sequence) 
            #     print("inital sequence = {}".format(program.get_sequence_and_set_residue_index_ranges()[0][len(structure_sequence):]))
            sequence, residue_indices = program.get_sequence_and_set_residue_index_ranges()
            folding_output = folding_callback.fold(sequence, residue_indices)

            energy_term_fns = program.get_energy_term_functions()
            energy_term_fn_values = [
                (name, weight, energy_fn(folding_output)) for name, weight, energy_fn in energy_term_fns
            ]
            for name, weight, value in energy_term_fn_values:
                if "MinimizeTargetDistance" in name:
                    print(f"MinimizeTargetDistance is {value}") #:.2f
                    if value < best_dist:
# collect best program state here based off MinimizeTargetDistance 
                        best_start = program
                        best_fold = folding_output
                        best_dist = value
### turn into a def?
    # at this point we have the sequences generated by inverse folding stored in results_df and conserved_positions
        if options.mode == "target":
            program = design_cmorf.asymmetric_binding_footprint(binding_site_atoms, conserved_atoms, min_length=10, max_length=30)
        else:
    # add in conserved_atoms input to this method?
            program = design_cmorf.asymmetric_binding_complex(structure_atoms, reduce(operator.add, binding_site_atoms), min_length=10, max_length=30)
      
        mask = np.zeros(len(best_fold.atoms), dtype=bool)

        for k, atom in enumerate(best_fold.atoms):
            if atom.res_id > 1000:
                mask[k] = True
        best_fold.atoms.chain_id[mask] = 'B' 
        
        file = PDBFile()
        # best_fold.atoms[output_receptor_len:].set_annotation("chain_id", ['B']*output_receptor_len-best_fold.atoms.)  
        file.set_structure(best_fold.atoms)  
        file.write(os.path.join(os.getcwd(), os.path.join(output_directory,options.output_name + f"_inital_{i}.pdb")))
        print(f"Best start is {best_dist:.2f} A away and this seq: {best_start.get_sequence_and_set_residue_index_ranges()[0][output_receptor_len:]}")
        best_start_children = best_start.get_children() #Scott's Tots lolz
        program.children[-1] = best_start_children[-1] # array of program nodes, swap variable node (peptide being designed), hopefully
        # program = best_start
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
            progress_verbose_print=False,
            output_dir=output_directory
        )
    
        optimized_sequence , residue_indices = optimized_program.get_sequence_and_set_residue_index_ranges()
        folding_output = folding_callback.fold(optimized_sequence, residue_indices)
        
 #%% Section 3: outputs   
        # print(folding_output)
        if options.mode == "target":
            # output_receptor_len = len(target_sequence)
            print("Final sequence = {}".format(optimized_program.get_sequence_and_set_residue_index_ranges()[0][output_receptor_len:]))
# Create the final sequence
            # final_sequence = structure_sequence + ":" + optimized_sequence[len(output_receptor_len):]
        else:
            # output_receptor_len = len(structure_sequence) 
            print("Final sequence = {}".format(optimized_program.get_sequence_and_set_residue_index_ranges()[0][output_receptor_len:]))
# Create the final sequence
        final_sequence = structure_sequence + ":" + optimized_sequence[output_receptor_len:]
                
        print("Target sequence = {}".format(target_sequence))
        print("Final pLDDT = {}".format(folding_output.plddt))
        print("Final PTM = {}".format(folding_output.ptm))
        
        file = PDBFile()

        print(residue_indices)
        mask = np.zeros(len(folding_output.atoms), dtype=bool)

        for k, atom in enumerate(folding_output.atoms):
            if atom.res_id > 1000:
                mask[k] = True
        folding_output.atoms.chain_id[mask] = 'B'   
        #.set_annotation("chain_id", 'B')[output_receptor_len:]
        file.set_structure(folding_output.atoms)
        # file.write(os.path.join(os.getcwd(), options.output_name+"_design.pdb"))
        file.write(os.path.join(os.getcwd(), os.path.join(output_directory,options.output_name + f"_design_{i}.pdb")))
        
# Join the current working directory path and the output file name
        # out_path = os.path.join(os.getcwd(), options.output_name + '_design.fasta')
        out_path = os.path.join(os.getcwd(), os.path.join(output_directory,options.output_name + f"_design_{i}.fasta"))

        with open(out_path, 'w') as f:
# Write the header
            f.write(">" + f"{options.output_name}_design_{i}" + "\n")
# Write the final sequence
            f.write(final_sequence + "\n")
    end_time = time.time()  # record end time
    total_time = end_time - start_time  # calculate total run time
    print(f"Total run time: {total_time} seconds")
if __name__ == "__main__":
    main()
