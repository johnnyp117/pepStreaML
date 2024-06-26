pepStreaML Documentation
Author : John Patterson (JP)

### Related works used to build this package

https://github.com/facebookresearch/esm
https://github.com/aqlaboratory/openfold

paper: " Evolutionary-scale prediction of atomic-level protein structure with a language model" 
	https://www.science.org/doi/abs/10.1126/science.ade2574

paper: " Learning inverse folding from millions of predicted structures "
	https://www.biorxiv.org/content/10.1101/2022.04.10.487779v2

paper: " A high-level programming language for generative protein design "
	https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1
	https://github.com/facebookresearch/esm/tree/main/examples/protein-programming-language

### Introduction
pepStreaML is a computational tool designed to perform protein structure-based peptide design. 
Its main function is to optimize a peptide's closeness to a given protein target site and various properties of the complex and/or peptide. 
The input to the program can be either a UniProt ID or a PDBID.

The tool provides two modes of operation:

Target mode: This mode uses conserved residues inside the user's selected residues. The residues can be discontinuous. <br>
Whole mode: This mode ignores the mode of operation described above in favor of the subtarget specified by the user.

This pipeline utilzies the Evolutionary Scale Model (ESM) to both create seequnces, predict structures, and find conserved residues in situ. 

![til](./calmodulin_design_pepsteaml.gif)

### Usage

python pepstream.py -i [PDB_ID] -c [CHAIN] -b [BINDING_SITE] -n [SAMPLES] -t [TEMPERATURE_RANGE] -s [STEPS] -o [OUTPUT_NAME] -z [DESIGN_NUM]


### Command-Line Arguments: 
##### -i, --input: Specifies the input PDB structure file to be downloaded.
##### -u, --uniprot: Specifies the UniProt ID to search for an associates AlphaFold structure to be downloaded.
##### -c, --chain: Specifies the chain of the input receptor PDB structure.
##### -o, --output: Name of the output CSV file, directory, and naming for predictions.
##### -n, --num_samples: Number of sequences generated per temperature inside the ESM-IF model. Default is 10.
##### -t, --temperature: Specifies the temperature range in the format (start, end, step) for the ESM-IF model.
##### -e, --design_temp: The initial temperature for simulated annealing used in optimization. Default is 1.0.
##### -b, --binding-site: Binding site residue indices, can parse discontinous sites with a comma "," (ex. 10-20,40-50).
##### -a, --rate: Annealing rate for simulated annealing of peptide sequence during optimization. Default is 0.97.
##### -s, --steps: Total number of steps for optimization. Default is 10000.
##### -d, --display: Flag to indicate whether to display progress or not. Default is True.
##### -m, --mode: Mode of operation - "whole" or "target" for structure design optimization. Default is "target".
##### -z, --num_designs: Number of designs to generate. Default is 2.
##### --diversify: Enable or disable diversification of target structure. Default is False.


### Steps:

Input Parsing and Validation: Collects and validates user input. <br>
Data Preparation: If given a UniProt ID, it fetches the corresponding structure; otherwise, it uses the provided PDB file. It then derives atom and sequence information from the structure. <br>

ESM-IF Diversification: Performs the Inverse Folding method using ESM to diversify the given structure.  <br>

Sequence Optimization: Depending on the mode ("target" or "whole"), it optimizes the sequence to be close to the target site and also optimizes its energy stability. Optimization utilizes scores (called Energies) that are the response of the sequence mutation and subseqeut structure prediction. At this time these are not exposed to the user interface.  <br>

Output Generation: Outputs the final optimized sequence in both PDB and FASTA formats. <br>

### The energy scores currently used in optimization (inside ./language/energy.py):

MaximizePTM() <br>
	- Perplexity of sequence <br>
MaximizePLDDT() <br>
	- goodness of structure from ESMFold (OpenFold) <br>
MinimizeSurfaceHydrophobics() <br>
	- hydrophobic_score and surface_ratio <br>
MinimizeSurfaceExposure() <br>
	- surface_ratio <br>
MaximizeGlobularity() <br>
	- globularity from centroid <br>
MinimizeTargetDistance <br>
	- distance between target residues or considerved residues and peptide <br>

## Installation !!Work in Progress, untested!!

### Using pip

You can install `pepStreaML` directly using pip:

```bash
pip install pepStreaML

### Using conda

conda env create -f environment.yml
conda activate pepStreaML
```
## To Do:
- Add energies calculated from openmm
- Add active learning
- Add new verion of designer for subsection of interface
- Pass MSA into diversify target?
