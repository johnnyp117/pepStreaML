# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 12:22:31 2023

@author: jp
"""

import torch
import esm
import esm.inverse_folding
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import logomaker
from optparse import OptionParser
import seaborn as sns
import matplotlib.pyplot as plt

class ESMInverseFolding:
    def __init__(self):
    # Check for GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load ESM-1b model and alphabet
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        self.model = model
        if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")
        self.alphabet = alphabet
        self.device = device
        
    def inv_folding(self, input_structure, chain, temperature_range, num_samples, out_name):
        results = []
        print(input_structure, chain)
        structure = esm.inverse_folding.util.load_structure(input_structure, chain)
        coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        print(f'Saving sampled sequences to {out_name}.')


        # Define the number of initial predictions to make
        initial_predictions = 30

        # Initialize a list to store recovery values
        recovery_values = []

        # Make the initial predictions and store recovery values
        for _ in range(initial_predictions):
            generated_sequence = self.model.sample(coords, temperature=np.mean(temperature_range), device=torch.device('cuda'))
            recovery = np.mean([(a==b) for a, b in zip(native_seq, generated_sequence)])
            recovery_values.append(recovery)

        # Calculate the threshold value for recovery as the mean of the top 5 recovery values
        recovery_threshold = np.mean(sorted(recovery_values, reverse=True)[:5])
        print(f'Recovery threshold calculated as: {recovery_threshold:.2f}')

        Path(out_name+".fasta").parent.mkdir(parents=True, exist_ok=True)
        i=0
        with open(out_name+".fasta", 'w') as f:
            for temperature in tqdm(temperature_range, desc="Temps "):
                for _ in tqdm(range(int(num_samples)) , desc="Samples "):
                    recovery = 0
                    ll = -1
                    while recovery < recovery_threshold: #0.5:  np.exp(-ll) > 1.5:
                        generated_sequence = self.model.sample(coords, temperature=temperature, device=torch.device('cuda'))
                        recovery = np.mean([(a==b) for a, b in zip(native_seq, generated_sequence)])
                        ll, _ = esm.inverse_folding.util.score_sequence(
                                self.model, self.alphabet, coords, generated_sequence) 

                        # print(f'Native sequence: {recovery:.2f}')
                    # print(f'Log likelihood: {ll:.2f}')
                    # print(f'Perplexity: {np.exp(-ll):.2f}')     
                    f.write(f'>sampled_seq_{i+1}\n')
                    f.write(generated_sequence + '\n')
                    results.append({
                        "temperature": temperature,
                        "sequence": generated_sequence,
                        "recovery": recovery,
                        "perplexity": np.exp(-ll)       
                    })
                    i+=1

            
        return pd.DataFrame(results)
    
    def logo(self,results, out_name):
        mat_df = logomaker.alignment_to_matrix(results['sequence'].tolist())
        # print(mat_df)
        ww_logo = logomaker.Logo(mat_df,
                                 font_name='Stencil Std',
                                 color_scheme='NajafabadiEtAl2017',
                                 vpad=.1,
                                 width=.8)
        
        # loop to check each position for conservation criterion
        highlighted_positions = []
        for pos in range(len(mat_df)):
            if max(mat_df.iloc[pos]) / sum(mat_df.iloc[pos]) > 0.8:
                ww_logo.highlight_position(p=pos, color='gold', alpha=.5)
                highlighted_positions.append(pos)
                
        ww_logo.fig.savefig( out_name+"_logo.png", dpi=300)
        plt.close('all')
        # plt.show()
        return highlighted_positions
        
    def plot(self,results, out_name):
        ax=sns.scatterplot(data=results,x="temperature",y="recovery",hue="perplexity")
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
        fig = ax.get_figure()
        fig.savefig( out_name+"_plot.png", bbox_inches='tight')
        # turn this off for linux
        # plt.show()
        plt.close('all')

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input_structure", help="Input PDB structure file")
    parser.add_option("-c", "--chain", dest="chain", help="Input PDB structure chain")
    parser.add_option("-o", "--output", dest="output_csv", help="Output CSV file with generated sequences")
    parser.add_option("-n", "--num_samples", dest="num_samples", help="Number of generated sequences per temp", default=10)
    parser.add_option("-t", "--temperature", dest="temperature_range", help="Temperature range (start, end, step)")
    (options, args) = parser.parse_args()
    print(options)
    if not options.input_structure or not options.output_csv or not options.temperature_range:
        parser.error("Incorrect options provided. Use -h or --help for more information.")

    
    # Create an instance of ESMInverseFolding
    esm_inverse_folding = ESMInverseFolding()

    # Parse temperature range
    temperature_range = [float(t) for t in options.temperature_range.split(',')]
    temperature_range_ary = np.arange(temperature_range[0], temperature_range[1], temperature_range[2])

    # Run inverse folding for the input structure over a range of temperatures
    results_df = esm_inverse_folding.inv_folding(options.input_structure, options.chain, temperature_range_ary, options.num_samples, options.output_csv)

    # Save the results to a CSV file
    results_df.to_csv(options.output_csv, index=False)
    
    conserved_positions = esm_inverse_folding.logo(results_df, options.output_csv)
    esm_inverse_folding.plot(results_df, options.output_csv)

if __name__ == "__main__":
    main()