# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from rich.live import Live
from rich.table import Table

from language.folding_callbacks import FoldingCallback
from language.program import ProgramNode

from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io import TrajectoryFile
import mdtraj as md

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
import time
import os

@dataclass
class MetropolisHastingsState:
    program: ProgramNode
    temperature: float
    annealing_rate: float
    num_steps: int
    energy: float
    best_energy: float
    energy_term_fn_values: list


def metropolis_hastings_step(
    state: MetropolisHastingsState,
    folding_callback: FoldingCallback,
    verbose: bool = False,
) -> MetropolisHastingsState:
    temperature = state.temperature * state.annealing_rate

    candidate: ProgramNode = deepcopy(state.program)
    candidate.mutate()

    sequence, residue_indices = candidate.get_sequence_and_set_residue_index_ranges()
    folding_output = folding_callback.fold(sequence, residue_indices)

    energy_term_fns = candidate.get_energy_term_functions()
    energy_term_fn_values = [
        (name, weight, energy_fn(folding_output)) for name, weight, energy_fn in energy_term_fns
    ]
    # TODO(scandido): Log these.
    energy: float = sum([
        weight * value for _, weight, value in energy_term_fn_values
    ])

    accept_candidate = False
    if state.energy is None:
        accept_candidate = True
    else:
        # NOTE(scandido): We are minimizing the function here so instead of
        # candidate - current we do -1 * (candidate - current) = -candidate + current.
        energy_differential: float = -energy + state.energy
        accept_probability: float = np.clip(
            # NOTE(scandido): We approximate the ratio of transition probabilities from
            # current to candidate vs. candidate to current to be equal, which is
            # approximately correct.
            np.exp(energy_differential / temperature),
            a_min=None,
            a_max=1.0,
        )
        accept_candidate: bool = np.random.uniform() < accept_probability

    if accept_candidate:
        sequence, _ = candidate.get_sequence_and_set_residue_index_ranges()
        if verbose:
            print(f"Accepted {sequence} with energy {energy:.2f}.")

    return MetropolisHastingsState(
        program=candidate if accept_candidate else state.program,
        temperature=temperature,
        annealing_rate=state.annealing_rate,
        num_steps=state.num_steps + 1,
        energy=energy if accept_candidate else state.energy,
        best_energy=min(energy, state.energy) if state.energy else energy,
        energy_term_fn_values=energy_term_fn_values,
    )


def run_simulated_annealing(
    program: ProgramNode,
    initial_temperature: float,
    annealing_rate: float,
    total_num_steps: int,
    folding_callback: FoldingCallback,
    display_progress: bool = True,
    progress_verbose_print: bool = False,
    output_dir: str = ""
) -> ProgramNode:
    # TODO(scandido): Track accept rate.
    # DataFrame to store all accepted steps
    # accepted_steps = pd.DataFrame(columns=["Step", "Energy", "BestEnergy"]+[name for name, _, _ in program.get_energy_term_functions()])
    accepted_steps = pd.DataFrame()
    accepted_frames = []
    state = MetropolisHastingsState(
        program=program,
        temperature=initial_temperature,
        annealing_rate=annealing_rate,
        num_steps=0,
        energy=None,
        best_energy=None,
        energy_term_fn_values=None,
    )

    def _generate_table(state):
        table = Table()
        table.add_column("Energy name")
        table.add_column("Weight")
        table.add_column("Value")
        if state.energy_term_fn_values is None:
            return table
        for name, weight, value in state.energy_term_fn_values:
            table.add_row(name, f"{weight:.2f}", f"{value:.2f}")
        table.add_row("Energy", "", f"{state.energy:.2f}")
        table.add_row(
            "Iterations", "",
            f"{state.num_steps} / {total_num_steps}"
        )
        return table

    with Live() as live:
        for _ in range(1, total_num_steps + 1):
            state = metropolis_hastings_step(
                state,
                folding_callback,
                verbose=progress_verbose_print,
            )
            if display_progress:
                live.update(_generate_table(state))
            # If candidate is accepted, save to DataFrame
            if state.energy is not None and (len(accepted_steps) == 0 or accepted_steps.iloc[-1].Energy != state.energy):
                step_info = {
                    "Step": state.num_steps,
                    "Temperature": state.temperature,
                    "Energy": state.energy,
                    "Best Energy": state.best_energy,
                }
        
                # Add energy terms info to step_info dict
                for name, weight, value in state.energy_term_fn_values:
                    step_info[name] = value
        
                # Convert step_info to DataFrame
                step_df = pd.DataFrame(step_info, index=[0])
                # Concatenate step_df to accepted_steps_df
                accepted_steps = pd.concat([accepted_steps, step_df], ignore_index=True)
                sequence, residue_indices = state.program.get_sequence_and_set_residue_index_ranges()
                accepted_frames.append(folding_callback.fold(sequence, residue_indices))
#                accepted_frames = pd.concat([accepted_frames, frame], ignore_index=True)
    # Save plot with a unique name
    timestamp = int(time.time())
    dir_path = output_dir
    # Save accepted steps to CSV file
    accepted_steps.to_csv(os.path.join(dir_path, f"accepted_steps_{timestamp}.csv"), index=False)
    # Save traj of designs (?? maybe)
    traj_dir = os.path.join(dir_path, f"steps_traj_{timestamp}")
    os.makedirs(traj_dir, exist_ok=True)
    for i,frame in enumerate(accepted_frames):
        mask = np.zeros(len(frame.atoms), dtype=bool)

        for k, atom in enumerate(frame.atoms):
            if atom.res_id > 1000:
                mask[k] = True
        frame.atoms.chain_id[mask] = 'B'
        output_path = os.path.join(traj_dir, f"frame_{i}.pdb")
    # Save this frame
        file = PDBFile()
        file.set_structure(frame.atoms)
        file.write(output_path)
    # Plot using seaborn
    pairplot = sns.pairplot(accepted_steps, diag_kind="kde")
    plt.savefig(os.path.join(dir_path,f"plot_{timestamp}.png"))
    # Plotting energy functions
    energy_plot = accepted_steps.plot(x='Step', y=accepted_steps.columns[1:], kind='line', title='Energy Functions vs Step')
    plt.savefig(os.path.join(dir_path,f"energy_plot_{timestamp}.png"))


    return state.program
