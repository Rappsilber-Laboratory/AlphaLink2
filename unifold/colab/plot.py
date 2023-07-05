# Construct multiclass b-factors to indicate confidence bands
# 0=very low, 1=low, 2=confident, 3=very high
# Color bands for visualizing plddt
import os
import numpy as np
import py3Dmol
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from IPython import display
from ipywidgets import GridspecLayout
from ipywidgets import Output
from typing import *
from unifold.data import protein

import Bio.PDB
from io import StringIO

def calc_residue_dist(residue_one, residue_two) :
    diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def colab_plot(
    best_result: Mapping[str, Any],
    output_dir: str,
    crosslinks,
    show_sidechains: bool = False,
    dpi: int = 100,
    cutoff: float = 25,
):
    best_protein = best_result["protein"]
    best_plddt = best_result["plddt"]
    best_pae = best_result.get("pae", None)
    
    to_visualize_pdb = protein.to_pdb(best_protein)

    structure = Bio.PDB.PDBParser().get_structure('X', StringIO(to_visualize_pdb))[0]

    # --- Visualise the prediction & confidence ---
    multichain_view = py3Dmol.view(width=800, height=600)
    multichain_view.addModelsAsFrames(to_visualize_pdb)
    multichain_style = {'cartoon': {'colorscheme': 'chain'}}
    multichain_view.setStyle({'model': -1}, multichain_style)
    for c1,v1 in crosslinks.items():
        for c2,v2 in v1.items():
            for i,j,_ in v2:
                i += 1
                j += 1
                if calc_residue_dist(structure[c1][i],structure[c2][j]) <= cutoff:
                    multichain_view.addCylinder({"start":{"resi":[i], "chain": c1},"end":{"resi":[j], "chain": c2},"color":"blue","radius":0.3});
                else:
                    multichain_view.addCylinder({"start":{"resi":[i], "chain": c1},"end":{"resi":[j], "chain": c2},"color":"red","radius":0.3});

    multichain_view.zoomTo()

    # Color the structure by per-residue pLDDT
    view = py3Dmol.view(width=800, height=600)
    view.addModelsAsFrames(to_visualize_pdb)
    style = {'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':0.5,'max':0.9}}}
    if show_sidechains:
        style['stick'] = {}
    view.setStyle({'model':-1}, style)
    view.zoomTo()

    grid = GridspecLayout(1, 3)
    out = Output()
    with out:
        multichain_view.show()
    grid[0, 0] = out

    out = Output()
    with out:
        view.show()
    grid[0, 1] = out


    out = Output()
    with out:
        plot_plddt_legend().show()
    grid[0, 2] = out

    display.display(grid)

PLDDT_BANDS = [(0., 0.50, '#FF7D45'),
            (0.50, 0.70, '#FFDB13'),
            (0.70, 0.90, '#65CBF3'),
            (0.90, 1.00, '#0053D6')]


def plot_plddt_legend():
    """Plots the legend for pLDDT."""
    thresh = ['Very low (pLDDT < 50)',
            'Low (70 > pLDDT > 50)',
            'Confident (90 > pLDDT > 70)',
            'Very high (pLDDT > 90)']

    colors = [x[2] for x in PLDDT_BANDS]

    plt.figure(figsize=(2, 2))
    for c in colors:
        plt.bar(0, 0, color=c)
    plt.legend(thresh, frameon=False, loc='center', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.title('Model Confidence', fontsize=20, pad=20)
    return plt


def colab_plot_confidence(
    best_result: Mapping[str, Any],
    output_dir: str,
    show_sidechains: bool = False,
    dpi: int = 100,
    cutoff: float = 25,
):
    best_protein = best_result["protein"]
    best_plddt = best_result["plddt"]
    best_pae = best_result.get("pae", None)

    satisfaction = 0
    mean_distance = 0
    if len(best_result['xl']) > 0:
        satisfaction = np.sum([d <= cutoff for _,_,d in best_result['xl']]) / len(best_result['xl'])
        mean_distance = np.mean([d for _,_,d in best_result['xl']])

    if "iptm" in best_result:
        print("Summary:\nModel confidence: %.3f\npTM: %.3f\nipTM: %.3f\nCrosslink satisfaction: %.3f\nMean distance of crosslinked residues: %.3f" %(best_result["model_confidence"],best_result["ptm"],best_result["iptm"],satisfaction,mean_distance))
    
    to_visualize_pdb = protein.to_pdb(best_protein)

    # Display pLDDT and predicted aligned error (if output by the model).
    num_plots = 1 if best_pae is None else 2

    plt.figure(figsize=[8 * num_plots , 6])
    plt.subplot(1, num_plots, 1)
    plt.plot(best_plddt * 100)
    plt.title('Predicted LDDT')
    plt.xlabel('Residue')
    plt.ylabel('pLDDT')
    plt.grid()
    plddt_svg_path = os.path.join(output_dir, 'plddt.svg')
    plt.savefig(plddt_svg_path, dpi=dpi, bbox_inches='tight')

    chain_ids = best_protein.chain_index
    for chain_boundary in np.nonzero(chain_ids[:-1] - chain_ids[1:]):
        if chain_boundary.size:
            plt.vlines(chain_boundary+1,0,100,color='red')

    if best_pae is not None:
        plt.subplot(1, 2, 2)
        max_pae = np.max(best_pae)
        #colors = ['#0F006F','#245AE6','#55CCFF','#FFFFFF']

        #cmap = LinearSegmentedColormap.from_list('mymap', colors)
        im = plt.imshow(best_pae, vmin=0., vmax=max_pae, cmap='gray')
        plt.colorbar(im, fraction=0.046, pad=0.04)

        # Display lines at chain boundaries.
        total_num_res = best_protein.residue_index.shape[-1]
        chain_ids = best_protein.chain_index
        for chain_boundary in np.nonzero(chain_ids[:-1] - chain_ids[1:]):
            if chain_boundary.size:
                plt.plot([0, total_num_res], [chain_boundary, chain_boundary], color='red')
                plt.plot([chain_boundary, chain_boundary], [0, total_num_res], color='red')

        for i,j,distance in best_result['xl']:
            if distance <= cutoff:
                plt.scatter(i,j,s=20,color='blue')
                plt.scatter(j,i,s=20,color='blue')
            else:
                plt.scatter(j,i,s=20,color='red')
               	plt.scatter(i,j,s=20,color='red')

        r = plt.Line2D((0,0), (0,0), linestyle='none', marker='o', markerfacecolor="red", markeredgecolor="black",alpha=1.00,markersize=8,label='Unsatisfied crosslink')
        b = plt.Line2D((0,0), (0,0), linestyle='none', marker='o', markerfacecolor="blue", markeredgecolor="black",alpha=1.00,markersize=8,label='Satisfied crosslink')

        plt.legend(handles=[b,r],ncol=2,loc='upper center', bbox_to_anchor=(0.5, -0.1))

        plt.ylim(total_num_res-1,0)
        plt.xlim(0,total_num_res-1)

        plt.title('Predicted Aligned Error')
        plt.xlabel('Scored residue')
        plt.ylabel('Aligned residue')
        pae_svg_path = os.path.join(output_dir, 'pae.svg')
        plt.savefig(pae_svg_path, dpi=dpi, bbox_inches='tight')


def plot_distance_distribution(
    best_result: Mapping[str, Any],
    output_dir: str,
    dpi: int = 100,
):
    ax = plt.figure().gca()
    dist = [d for _,_,d in best_result['xl']]
    plt.hist(dist,rwidth=0.7);
    plt.title('Crosslink distance distribution (CA-CA)')
    plt.xlim(0,np.max(dist)+10)
    # plt.xticks(np.arange(0, np.max(dist)+1, 2.0))
    plt.xlabel('CA-CA Distance [Å]')
    plt.ylabel('Count')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    dist_svg_path = os.path.join(output_dir, 'distance_distribution.svg')
    plt.savefig(dist_svg_path, dpi=dpi, bbox_inches='tight')
