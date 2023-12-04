from typing import *
import torch
import time
import numpy as np
import json
import os
from unicore.utils import (
    tensor_tree_map,
)

from unifold.config import model_config
from unifold.modules.alphafold import AlphaFold
from unifold.data import protein, residue_constants
from unifold.colab.data import load_feature_for_one_target
from unifold.inference import automatic_chunk_size

from unifold.data.data_ops import get_pairwise_distances
from unifold.data import residue_constants as rc

def colab_inference(
    target_id: str,
    data_dir: str,
    param_dir: str,
    output_dir: str,
    is_multimer: bool,
    max_recycling_iters: int,
    num_ensembles: int,
    times: int,
    manual_seed: int,
    device: str = "cuda:0",
):

    model_name = "multimer_af2_crop"
    param_path = os.path.join(param_dir, "alphalink_weights.pt")

    config = model_config(model_name)
        
    config.data.common.max_recycling_iters = max_recycling_iters
    config.globals.max_recycling_iters = max_recycling_iters
    config.data.predict.num_ensembles = num_ensembles

    # faster prediction with large chunk
    config.globals.chunk_size = 128
    model = AlphaFold(config)
    print("start to load params {}".format(param_path))
    state_dict = torch.load(param_path)["ema"]["params"]
    state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.inference_mode()

    # data path is based on target_name
    cur_param_path_postfix = os.path.split(param_path)[-1]

    print("start to predict {}".format(target_id))
    plddts = {}
    ptms = {}
    iptms = {}
    model_confidences = {}

    best_result = None
    best_score = 0

    for seed in range(times):
        cur_seed = hash((manual_seed, seed)) % 100000
        batch = load_feature_for_one_target(
            config,
            data_dir,
            cur_seed,
            is_multimer=is_multimer,
            use_uniprot=is_multimer,
            symmetry_group=None,
        )
        seq_len = batch["aatype"].shape[-1]
        chunk_size, block_size = automatic_chunk_size(
                                    seq_len,
                                    device,
                                    is_bf16=False,
                                )
        model.globals.chunk_size = chunk_size
        model.globals.block_size = block_size

        print("using %d crosslink(s)" %(torch.sum((batch['xl'] > 0) / 2)))

        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device=device)
                for k, v in batch.items()
            }
            shapes = {k: v.shape for k, v in batch.items()}
            print(shapes)
            t = time.perf_counter()
            out = model(batch)
            print(f"Inference time: {time.perf_counter() - t}")

        def to_float(x):
            if x.dtype == torch.bfloat16 or x.dtype == torch.half:
                return x.float()
            else:
                return x

        # Toss out the recycling dimensions --- we don't need them anymore
        batch = tensor_tree_map(lambda t: t[-1, 0, ...], batch)
        batch = tensor_tree_map(to_float, batch)
        out = tensor_tree_map(lambda t: t[0, ...], out)
        out = tensor_tree_map(to_float, out)
        batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        plddt = out["plddt"]
        mean_plddt = np.mean(plddt)
        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )
        # TODO: , may need to reorder chains, based on entity_ids
        cur_protein = protein.from_prediction(
            features=batch, result=out, b_factors=plddt_b_factors
        )

        cur_save_name = (
            f"AlphaLink2_{cur_seed}"
        )
        plddts[cur_save_name] = str(mean_plddt)
        if is_multimer:
            model_confidences[cur_save_name] = str(np.mean(out["iptm+ptm"]))
            iptms[cur_save_name] = str(np.mean(out["iptm"]))
            ptms[cur_save_name] = str(np.mean(out["ptm"]))

        with open(os.path.join(output_dir, cur_save_name + '.pdb'), "w") as f:
            f.write(protein.to_pdb(cur_protein))

        if is_multimer:
            mean_ptm = np.mean(out["iptm+ptm"])
            if mean_ptm>best_score:
                best_result = {
                    "protein": cur_protein,
                    "plddt": out["plddt"],
                    "pae": out["predicted_aligned_error"],
                    "ptm": out["ptm"],
                    "iptm": out["iptm"],
                    "model_confidence": mean_ptm
                }
        else:
            if mean_plddt>best_score:
                best_result = {
                    "protein": cur_protein,
                    "plddt": out["plddt"],
                    "pae": None
                }

    # print("plddts", plddts)
    score_name = "AlphaLink2"
    plddt_fname = score_name + "_plddt.json"
    json.dump(plddts, open(os.path.join(output_dir, plddt_fname), "w"), indent=4)
    if ptms:
        print("ipTMs", iptms)
        ptm_fname = score_name + "_ptm.json"
        json.dump(ptms, open(os.path.join(output_dir, ptm_fname), "w"), indent=4)
        iptm_fname = score_name + "_iptm.json"
        json.dump(iptms, open(os.path.join(output_dir, iptm_fname), "w"), indent=4)

    model_confidences_fname = score_name + "_model_confidence.json"
    json.dump(model_confidences, open(os.path.join(output_dir, model_confidences_fname), "w"), indent=4)



    ca_idx = rc.atom_order["CA"]
    ca_coords = torch.from_numpy(out["final_atom_positions"][..., ca_idx, :])

    distances = get_pairwise_distances(ca_coords)

    xl = batch['xl'][...,0] > 0

    best_result['xl'] = [(i.item(),j.item(),distances[i,j].item()) for i,j in torch.nonzero(torch.from_numpy(xl)) if i < j ]
    
    return best_result
