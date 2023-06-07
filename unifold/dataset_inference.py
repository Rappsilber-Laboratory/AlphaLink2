import os
import json
import ml_collections as mlc
import numpy as np
import copy
import torch
from typing import *
from unifold.data import utils
from unifold.data.data_ops import NumpyDict, TorchDict
from unifold.data.process import process_features, process_labels
from unifold.data.process_multimer import (
    pair_and_merge,
    add_assembly_features,
    convert_monomer_features,
    post_process,
    merge_msas,
)

from unicore.data import UnicoreDataset, data_utils
from unicore.distributed import utils as distributed_utils

import random

Rotation = Iterable[Iterable]
Translation = Iterable
Operation = Union[str, Tuple[Rotation, Translation]]
NumpyExample = Tuple[NumpyDict, Optional[List[NumpyDict]]]
TorchExample = Tuple[TorchDict, Optional[List[TorchDict]]]


import logging
import gzip
import pickle
import math

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_data_config(
    config: mlc.ConfigDict,
    mode: str,
    num_res: int,
) -> Tuple[mlc.ConfigDict, List[str]]:
    cfg = copy.deepcopy(config)
    mode_cfg = cfg[mode]
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            mode_cfg.crop_size = num_res
    feature_names = cfg.common.unsupervised_features + cfg.common.recycling_features
    if cfg.common.use_templates:
        feature_names += cfg.common.template_features
    if cfg.common.is_multimer:
        feature_names += cfg.common.multimer_features
    if cfg[mode].supervised:
        feature_names += cfg.supervised.supervised_features

    return cfg, feature_names


def process_label(all_atom_positions: np.ndarray, operation: Operation) -> np.ndarray:
    if operation == "I":
        return all_atom_positions
    rot, trans = operation
    rot = np.array(rot).reshape(3, 3)
    trans = np.array(trans).reshape(3)
    return all_atom_positions @ rot.T + trans


@utils.lru_cache(maxsize=8, copy=True)
def load_single_feature(
    sequence_id: str,
    monomer_feature_dir: str,
    mode: str,
    uniprot_msa_dir: Optional[str] = None,
    is_monomer: bool = False,
) -> NumpyDict:

    monomer_feature = utils.load_pickle(
        os.path.join(monomer_feature_dir, f"{sequence_id}.feature.pkl.gz")
    )

    seq = np.array(list(monomer_feature['sequence'][0].decode("utf-8")))

    monomer_feature = convert_monomer_features(monomer_feature)
    chain_feature = {**monomer_feature}


    if uniprot_msa_dir is not None:
        if not os.path.isfile(os.path.join(uniprot_msa_dir, f"{sequence_id}.uniprot.pkl.gz")):
            return chain_feature
        all_seq_feature = utils.load_pickle(
            os.path.join(uniprot_msa_dir, f"{sequence_id}.uniprot.pkl.gz")
        )
        if is_monomer:
            chain_feature["msa"], chain_feature["deletion_matrix"] = merge_msas(
                chain_feature["msa"],
                chain_feature["deletion_matrix"],
                all_seq_feature["msa"],
                all_seq_feature["deletion_matrix"],
            )
        else:
            all_seq_feature["deletion_matrix"] = np.asarray(
                all_seq_feature.pop("deletion_matrix_int"), dtype=np.float32
            )
            all_seq_feature = utils.convert_all_seq_feature(all_seq_feature)
            for key in [
                "msa_all_seq",
                "msa_species_identifiers_all_seq",
                "deletion_matrix_all_seq",
            ]:
                chain_feature[key] = all_seq_feature[key]

    return chain_feature


def load_single_label(
    label_id: str,
    label_dir: str,
    symmetry_operation: Optional[Operation] = None,
) -> NumpyDict:
    label = utils.load_pickle(os.path.join(label_dir, f"{label_id}.label.pkl.gz"))
    if symmetry_operation is not None:
        label["all_atom_positions"] = process_label(
            label["all_atom_positions"], symmetry_operation
        )
    label = {
        k: v
        for k, v in label.items()
        if k in ["aatype", "all_atom_positions", "all_atom_mask", "resolution"]
    }
#    print(label['resolution'],label['resolution'].shape)
    label['resolution'][0] = 3.0

    return label


def prepare_crosslinks(tp, chain_ids, offsets, lengths):
    result = []
    seen = set()
    for i, chain1 in enumerate(chain_ids):
        for j, chain2 in enumerate(chain_ids):

            if chain1 not in tp:
                continue

            if chain2 not in tp[chain1]:
                continue

            if (chain1,chain2) in seen:
                continue

            seen.add((chain1,chain2))
            seen.add((chain2,chain1))

            links = []

            for ii, jj in tp[chain1][chain2]:
                ii += offsets[i]
                jj += offsets[j]
                if chain1 == chain2 and abs(ii - jj) < 6:
                    continue
                links.append((ii,jj))

            if len(links) == 0:
                continue

            links = torch.tensor(links)

            result.append(links)


    return [] if len(result) == 0 else torch.cat(result,dim=0)

def bucketize_xl(xl,size):
    buckets = torch.arange(0,1.05,0.05)

    n = size
    xl_ = np.zeros((n,n,1))

    for i, (r1,r2) in enumerate(xl):
        r1 = int(r1.item())
        r2 = int(r2.item())

        xl_[r1,r2,0] = xl_[r2,r1,0] = torch.bucketize(1-fdr, buckets)

    return xl_

def load(
    sequence_ids: List[str],
    monomer_feature_dir: str,
    crosslinks: str,
    mode: str,
    uniprot_msa_dir: Optional[str] = None,
    label_ids: Optional[List[str]] = None,
    label_dir: Optional[str] = None,
    symmetry_operations: Optional[List[Operation]] = None,
    is_monomer: bool = False,
) -> NumpyExample:

    all_chain_features = [
        load_single_feature(s, monomer_feature_dir, mode, uniprot_msa_dir, is_monomer)
        for s in sequence_ids
    ]

    all_chain_features = add_assembly_features(all_chain_features)

    asym_len = np.array([c["seq_length"] for c in all_chain_features], dtype=np.int64)

    offsets = np.cumsum([0] + list(asym_len[:-1]))

    assembly = np.unique([ s.split('_')[0] for s in sequence_ids])[0]

    tp_ = pickle.load(gzip.open(crosslinks,'rb'))

    tp = prepare_crosslinks(tp_, sequence_ids, offsets, asym_len)

    size = np.sum(asym_len)
   
    if len(tp) == 0:
        xl = np.zeros((size,size,1))
        print("no crosslinks",assembly,len(tp))
    else:
        xl = bucketize_xl(tp)

    if is_monomer:
        all_chain_features = all_chain_features[0]
    else:
        all_chain_features = pair_and_merge(all_chain_features)
        all_chain_features = post_process(all_chain_features)
    all_chain_features["asym_len"] = asym_len

    all_chain_features['xl'] = xl

    return all_chain_features, None


def process(
    config: mlc.ConfigDict,
    mode: str,
    features: NumpyDict,
    labels: Optional[List[NumpyDict]] = None,
    seed: int = 0,
    batch_idx: Optional[int] = None,
    data_idx: Optional[int] = None,
    is_distillation: bool = False,
) -> TorchExample:

    if mode == "train":
        assert batch_idx is not None
        with data_utils.numpy_seed(seed, batch_idx, key="recycling"):
            num_iters = np.random.randint(0, config.common.max_recycling_iters + 1)
            use_clamped_fape = np.random.rand() < config[mode].use_clamped_fape_prob
    else:
        num_iters = config.common.max_recycling_iters
        use_clamped_fape = 1

    features["num_recycling_iters"] = int(num_iters)
    features["use_clamped_fape"] = int(use_clamped_fape)
    features["is_distillation"] = int(is_distillation)
    if is_distillation and "msa_chains" in features:
        features.pop("msa_chains")

    num_res = int(features["seq_length"])
    cfg, feature_names = make_data_config(config, mode=mode, num_res=num_res)

    if labels is not None:
        features["resolution"] = labels[0]["resolution"].reshape(-1)

    with data_utils.numpy_seed(seed, data_idx, key="protein_feature"):
        features["crop_and_fix_size_seed"] = np.random.randint(0, 63355)
        features = utils.filter(features, desired_keys=feature_names)
        features = {k: torch.tensor(v) for k, v in features.items()}
        with torch.no_grad():
            features = process_features(features, cfg.common, cfg[mode])

    if labels is not None:
        labels = [{k: torch.tensor(v) for k, v in l.items()} for l in labels]
        with torch.no_grad():
            labels = process_labels(labels)

    return features, labels


def load_and_process(
    config: mlc.ConfigDict,
    mode: str,
    seed: int = 0,
    batch_idx: Optional[int] = None,
    data_idx: Optional[int] = None,
    is_distillation: bool = False,
    **load_kwargs,
):
    is_monomer = (
        is_distillation
        if "is_monomer" not in load_kwargs
        else load_kwargs.pop("is_monomer")
    )
    features, labels = load(**load_kwargs, mode=mode, is_monomer=is_monomer)
    features, labels = process(
        config, mode, features, labels, seed, batch_idx, data_idx, is_distillation
    )
    #print(features.keys())
    #print("MSA size", features["msa_feat"].shape, features["template_aatype"].shape, features["xl"].shape)
    return features, labels


class UnifoldDataset(UnicoreDataset):
    def __init__(
        self,
        args,
        seed,
        config,
        data_path,
        mode="train",
        max_step=None,
        disable_sd=False,
        json_prefix="",
    ):
        self.path = data_path

        def load_json(filename):
            return json.load(open(filename, "r"))

        sample_weight = load_json(
            os.path.join(self.path, json_prefix + mode + "_sample_weight.json")
        )
        self.multi_label = load_json(
            os.path.join(self.path, json_prefix + mode + "_multi_label.json")
        )
        self.inverse_multi_label = self._inverse_map(self.multi_label)
        self.sample_weight = {}
        for chain in self.inverse_multi_label:
            entity = self.inverse_multi_label[chain]
            self.sample_weight[chain] = sample_weight[entity]
        self.seq_sample_weight = sample_weight
        logger.info(
            "load {} chains (unique {} sequences)".format(
                len(self.sample_weight), len(self.seq_sample_weight)
            )
        )
        self.feature_path = os.path.join(self.path, "pdb_features")
        self.crosslink_path = os.path.join(self.path, "crosslinks")

        self.label_path = os.path.join(self.path, "pdb_labels")
        sd_sample_weight_path = os.path.join(
            self.path, json_prefix + "sd_train_sample_weight.json"
        )
        if mode == "train" and os.path.isfile(sd_sample_weight_path) and not disable_sd:
            self.sd_sample_weight = load_json(sd_sample_weight_path)
            logger.info(
                "load {} self-distillation samples.".format(len(self.sd_sample_weight))
            )
            self.sd_feature_path = os.path.join(self.path, "sd_features")
            self.sd_label_path = os.path.join(self.path, "sd_labels")
        else:
            self.sd_sample_weight = None
        self.batch_size = (
            args.batch_size
            * distributed_utils.get_data_parallel_world_size()
            * args.update_freq[0]
        )

        self.data_len = (
            max_step * self.batch_size
            if max_step is not None
            else len(self.sample_weight)
        )

        self.mode = mode
        self.num_seq, self.seq_keys, self.seq_sample_prob = self.cal_sample_weight(
            self.seq_sample_weight
        )
        self.num_chain, self.chain_keys, self.sample_prob = self.cal_sample_weight(
            self.sample_weight
        )

#        self.data_len = self.num_seq



        if self.sd_sample_weight is not None:
            (
                self.sd_num_chain,
                self.sd_chain_keys,
                self.sd_sample_prob,
            ) = self.cal_sample_weight(self.sd_sample_weight)
        self.config = config.data
        self.seed = seed
        self.sd_prob = args.sd_prob

    def cal_sample_weight(self, sample_weight):
        prot_keys = list(sample_weight.keys())
        sum_weight = sum(sample_weight.values())
        sample_prob = [sample_weight[k] / sum_weight for k in prot_keys]
        num_prot = len(prot_keys)
        return num_prot, prot_keys, sample_prob

    def sample_chain(self, idx, sample_by_seq=False):
        is_distillation = False
        if self.mode == "train":
            with data_utils.numpy_seed(self.seed, idx, key="data_sample"):
                is_distillation = (
                    (np.random.rand(1)[0] < self.sd_prob)
                    if self.sd_sample_weight is not None
                    else False
                )
                if is_distillation:
                    prot_idx = np.random.choice(
                        self.sd_num_chain, p=self.sd_sample_prob
                    )
                    label_name = self.sd_chain_keys[prot_idx]
                    seq_name = label_name
                else:
                    if not sample_by_seq:
                        prot_idx = np.random.choice(self.num_chain, p=self.sample_prob)
                        label_name = self.chain_keys[prot_idx]
                        seq_name = self.inverse_multi_label[label_name]
                    else:
                        seq_idx = np.random.choice(self.num_seq, p=self.seq_sample_prob)
                        seq_name = self.seq_keys[seq_idx]
                        label_name = np.random.choice(self.multi_label[seq_name])
        else:
            label_name = self.chain_keys[idx]
            seq_name = self.inverse_multi_label[label_name]
        return seq_name, label_name, is_distillation

    def __getitem__(self, idx):
        #print(self.mode, idx)
        if self.mode == "train":
            sequence_id = self.seq_keys[idx]
            label_id = np.random.choice(self.multi_label[sequence_id])
        else:
            label_id = self.chain_keys[idx]
            sequence_id = self.inverse_multi_label[label_id]

        is_distillation = False


        feature_dir, crosslink_dir, label_dir = (
            (self.feature_path, self.crosslink_path, self.label_path)
            if not is_distillation
            else (self.sd_feature_path, self.crosslink_path, self.sd_label_path)
        )
        features, _ = load_and_process(
            self.config,
            self.mode,
            self.seed,
            batch_idx=(idx // self.batch_size),
            data_idx=idx,
            is_distillation=is_distillation,
            sequence_ids=[sequence_id],
            monomer_feature_dir=feature_dir,
            crosslinks=crosslink_dir,
            uniprot_msa_dir=None,
            label_ids=[label_id],
            label_dir=label_dir,
            symmetry_operations=None,
            is_monomer=True,
        )
        return features

    def __len__(self):
        return self.data_len

    @staticmethod
    def collater(samples):
        # first dim is recyling. bsz is at the 2nd dim
        return data_utils.collate_dict(samples, dim=1)

    @staticmethod
    def _inverse_map(mapping: Dict[str, List[str]]):
        inverse_mapping = {}
        for ent, refs in mapping.items():
            for ref in refs:
                if ref in inverse_mapping:  # duplicated ent for this ref.
                    ent_2 = inverse_mapping[ref]
                    assert (
                        ent == ent_2
                    ), f"multiple entities ({ent_2}, {ent}) exist for reference {ref}."
                inverse_mapping[ref] = ent
        return inverse_mapping


class UnifoldMultimerDataset(UnifoldDataset):
    def __init__(
        self,
        args: mlc.ConfigDict,
        seed: int,
        config: mlc.ConfigDict,
        data_path: str,
        mode: str = "train",
        max_step: Optional[int] = None,
        disable_sd: bool = False,
        json_prefix: str = "",
        **kwargs,
    ):
        super().__init__(
            args, seed, config, data_path, mode, max_step, disable_sd, json_prefix
        )
        self.data_path = data_path
        self.pdb_assembly = json.load(
            open(os.path.join(self.data_path, json_prefix + "pdb_assembly.json"))
        )
        self.pdb_chains = self.get_chains(self.inverse_multi_label)

        self.monomer_feature_path = os.path.join(self.data_path, "pdb_features")
        self.uniprot_msa_path = os.path.join(self.data_path, "pdb_uniprots")
        self.label_path = os.path.join(self.data_path, "pdb_labels")
        self.crosslink_path_tp = os.path.join(self.path, "sulfo_sda_xl_tp")
        self.crosslink_path_fp = os.path.join(self.path, "sulfo_sda_xl_fp")
        self.max_chains = args.max_chains
        if self.mode == "train":
            self.pdb_chains, self.sample_weight = self.filter_pdb_by_max_chains(
                self.pdb_chains, self.pdb_assembly, self.sample_weight, self.max_chains
            )
            self.num_chain, self.chain_keys, self.sample_prob = self.cal_sample_weight(
                self.sample_weight
            )

    def __getitem__(self, idx):
        seq_id, label_id, is_distillation = self.sample_chain(idx)
        if is_distillation:
            label_ids = [label_id]
            sequence_ids = [seq_id]
            monomer_feature_path, uniprot_msa_path, label_path = (
                self.sd_feature_path,
                None,
                self.sd_label_path,
            )
            symmetry_operations = None
        else:
            pdb_id = self.get_pdb_name(label_id)
            if pdb_id in self.pdb_assembly and self.mode == "train":
                label_ids = [
                    pdb_id + "_" + id for id in self.pdb_assembly[pdb_id]["chains"]
                ]
                symmetry_operations = [t for t in self.pdb_assembly[pdb_id]["opers"]]
            else:
                label_ids = self.pdb_chains[pdb_id]
                symmetry_operations = None
            sequence_ids = [
                self.inverse_multi_label[chain_id] for chain_id in label_ids
            ]

            monomer_feature_path, uniprot_msa_path, label_path, crosslink_dir= (
                self.monomer_feature_path,
                self.uniprot_msa_path,
                self.label_path,
                self.crosslink_path,
            )

        return load_and_process(
            self.config,
            self.mode,
            self.seed,
            batch_idx=(idx // self.batch_size),
            data_idx=idx,
            is_distillation=is_distillation,
            sequence_ids=sequence_ids,
            monomer_feature_dir=monomer_feature_path,
            crosslinks=crosslink_dir,
            uniprot_msa_dir=uniprot_msa_path,
            label_ids=label_ids,
            label_dir=label_path,
            symmetry_operations=symmetry_operations,
            is_monomer=False,
        )

    @staticmethod
    def collater(samples):
        # first dim is recyling. bsz is at the 2nd dim
        if len(samples) <= 0:  # tackle empty batch
            return None
        feats = [s[0] for s in samples]
        labs = [s[1] for s in samples if s[1] is not None]
        try:
            feats = data_utils.collate_dict(feats, dim=1)
        except:
            raise ValueError("cannot collate features", feats)
        if not labs:
            labs = None
        return feats, labs

    @staticmethod
    def get_pdb_name(chain):
        return chain.split("_")[0]

    @staticmethod
    def get_chains(canon_chain_map):
        pdb_chains = {}
        for chain in canon_chain_map:
            pdb = UnifoldMultimerDataset.get_pdb_name(chain)
            if pdb not in pdb_chains:
                pdb_chains[pdb] = []
            pdb_chains[pdb].append(chain)
        return pdb_chains

    @staticmethod
    def filter_pdb_by_max_chains(pdb_chains, pdb_assembly, sample_weight, max_chains):
        new_pdb_chains = {}
        for chain in pdb_chains:
            if chain in pdb_assembly:
                size = len(pdb_assembly[chain]["chains"])
                if size <= max_chains:
                    new_pdb_chains[chain] = pdb_chains[chain]
            else:
                size = len(pdb_chains[chain])
                if size == 1:
                    new_pdb_chains[chain] = pdb_chains[chain]
        new_sample_weight = {
            k: sample_weight[k]
            for k in sample_weight
            if UnifoldMultimerDataset.get_pdb_name(k) in new_pdb_chains
        }
        logger.info(
            f"filtered out {len(pdb_chains) - len(new_pdb_chains)} / {len(pdb_chains)} PDBs "
            f"({len(sample_weight) - len(new_sample_weight)} / {len(sample_weight)} chains) "
            f"by max_chains {max_chains}"
        )
        return new_pdb_chains, new_sample_weight
