import itertools
import os
import pickle as pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fast_jtnn.jtmpn import JTMPN
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.mpn import MPN


class PairTreeFolder(object):
    def __init__(
        self,
        data_folder,
        vocab,
        batch_size,
        num_workers=4,
        shuffle=True,
        y_assm=True,
        replicate=None,
    ):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, "rb") as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # shuffle data before batch

            batches = [
                data[i : i + self.batch_size]
                for i in range(0, len(data), self.batch_size)
            ]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(
                dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
            )  # , num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class MolTreeFolder_prop(object):
    def __init__(
        self,
        data_folder,
        vocab,
        batch_size,
        num_workers=4,
        shuffle=False,
        assm=True,
        replicate=None,
        optimize=False,
    ):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.optimize = optimize

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, "rb") as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # shuffle data before batch

            data, prop_data = data
            batches = [
                data[i : i + self.batch_size]
                for i in range(0, len(data), self.batch_size)
            ]
            batches_prop = [
                prop_data[i : i + self.batch_size]
                for i in range(0, len(prop_data), self.batch_size)
            ]

            dataset = MolTreeDataset_prop(
                batches, batches_prop, self.vocab, self.assm, self.optimize
            )
            dataloader = DataLoader(
                dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
            )  # , num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class PairTreeDataset(Dataset):
    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch0, batch1 = list(zip(*self.data[idx]))
        return tensorize_prop(batch0, self.vocab, assm=False), tensorize_prop(
            batch1, self.vocab, assm=self.y_assm
        )


class MolTreeDataset_prop(Dataset):
    def __init__(self, data, prop_data, vocab, assm=True, optimize=False):
        self.data = data
        self.prop_data = prop_data
        self.vocab = vocab
        self.assm = assm
        self.optimize = optimize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize_prop(
            self.data[idx],
            self.prop_data[idx],
            self.vocab,
            assm=self.assm,
            optimize=self.optimize,
        )


def tensorize_prop(tree_batch, prop_batch, vocab, assm=True, optimize=False):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    # TL debug, with https://stackoverflow.com/a/70323486 {
    # print(prop_batch)
    # print(type(prop_batch))
    # prop_batch = np.vstack(prop_batch).astype(np.float)
    # print(prop_batch)
    # print(type(prop_batch))
    # torch.from_numpy(a)

    prop_batch_tensor = torch.FloatTensor(prop_batch)

    if assm is False:
        return tree_batch, prop_batch_tensor, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i, mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            # Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1:
                continue
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
            batch_idx.extend([i] * len(node.cands))

    # WARNING: THIS WAS ADDED WHEN DOING LOCAL OPTIMIZATION.
    # the holder throws an error when doing local optimization.
    if optimize:
        jtmpn_holder = None
    else:
        jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return (
        tree_batch,
        prop_batch_tensor,
        jtenc_holder,
        mpn_holder,
        (jtmpn_holder, batch_idx),
    )


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
