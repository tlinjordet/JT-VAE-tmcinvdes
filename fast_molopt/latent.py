import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import torch
import torch.nn as nn
from rdkit.Chem import Descriptors
from torch.autograd import Variable

from fast_jtnn import *

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))
from fast_molopt.preprocess_prop import main_preprocess

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def parse_args(arg_list: list = None) -> argparse.Namespace:
    """Parse arguments from command line.

    Args:
        arg_list (list, optional): Automatically obtained from the command line if provided.
        If no arguments are given but default arguments are defined, the latter are used.

    Returns:
        argparse.Namespace: Dictionary-like class that contains the driver arguments.
    """
    parser = argparse.ArgumentParser(
        description="Strip substitute atoms from generated ligands"
    )

    # parser.add_argument("--training_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--output_path", default="optimize-processed")
    parser.add_argument("--prop_path", default=True)
    parser.add_argument("--nsplits", type=int, default=2)

    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)
    parser.add_argument("--cutoff", type=float, default=0.2)

    parser.add_argument(
        "--type", type=str, default="first", help="Which property to optimize on"
    )

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--clip_norm", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--step_beta", type=float, default=0.002)
    parser.add_argument("--max_beta", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=500)

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--anneal_rate", type=float, default=0.9)
    parser.add_argument("--anneal_iter", type=int, default=1000)
    parser.add_argument("--kl_anneal_iter", type=int, default=3000)
    parser.add_argument("--print_iter", type=int, default=50)
    parser.add_argument("--save_iter", type=int, default=1000)

    return parser.parse_args(arg_list)


def main():
    opts = parse_args()

    vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
    vocab = Vocab(vocab)

    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    float(opts.cutoff)

    model = JTpropVAE(
        vocab, int(hidden_size), int(latent_size), int(opts.depthT), int(opts.depthG)
    ).cuda()
    print(model)
    model.load_state_dict(torch.load(opts.model_path))
    model = model.cuda()

    output_dir = Path(f"latent_{time.strftime('%Y%m%d-%H%M%S')}")
    output_dir.mkdir(exist_ok=True)

    # Preprocess data

    # main_preprocess(train_path, prop_path, opts.output_path, opts.nsplits)
    #
    # loader = MolTreeFolder_prop(
    #     opts.output_path, vocab, batch_size=1, shuffle=False, num_workers=6
    # )

    # Load smiles from file
    with open("../data/labeled_set/train_full.txt") as f:
        smiles = [x.strip("\r\n ") for x in f.readlines()]
    with open("../data/labeled_set/train_prop_full.txt") as f:
        prop_data = [line.strip("\r\n ").split(",") for line in f]
    # Convert to float
    props = [[float(y) for y in x] for x in prop_data]

    # with open(output_dir / "opts.json", "w") as file:
    #     json.dump(vars(opts), file)

    # smiles=smiles[0:100]
    # props = props[0:100]
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    smi_chunked = chunks(smiles, 50)
    prop_chunked = chunks(props, 50)
    arr = np.zeros(shape=(len(smiles), 56))
    np.zeros(shape=(len(smiles), 2))
    for i, (s, prop) in enumerate(zip(smi_chunked, prop_chunked)):
        if i % 10 == 0:
            print(i)
        start_idx = i * 50
        end_idx = start_idx + 50
        torch.cuda.empty_cache()
        z1, z2 = model.encode_latent_from_smiles(s, prop)

        array = z1.cpu().detach().numpy()
        arr[start_idx:end_idx, :] = array

        # Get property prediction
        # props= model.propNN(z1).squeeze().cpu().detach().numpy()
        # property_predictions[start_idx:end_idx,:] = props

    np.save(f"{opts.model_path.stem.name}_latent.npy", arr)  # save
    # np.save(f'{opts.model_path.name}_latent_property_predictions.npy', property_predictions)  # save
    # new_num_arr = np.load('latent.npy')  # load
    # with open(output_dir / "latent.pkl", "wb") as f:
    #     pickle.dump(array, f)

    print("lol")


if __name__ == "__main__":
    main()
