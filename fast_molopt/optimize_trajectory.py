import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import rdkit
import rdkit.Chem as Chem
import torch
import torch.nn as nn
from rdkit.Chem import Descriptors
from torch.autograd import Variable

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))
from fast_jtnn import JTpropVAE, MolTreeFolder_prop, Vocab
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

    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--model_path", required=True)
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
        "--type",
        type=str,
        default="homo_lumo_gap",
        help="Which property to optimize on",
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
    sim_cutoff = float(opts.cutoff)

    model = JTpropVAE(
        vocab, int(hidden_size), int(latent_size), int(opts.depthT), int(opts.depthG)
    ).cuda()
    print(model)
    model.load_state_dict(torch.load(opts.model_path))
    model = model.cuda()

    output_dir = Path(f"traj_{time.strftime('%Y%m%d-%H%M%S')}")
    output_dir.mkdir(exist_ok=True)

    input_dir_path = Path(
        "/home/magstr/git/xyz2mol_tm/jt_vae_research/notebooks/conditional_opt_16_samples/smiles_samples.csv"
    )
    input_dir_df = pd.read_csv(input_dir_path)
    output_dir_smiles = input_dir_path.parent / "smiles_samples.txt"
    output_dir_props = input_dir_path.parent / "smiles_samples_props.txt"

    # Process dataframe to input files
    create_input_files(input_dir_df, output_dir_smiles, output_dir_props)
    main_preprocess(output_dir_smiles, output_dir_props, output_dir, opts.nsplits)

    loader = MolTreeFolder_prop(
        output_dir,
        vocab,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        optimize=True,
    )

    defaultdict(list)

    with open(output_dir / "opts.json", "w") as file:
        json.dump(vars(opts), file)

    for i, (batch, (id_row, row)) in enumerate(zip(loader, input_dir_df.iterrows())):
        current_type = row["current_type"]
        smiles = batch[0][0].smiles
        print(i, smiles)

        batch[1].numpy().squeeze()

        Chem.MolFromSmiles(smiles)
        minimize = True if row["minimize"] else False
        new_smiles, sim = model.optimize(
            batch,
            sim_cutoff=sim_cutoff,
            lr=opts.lr,
            num_iter=250,
            type=current_type,
            prob_decode=False,
            minimize=minimize,
        )

        Chem.MolFromSmiles(new_smiles)
        # results["new_smiles"].append(new_smiles)
        # results["sim"].append(sim)
        # results["type"].append(current_type)
        # results["minimize"].append(minimize)

        # Write a row to a csv file.
        with open(output_dir / "optimize_results.csv", "a") as f1:
            csv.writer(
                f1,
                delimiter=",",
                lineterminator="\n",
            )
            # Get the row elements of the original data
            # r = input_df.iloc[i].to_list()
            # Append the data from optimized row
            # list_of_props = r + [smiles, new_smiles, sim, current_type, minimize]
            # writer.writerow(list_of_props)


def create_input_files(input_df, output_dir_smiles, output_dir_props):
    if os.path.exists(output_dir_smiles):
        os.remove(output_dir_smiles)
    if os.path.exists(output_dir_props):
        os.remove(output_dir_props)

    for i, row in input_df.iterrows():
        with open(
            output_dir_smiles,
            "a",
        ) as f:
            f.write(f"{row['smiles']}\n")
        with open(
            output_dir_props,
            "a",
        ) as f:
            f.write(f"{row['homo-lumo']},{row['Ir-cm5']}\n")


if __name__ == "__main__":
    main()
