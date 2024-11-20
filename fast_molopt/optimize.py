import argparse
import csv
import json
import os
import sys
import time
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

    parser.add_argument("--input_dir_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--model_path", required=True)
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

    parser.add_argument("--lr", type=float, default=0.02)
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
    parser.add_argument(
        "--denticity",  # initial. TODO support mixed?
        choices=["monodentate", "bidentate"],
        type=str,
        default="monodentate",
    )
    parser.add_argument(
        "--desired_denticity",
        choices=["monodentate", "bidentate"],
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train_mode",
        nargs="*",
        default=[],
        choices=["denticity", "isomer", "None"],
        help="Selects which extra property terms that where included in the training, when using the argument each extra term should be separated by a space",
    )
    parser.add_argument(
        "--initial_isomer",
        choices=["cis", "trans"],
        type=str,
        default=None,  # "cis", # TODO later modify the configurations to be fully optional and modular.
    )
    parser.add_argument(
        "--desired_isomer",
        choices=["cis", "trans"],
        type=str,
        default=None,  # "cis", # TODO later modify the configurations to be fully optional and modular.
    )
    parser.add_argument(
        "--labeling",
        choices=["DFT", "isolated_ligands", "None"],  # TODO support mixed?
        type=str,
        default="DFT",
    )

    return parser.parse_args(arg_list)


def main():
    opts = parse_args()

    vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
    vocab = Vocab(vocab)

    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    sim_cutoff = float(opts.cutoff)

    model = JTpropVAE(
        vocab,
        int(hidden_size),
        int(latent_size),
        int(opts.depthT),
        int(opts.depthG),
        train_mode=opts.train_mode,
    ).cuda()
    print(model)
    model.load_state_dict(torch.load(opts.model_path))
    model = model.cuda()

    output_dir = Path(f"dump_{time.strftime('%Y%m%d-%H%M%S')}")
    output_dir.mkdir(exist_ok=True)

    input_dir_path = Path(opts.input_dir_path)
    input_dir_df = pd.read_csv(input_dir_path)
    output_dir_smiles = input_dir_path.parent / "smiles_samples.txt"
    output_dir_props = input_dir_path.parent / "smiles_samples_props.txt"

    # Process dataframe to input files
    extra_properties = []
    create_input_files(
        input_dir_df,
        output_dir_smiles,
        output_dir_props,
        extra_properties,
        opts.labeling,
    )
    main_preprocess(output_dir_smiles, output_dir_props, output_dir, opts.nsplits)

    loader = MolTreeFolder_prop(
        output_dir,
        vocab,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        optimize=True,
    )

    with open(output_dir / "opts.json", "w") as file:
        json.dump(vars(opts), file)

    if opts.labeling == "DFT":
        directions = [
            ("homo_lumo_gap", "maximize"),
            ("homo_lumo_gap", "minimize"),
            ("both", "maximize"),
            ("both", "minimize"),
            ("charge", "maximize"),
            ("charge", "minimize"),
            ("both_opposite", "maximize"),
            ("both_opposite", "minimize"),
        ]
    elif opts.labeling == "isolated_ligands":
        directions = [
            ("log P", "maximize"),
            ("log P", "minimize"),
            ("log P + G parameter", "maximize"),
            ("log P + G parameter", "minimize"),
            ("G parameter", "maximize"),
            ("G parameter", "minimize"),
            ("log P - G parameter", "maximize"),
            ("log P - G parameter", "minimize"),
        ]
    elif opts.labeling == "DFT_and_isolated_ligands":
        pass
    #     directions = [
    #     # ("homo_lumo_gap", "maximize"),
    #     # ("homo_lumo_gap", "minimize"),
    #     # ("both", "maximize"),
    #     # ("both", "minimize"),
    #     # ("charge", "maximize"),
    #     # ("charge", "minimize"),
    #     # ("both_opposite", "maximize"),
    #     # ("both_opposite", "minimize"),
    #     ("log P", "maximize"),
    #     ("log P", "minimize"),
    #     ("log P + G parameter", "maximize"),
    #     ("log P + G parameter", "minimize"),
    #     ("G parameter", "maximize"),
    #     ("G parameter", "minimize"),
    #     ("log P - G parameter", "maximize"),
    #     ("log P - G parameter", "minimize"),
    # ]

    for i, batch in enumerate(loader):
        for dir in directions:
            current_type = dir[0]
            smiles = batch[0][0].smiles
            print(i, smiles, "-".join(dir))

            # Set the minimize flag based on the given direction
            minimize = True if dir[1] == "minimize" else False

            new_smiles, sim = model.optimize(
                batch,
                sim_cutoff=sim_cutoff,
                lr=opts.lr,
                num_iter=250,
                type=current_type,
                prob_decode=False,
                minimize=minimize,
                desired_denticity=opts.desired_denticity,
                desired_isomer=opts.desired_isomer,
            )

            # Write a row to a csv file.
            with open(output_dir / "optimize_results.csv", "a") as f1:
                writer = csv.writer(
                    f1,
                    delimiter=",",
                    lineterminator="\n",
                )
                # Get the row elements of the original data
                r = input_dir_df.iloc[i].to_list()
                # Append the data from optimized row
                list_of_props = r + [smiles, new_smiles, sim, current_type, minimize]
                writer.writerow(list_of_props)


def create_input_files(
    input_df, output_dir_smiles, output_dir_props, extra_properties, labeling
):
    if os.path.exists(output_dir_smiles):
        os.remove(output_dir_smiles)
    if os.path.exists(output_dir_props):
        os.remove(output_dir_props)
    if labeling == "DFT":
        properties = ["homo-lumo", "Ir-cm5"]
        encoded_smiles_col = "sub_smi"
    elif labeling == "isolated_ligands":
        properties = ["log P", "G parameter"]
        encoded_smiles_col = "Encoded SMILES"
    for term in extra_properties:
        properties.append(term)

    # property_header = ",".join(properties)
    input_df[properties].to_csv(output_dir_props, index=None, sep=",")
    input_df[encoded_smiles_col].to_csv(
        output_dir_smiles, index=None, sep=",", header=None
    )


if __name__ == "__main__":
    main()
