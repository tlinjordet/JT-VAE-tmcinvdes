import argparse
import math
import os
import random
import sys
from collections import deque
from pathlib import Path

import rdkit
import rdkit.Chem as Chem
import sascorer
import torch
import torch.nn as nn
from rdkit.Chem import Descriptors
from torch.autograd import Variable

from fast_jtnn import *

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))
from fast_molvae.preprocess import main_preprocess

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", required=True)
parser.add_argument("--vocab_path", required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--output_path", default="optimize-processed")
parser.add_argument("--nsplits", type=int, default=10)

parser.add_argument("--hidden_size", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--latent_size", type=int, default=24)
parser.add_argument("--depthT", type=int, default=7)
parser.add_argument("--depthG", type=int, default=2)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--clip_norm", type=float, default=50.0)
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--step_beta", type=float, default=0.002)
parser.add_argument("--max_beta", type=float, default=1.0)
parser.add_argument("--cutoff", type=float, default=0.0)
parser.add_argument("--warmup", type=int, default=40000)

parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--anneal_rate", type=float, default=0.9)
parser.add_argument("--anneal_iter", type=int, default=40000)
parser.add_argument("--kl_anneal_iter", type=int, default=2000)
parser.add_argument("--print_iter", type=int, default=10)
parser.add_argument("--save_iter", type=int, default=5000)

opts = parser.parse_args()
print(opts)

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

# data = []
# with open(opts.test_path) as f:
#     for line in f:
#         s = line.strip("\r\n ").split()[0]
#         data.append(s)

res = []


# Preprocess data
# Check if processed stuff exists
preprocess = False
with os.scandir(opts.output_path) as it:
    if any(it):
        print("Skipping preprocessing")
    else:
        preprocess = True
if preprocess:
    main_preprocess(opts.test_path, opts.output_path, opts.nsplits)

loader = MolTreeFolder(
    opts.output_path, vocab, batch_size=1, shuffle=False
)  # , num_workers=4)

for batch in loader:
    # Extract smiles
    smiles = batch[0][0].smiles

    mol = Chem.MolFromSmiles(smiles)
    score = Descriptors.MolLogP(mol) - sascorer.calculateScore(mol)

    new_smiles, sim = model.optimize(batch, sim_cutoff=sim_cutoff, lr=2, num_iter=80)
    new_mol = Chem.MolFromSmiles(new_smiles)
    new_score = Descriptors.MolLogP(new_mol) - sascorer.calculateScore(new_mol)

    res.append((new_score - score, sim, score, new_score, smiles, new_smiles))
    print(new_score - score, sim, score, new_score, smiles, new_smiles)

print(sum([x[0] for x in res]), sum([x[1] for x in res]))
