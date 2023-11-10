import math
import random
import sys
from collections import deque
from optparse import OptionParser

import rdkit
import rdkit.Chem as Chem
import sascorer
import torch
import torch.nn as nn
from rdkit.Chem import Descriptors
from torch.autograd import Variable

from fast_jtnn import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=50)
parser.add_option("-l", "--latent", dest="latent_size", default=24)
parser.add_option("-d", "--depthT", dest="depthT", default=7)
parser.add_option("-g", "--depthG", dest="depthG", default=2)
parser.add_option("-s", "--sim", dest="cutoff", default=0.0)
opts, args = parser.parse_args()

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

data = []
with open(opts.test_path) as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

res = []
for smiles in data:
    mol = Chem.MolFromSmiles(smiles)
    score = Descriptors.MolLogP(mol) - sascorer.calculateScore(mol)

    new_smiles, sim = model.optimize(smiles, sim_cutoff=sim_cutoff, lr=2, num_iter=80)
    new_mol = Chem.MolFromSmiles(new_smiles)
    new_score = Descriptors.MolLogP(new_mol) - sascorer.calculateScore(new_mol)

    res.append((new_score - score, sim, score, new_score, smiles, new_smiles))
    print(new_score - score, sim, score, new_score, smiles, new_smiles)

print(sum([x[0] for x in res]), sum([x[1] for x in res]))
