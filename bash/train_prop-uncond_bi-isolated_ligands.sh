#!/bin/bash
python -m fast_molopt.vae_train_prop --train data/uncond_bi_ligand_desc/preprocessed/ \
                                     --vocab vocabs/isolated_ligands_8b_uncond_bi-vocab.txt \
                                     --save_dir models/uncond_bi_ligand_desc/ \
                                     --train_mode None
# Start with only default hyperparameters.
