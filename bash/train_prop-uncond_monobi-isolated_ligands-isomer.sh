#!/bin/bash
python -m fast_molopt.vae_train_prop --train data/uncond_monobi_ligand_desc/preprocessed/ \
                                     --vocab vocabs/isolated_ligands_8b_uncond_monobi-vocab.txt \
                                     --save_dir models/uncond_monobi_ligand_desc-isomer/ \
                                     --train_mode isomer
# Start with only default hyperparameters.
