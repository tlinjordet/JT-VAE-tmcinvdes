#!/bin/bash

python -m fast_molopt.preprocess_prop --train data/uncond_monobi_ligand_desc/isolated_ligands_8b_uncond_monobi-train-smiles.txt \
                                      --prop_path data/uncond_monobi_ligand_desc/isolated_ligands_8b_uncond_monobi-train-props.txt \
                                      --split 100 --jobs 8 --output data/uncond_monobi_ligand_desc/preprocessed
