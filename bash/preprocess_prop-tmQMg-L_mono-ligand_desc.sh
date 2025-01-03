#!/bin/bash

python -m fast_molopt.preprocess_prop --train data/tmQMg-L_mono-ligand_desc/isolated_ligands_01_tmQMg-L_mono-train-smiles.txt \
                                      --prop_path data/tmQMg-L_mono-ligand_desc/isolated_ligands_01_tmQMg-L_mono-train-props.txt \
                                      --split 100 --jobs 8 --output data/tmQMg-L_mono-ligand_desc/preprocessed
