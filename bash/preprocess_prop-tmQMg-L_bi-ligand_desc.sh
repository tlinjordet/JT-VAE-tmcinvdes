#!/bin/bash

python -m fast_molopt.preprocess_prop --train data/tmQMg-L_bi-ligand_desc/isolated_ligands_01_tmQMg-L_bi-train-smiles.txt \
                                      --prop_path data/tmQMg-L_bi-ligand_desc/isolated_ligands_01_tmQMg-L_bi-train-props.txt \
                                      --split 100 --jobs 8 --output data/tmQMg-L_bi-ligand_desc/preprocessed
