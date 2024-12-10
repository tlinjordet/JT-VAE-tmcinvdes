#!/bin/bash

python -m fast_molopt.preprocess_prop --train data/uncond_bi_logP/isolated_ligands_8b_uncond_bi-train-smiles.txt \
                                      --prop_path data/uncond_bi_logP/isolated_ligands_8b_uncond_bi-train-props.txt \
                                      --split 100 --jobs 8 --output data/uncond_bi_logP/preprocessed
