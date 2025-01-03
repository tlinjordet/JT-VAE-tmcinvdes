#!/bin/bash

python -m fast_molopt.preprocess_prop --train data/bi_isolig/isolated_ligands_8b_uncond_bi-train-smiles.txt \
                                      --prop_path data/bi_isoligs/isolated_ligands_8b_uncond_bi-train-props.txt \
                                      --split 100 --jobs 8 --output data/bi_isolig/preprocessed
