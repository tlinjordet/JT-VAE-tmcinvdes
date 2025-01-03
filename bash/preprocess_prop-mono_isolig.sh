#!/bin/bash

python -m fast_molopt.preprocess_prop --train data/mono_isolig/isolated_ligands_8b_uncond_mono-train-smiles.txt \
                                      --prop_path data/mono_isoligs/isolated_ligands_8b_uncond_mono-train-props.txt \
                                      --split 100 --jobs 8 --output data/mono_isolig/preprocessed
