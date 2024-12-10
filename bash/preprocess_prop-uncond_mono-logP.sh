#!/bin/bash

python -m fast_molopt.preprocess_prop --train data/uncond_mono_logP/isolated_ligands_8b_uncond_mono-train-smiles.txt \
                                      --prop_path data/uncond_mono_logP/isolated_ligands_8b_uncond_mono-train-props.txt \
                                      --split 100 --jobs 8 --output data/uncond_mono_logP/preprocessed
