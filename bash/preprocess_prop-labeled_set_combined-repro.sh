#!/bin/bash

python -m fast_molopt.preprocess_prop --train data/labeled_set_combined/train_smiles.txt \
                                      --prop_path data/labeled_set_combined/train_props.txt \
                                      --split 100 --jobs 8 --output data/labeled_set_combined/repro-preprocessed
