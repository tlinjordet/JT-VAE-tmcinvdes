#!/bin/bash
python -m fast_molopt.vae_train_prop --train data/labeled_set_combined/repro-preprocessed/ --vocab vocabs/labeled_set_combined-repro-vocab.txt \
                                --save_dir models/labeled_set_combined-repro-denticity \
                                --train_mode denticity
# Start with only default hyperparameters.
