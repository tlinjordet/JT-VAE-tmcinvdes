#!/bin/bash
python -m fast_molopt.vae_train_prop --train data/uncond_mono_logP/preprocessed/ \
                                     --vocab vocabs/logP_8b_uncond_mono-vocab.txt \
                                     --save_dir models/uncond_mono_logP/ \
                                     --train_mode None
# Start with only default hyperparameters.
