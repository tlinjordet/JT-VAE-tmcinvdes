#!/bin/bash
python -m fast_molopt.optimize --input_dir_path data/uncond_mono_logP/sample_input.csv  \
                               --vocab_path vocabs/logP_8b_uncond_mono-vocab.txt \
                               --model_path models/uncond_mono_logP/model.epoch-149 \
                               --denticity monodentate \
                               --train_mode None \
                               --labeling logP
