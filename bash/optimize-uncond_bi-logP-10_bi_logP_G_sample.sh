#!/bin/bash
python -m fast_molopt.optimize --input_dir_path data/uncond_bi_logP/sample_input.csv  \
                               --vocab_path vocabs/logP_8b_uncond_bi-vocab.txt \
                               --model_path models/uncond_bi_logP/model.epoch-149 \
                               --denticity bidentate \
                               --train_mode None \
                               --labeling logP
