#!/bin/bash
python -m fast_molopt.optimize --input_dir_path data/labeled_set_combined/small_bi_sample.csv \
                               --vocab_path vocabs/labeled_set_combined-repro-vocab.txt \
                               --model_path models/labeled_set_combined-repro/model.epoch-149 \
                               --denticity bidentate \
                               --desired_denticity monodentate \
                               --train_mode denticity
