#!/bin/bash
python -m fast_molopt.optimize --input_dir_path data/labeled_set_combined/small_mono_sample.csv \
                               --vocab_path vocabs/labeled_set_combined-repro-vocab.txt \
                               --model_path models/labeled_set_combined-repro/model.epoch-149 \
                               --denticity monodentate \
                               --desired_denticity bidentate \
                               --train_mode denticity
