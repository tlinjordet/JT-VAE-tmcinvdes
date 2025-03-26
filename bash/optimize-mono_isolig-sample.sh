#!/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'
python -m fast_molopt.optimize --input_dir_path data/mono_isolig/uncond_mono-min15k-labeled-included-isolated_ligands-sampled_for_cond_mono.csv  \
                               --vocab_path vocabs/mono_isolig-vocab.txt \
                               --model_path models/mono_isolig/model.epoch-149 \
                               --denticity monodentate \
                               --train_mode None \
                               --labeling isolated_ligands
