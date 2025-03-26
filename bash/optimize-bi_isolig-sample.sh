#!/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'
python -m fast_molopt.optimize --input_dir_path data/bi_isolig/uncond_bi-min10k-labeled-included-isolated_ligands-sampled_for_cond_bi.csv  \
                               --vocab_path vocabs/bi_isolig-vocab.txt \
                               --model_path models/bi_isolig/model.epoch-149 \
                               --denticity bidentate \
                               --train_mode None \
                               --labeling isolated_ligands
