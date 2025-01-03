#!/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'
python -m fast_molopt.optimize --input_dir_path data/tmQMg-L_bi-ligand_desc/tmQMg-L_bi-isolated_ligands-sampled_for_cond_bi.csv  \
                               --vocab_path vocabs/tmQMg-L_bi-vocab.txt \
                               --model_path models/tmQMg-L_bi-ligand_desc/model.epoch-149 \
                               --denticity bidentate \
                               --train_mode None \
                               --labeling isolated_ligands
