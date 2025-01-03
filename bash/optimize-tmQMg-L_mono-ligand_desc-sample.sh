#!/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'
python -m fast_molopt.optimize --input_dir_path data/tmQMg-L_mono-ligand_desc/tmQMg-L_mono-isolated_ligands-sampled_for_cond_mono.csv  \
                               --vocab_path vocabs/tmQMg-L_mono-vocab.txt \
                               --model_path models/tmQMg-L_mono-ligand_desc/model.epoch-149 \
                               --denticity monodentate \
                               --train_mode None \
                               --labeling isolated_ligands
