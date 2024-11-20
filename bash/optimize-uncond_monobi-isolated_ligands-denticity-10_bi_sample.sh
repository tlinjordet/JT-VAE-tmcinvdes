#!/bin/bash
python -m fast_molopt.optimize --input_dir_path data/uncond_monobi_ligand_desc/10_bi_sample.csv  \
                               --vocab_path vocabs/isolated_ligands_8b_uncond_monobi-vocab.txt \
                               --model_path models/uncond_monobi_ligand_desc-denticity/model.epoch-149 \
                               --denticity bidentate \
                               --train_mode denticity \
                               --labeling isolated_ligands
