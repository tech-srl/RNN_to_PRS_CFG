#!/bin/bash



BS=500 # batch size

iters_per_lr=3
lrs="(0.004,0.002,0.001,0.0003,0.004,0.001,0.0003)"
input_dim=10

train_set_size=2e4
val_set_size=2e3
val_check_freq=10

extraction_limit=1000 # extraction time limit

subfolder=your_subfolder_here

python3 print_grammars_to_subfolder.py --subfolder=$subfolder # store the grammars as they were when making these RNNs, for easy recall if things shift over time

python3 make_rnns_and_dfas.py --lang=TACAS.all --make-new \
--hidden-dim=100 --num-layers=3 --input-dim=10 --RNNClass=LSTM \
--train-set-size=$train_set_size --validation-set-size=$val_set_size \
--batch-size=$BS --check-validation-improvement-every=$val_check_freq \
--transition-reject-threshold=0.01 --initial-split-depth=10 \
--learning-rates=$lrs --iterations-per-learning-rate=$iters_per_lr \
--token-predictor-samples=100 --token-predictor-cutoff=50 --extraction-time-limit=$extraction_limit \
--subfolder=$subfolder

# template for running only some languages
# for l in LG13 LG14 LG15
# do
#     python3 make_rnns_and_dfas.py --lang=TACAS.$l --make-new \
#     --hidden-dim=100 --num-layers=3 --input-dim=10 --RNNClass=LSTM \
#     --train-set-size=$train_set_size --validation-set-size=$val_set_size \
#     --batch-size=$BS --check-validation-improvement-every=$val_check_freq \
#     --transition-reject-threshold=0.01 --initial-split-depth=10 \
#     --learning-rates=$lrs --iterations-per-learning-rate=$iters_per_lr \
#     --token-predictor-samples=100 --token-predictor-cutoff=50 --extraction-time-limit=$extraction_limit \
#     --subfolder=$subfolder
# done


