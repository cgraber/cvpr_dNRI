GPU=0 # Set to whatever GPU you want to use

# First: process data

# Make sure to replace this with the directory containing the data files
IN_DIR='data/motion/raw/118/'
DATA_PATH='data/motion/processed/118/'
mkdir -p $DATA_PATH
python -u dnri/datasets/cmu_motion_data.py --data_path $IN_DIR --out_path $DATA_PATH --num_train_trials 14 --num_val_trials 7 --train_seq_len -1 --test_seq_len -1

BASE_RESULTS_DIR="results/motion_118/"

EVAL_ARGS="--batch_size 16 --val_batch_size 1"

for SEED in {1..5}
do
    WORKING_DIR="${BASE_RESULTS_DIR}/nri_2edge/seed_${SEED}/"
    ENCODER_ARGS='--num_edge_types 2 --encoder_hidden 256 --skip_first --encoder_mlp_hidden 256 --encoder_mlp_num_layers 3'
    DECODER_ARGS=''
    MODEL_ARGS="--model_type nri --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
    TRAINING_ARGS='--expand_train --train_data_len 50 --batch_size 8 --val_batch_size 1 --lr 5e-4 --use_adam --num_epochs 600 --lr_decay_factor 0.5 --lr_decay_steps 300 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing'
    mkdir -p $WORKING_DIR
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/motion_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/motion_experiment.py --gpu --mode eval_fixedwindow --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $EVAL_ARGS |& tee "${WORKING_DIR}fixedwindow_eval_results.txt"
    MODEL_ARGS="--graph_type dynamic ${ENCODER_ARGS} ${DECODER_ARGS}"
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/motion_experiment.py --gpu --mode eval_fixedwindow --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $EVAL_ARGS --error_out_name prediction_errors_dynamic.npy |& tee "${WORKING_DIR}fixedwindow_eval_results_dynamic.txt"

    WORKING_DIR="${BASE_RESULTS_DIR}/nri_4edge/seed_${SEED}/"
    ENCODER_ARGS='--num_edge_types 4 --encoder_hidden 256 --skip_first --encoder_mlp_hidden 256 --encoder_mlp_num_layers 3'
    DECODER_ARGS=''
    MODEL_ARGS="--model_type nri --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
    TRAINING_ARGS='--expand_train --train_data_len 50 --batch_size 8 --val_batch_size 1 --lr 5e-4 --use_adam --num_epochs 600 --lr_decay_factor 0.5 --lr_decay_steps 300 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing'
    mkdir -p $WORKING_DIR
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/motion_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/motion_experiment.py --gpu --mode eval_fixedwindow --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $EVAL_ARGS |& tee "${WORKING_DIR}fixedwindow_eval_results.txt"
    MODEL_ARGS="--graph_type dynamic ${ENCODER_ARGS} ${DECODER_ARGS}"
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/motion_experiment.py --gpu --mode eval_fixedwindow --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $EVAL_ARGS --error_out_name prediction_errors_dynamic.npy |& tee "${WORKING_DIR}fixedwindow_eval_results_dynamic.txt"

    WORKING_DIR="${BASE_RESULTS_DIR}/dnri_2edge/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dnri --graph_type dynamic --skip_first --num_edge_types 2 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--expand_train --train_data_len 50 --val_batch_size 1 --batch_size 2 --accumulate_steps 4 --lr 5e-4 --use_adam --num_epochs 600 --lr_decay_factor 0.5 --lr_decay_steps 300 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/motion_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/motion_experiment.py --gpu --mode eval_fixedwindow --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $EVAL_ARGS |& tee "${WORKING_DIR}fixedwindow_eval_results.txt"


    WORKING_DIR="${BASE_RESULTS_DIR}/dnri_4edge/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dnri --graph_type dynamic --skip_first --num_edge_types 4 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--expand_train --train_data_len 50 --val_batch_size 1 --batch_size 2 --accumulate_steps 4 --lr 5e-4 --use_adam --num_epochs 600 --lr_decay_factor 0.5 --lr_decay_steps 300 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/motion_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/motion_experiment.py --gpu --mode eval_fixedwindow --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $EVAL_ARGS |& tee "${WORKING_DIR}fixedwindow_eval_results.txt"

done