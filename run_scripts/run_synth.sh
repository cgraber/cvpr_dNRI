GPU=0 # Set to whatever GPU you want to use

# First: process data

# Make sure to replace this with the directory containing the data files
DATA_PATH='data/synth/'
mkdir -p $DATA_PATH
# If for some reason you want to regenerate this data, uncomment this line
#python dnri/datasets/small_synth_data.py --output_dir $DATA_PATH

BASE_RESULTS_DIR="results/synth/"

for SEED in {1..5}
do

    WORKING_DIR="${BASE_RESULTS_DIR}nri/seed_${SEED}/"
    ENCODER_ARGS='--num_edge_types 2 --encoder_hidden 256 --skip_first --encoder_mlp_hidden 256 --encoder_mlp_num_layers 3'
    DECODER_ARGS=''
    MODEL_ARGS="--model_type nri --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
    TRAINING_ARGS='--no_edge_prior 0.9 --batch_size 16 --lr 5e-4 --use_adam --num_epochs 200 --lr_decay_factor 0.5 --lr_decay_steps 200 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/small_synth_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/small_synth_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/small_synth_experiment.py --gpu --mode eval --test_burn_in_steps 25 --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results_25step.txt"


    WORKING_DIR="${BASE_RESULTS_DIR}dnri/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256 --decoder_type ref_mlp"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dnri --graph_type dynamic --skip_first --num_edge_types 2 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--add_uniform_prior --no_edge_prior 0.9 --batch_size 16 --lr 5e-4 --use_adam --num_epochs 200 --lr_decay_factor 0.5 --lr_decay_steps 200 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/small_synth_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/small_synth_experiment.py --gpu --load_best_model --mode eval --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
    CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/small_synth_experiment.py --gpu --load_best_model --test_burn_in_steps 25 --mode eval --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results_25step.txt"

done