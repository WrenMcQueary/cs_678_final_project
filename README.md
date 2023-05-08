Final project for CS 678 at George Mason University.  Builds on top of the tool [LimitedInk](https://github.com/huashen218/LimitedInk) and the dataset [explaiNLI (aka e-XNLI)](https://github.com/KeremZaman/explaiNLI).

## Data
The data for our multilinguality experiments is contained in the e-XNLI directory. It contains the original e-XNLI csv file from explainNLI as well as split train, test, and val csv files. The \data\e-XNLI-fever directory needs to be populated with the properly formatted data in order to run the e-XNLI experiments. To populate the directory, run:
```
$ ./build_exnli_fever_data.sh
```
Passing an -e option also builds the \data\e-XNLI-just-english directory, which may be useful for testing and debugging purposes. This can be executed as follows:
```
$ ./build_exnli_fever_data.sh -e
```

## Run LimitedInk with e-XNLI
Here is a sample bash script for running LimitedInk on e-XNLI:
```
set -x;
set -e;
REPO_PATH="/home/jhus/cs678/cs_678_final_project"
export PYTHONPATH=$REPO_PATH

k=0.8;
rand="1234";
SAVE_DIR="/scratch/jhus/exnli/token_rationale/length_level_$k/seed_$rand"
mkdir -p $SAVE_DIR
LOG_DIR="$SAVE_DIR/train.log";

#CUDA_VISIBLE_DEVICES=1 
python ../main.py --data_dir "$REPO_PATH/data/e-XNLI-fever" --save_dir $SAVE_DIR --configs "$REPO_PATH/limitedink/params/exnli_config_token.json" --length $k --seed $rand > $LOG_DIR
```
