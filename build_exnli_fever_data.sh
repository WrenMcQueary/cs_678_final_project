#!/bin/bash
# Script to build the e-XNLI-fever dataset
# Assumes e-XNLI data has been downloaded in directory at top level
# and that the data has been split into train, test, and val files
# Command line argument -e optionally builds the english-only version

DATASET_DIR="e-XNLI-fever"
echo Building $DATASET_DIR

while getopts 'e' flag
do
  case "${flag}" in
    e) 
      ENGLISH=true
  esac
done


if [ -d "./data/e-XNLI-fever" ]
then
  echo Removing existing directory...
  rm -rf ./data/e-XNLI-fever
fi

echo Create Directory $DATASET_DIR
mkdir ./data/$DATASET_DIR
cd ./e-XNLI

python build_folder_for_limitedink_fever-style.py

cd ../data
if [ $ENGLISH ]
then
  echo "Build English-only version too"
  if [ -d "e-XNLI-just-english" ]
  then
    echo "Remove existing english-only directory..."
    rm -rf e-XNLI-just-english
  fi
  mkdir e-XNLI-just-english
  cp $DATASET_DIR/test.jsonl $DATASET_DIR/train.jsonl $DATASET_DIR/val.jsonl e-XNLI-just-english
  mkdir e-XNLI-just-english/docs
  cp $DATASET_DIR/docs/en_* e-XNLI-just-english/docs
  cd ../e-XNLI
  python remove_all_but_english_from_train_val_test_jsonl_files.py
else
  echo "Don't build English-only version"
fi