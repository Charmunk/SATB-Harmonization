#!/bin/bash

mkdir cocochorales_tiny_v1_zipped
cd cocochorales_tiny_v1_zipped

# download md5
wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/cocochorales_md5s.txt

# download main dataset
mkdir main_dataset
mkdir main_dataset/train
mkdir main_dataset/valid
mkdir main_dataset/test
for i in 1; do
  wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/train/"$i".tar.bz2 -P main_dataset/train
done

# download metadata
mkdir metadata
mkdir metadata/train
mkdir metadata/valid
mkdir metadata/test
for i in 1; do
  wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/metadata/train/"$i".tar.bz2 -P metadata/train
done

# download f0
mkdir f0
mkdir f0/train
mkdir f0/valid
mkdir f0/test
for i in 1; do
  wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/f0/train/"$i".tar.bz2 -P f0/train
done

# download note expression
mkdir note_expression
mkdir note_expression/train
mkdir note_expression/valid
mkdir note_expression/test
for i in 1; do
  wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/note_expression/train/"$i".tar.bz2 -P note_expression/train
done

# download synthesis parameters
mkdir synthesis_parameters
mkdir synthesis_parameters/train
mkdir synthesis_parameters/valid
mkdir synthesis_parameters/test
for i in 1; do
  wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/synthesis_parameters/train/"$i".tar.bz2 -P synthesis_parameters/train
done