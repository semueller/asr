#!/bin/sh

mkdir data
cd data

mkdir tf
cd tf
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xvzf speech_commands_v0.02.tar.gz
rm speech_commands_v0.02.tar.gz
rm -rf _background_noise_
cd ../../
