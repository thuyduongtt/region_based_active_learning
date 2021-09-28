#!/bin/bash
cd DATA
datafolder=QB
if [ -d "$datafolder" ]; then
    echo "YEAH, $datafolder exist"
    echo "next download the resnet-v2-50 pretrained weight from tensorflow website..............."
fi
cd ..
pwd
echo "download the resnet-v2-50 pretrained weight............."
pretrainfolder=pretrain_model
if [ -d "$pretrainfolder" ]; then
    echo "YEAH, folder $pretrainfolder exists"
else
    echo "create the folder to save resnet-v2-50 pretrained weight"
    mkdir $pretrainfolder
fi
resnetname=resnet_v2_50.ckpt
cd pretrain_model
if [ -f "$resnetname" ]; then
    echo "YEAH, $resnetname exists"
    echo "next prepare the dataset ...................."
else
    echo "download the resnet-v2-50 pretrained weight............."
    wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
    tar -xvf resnet_v2_50_2017_04_14.tar.gz
    rm resnet_v2_50_2017_04_14.tar.gz
    rm train.graph
    rm eval.graph    
fi
cd ..
echo "print current directory"
pwd
echo "prepare the dataset"
python3 -c 'import data_utils.qb as qb;qb.transfer_data_to_dict()'
python3 -c 'import data_utils.qb as qb;qb.transfer_data_to_dict_test()'
echo "-------------------------------"
echo "YEAH, FINISH PREPARING THE DATA"
echo "-------------------------------"
