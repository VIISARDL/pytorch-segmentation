#!/bin/bash

# parameters
DATA=$HOME/.datasets
NAMEDATASET='cambiaext'
PROJECT='../out/netruns'
EPOCHS=30
BATCHSIZE=48
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=100
WORKERS=20
RESUME='chk000015.pth.tar' #model_best
GPU=0
ARCH='unetresnet34'
LOSS='mcedice'
OPT='adam'
SCHEDULER='fixed'
SNAPSHOT=5
IMAGESIZE=256 #256 #64
EXP_NAME='baseline_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_0001'


#rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
#rm -rf $PROJECT/$EXP_NAME/
#mkdir $PROJECT
#mkdir $PROJECT/$EXP_NAME


## execute
python ../train.py \
$DATA/$NAMEDATASET \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--batch-size=$BATCHSIZE \
--learning-rate=$LEARNING_RATE \
--momentum=$MOMENTUM \
--print-freq=$PRINT_FREQ \
--workers=$WORKERS \
--resume=$RESUME \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--snapshot=$SNAPSHOT \
--scheduler=$SCHEDULER \
--arch=$ARCH \
--image-size=$IMAGESIZE \
--finetuning \
--parallel \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

