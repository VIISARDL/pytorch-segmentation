#!/bin/bash

# parameters
DATA=$HOME/.datasets
NAMEDATASET='cambiaext'
PROJECT='../out/netruns'
EPOCHS=120
BATCHSIZETRAIN=45
BATCHSIZETEST=45
LEARNING_RATE=0.00001
MOMENTUM=0.5
PRINT_FREQ=100
WORKERS=30
RESUME='chk000015.pth.tar' #model_best, chk000000
GPU=0
ARCH='unetresnet34'
LOSS='mcedice'
OPT='adam'
SCHEDULER='fixed'
SNAPSHOT=5
COUNTTRAIN=50000
COUNTTEST=5000
IMAGECROP=512
IMAGESIZE=256 #256 #64
NUMCHANNELS=3
NUMCLASSES=3
EXP_NAME='baseline_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_0001'


# rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
# rm -rf $PROJECT/$EXP_NAME/
# mkdir $PROJECT
# mkdir $PROJECT/$EXP_NAME


## execute
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../train.py \
$DATA/$NAMEDATASET \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--batch-size-train=$BATCHSIZETRAIN \
--batch-size-test=$BATCHSIZETEST \
--count-train=$COUNTTRAIN \
--count-test=$COUNTTEST \
--num-classes=$NUMCLASSES \
--num-channels=$NUMCHANNELS \
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
--image-crop=$IMAGECROP \
--finetuning \
--parallel \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

