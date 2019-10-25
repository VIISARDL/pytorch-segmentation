

PATHDATASET=$HOME/.datasets/datasciencebowl

mkdir $PATHDATASET
mkdir $PATHDATASET/train
mkdir $PATHDATASET/test

#kaggle competitions download -c data-science-bowl-2018 -p $PATHDATASET
#unzip $PATHDATASET/stage1_test.zip -d $PATHDATASET/test/images 
#unzip $PATHDATASET/stage1_train.zip -d $PATHDATASET/train/images/

unzip $PATHDATASET/stage1_train_labels.csv.zip -d $PATHDATASET/
unzip $PATHDATASET/stage1_sample_submission.csv.zip -d $PATHDATASET/
unzip $PATHDATASET/stage1_solution.csv.zip -d $PATHDATASET/

#stage1_sample_submission.csv
#stage1_solution.csv
#stage1_train_labels.csv
#stage2_sample_submission_final.csv
#stage2_test_final.zip


