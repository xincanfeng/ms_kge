#!/usr/bin/env bash

# argues
MODEL=$1
DATASET=$2
ID=$3

# file path
file_path=models/$MODEL'_'$DATASET'_'$ID/train.log
output_path=results/$MODEL'_'$DATASET'_'$ID.txt
echo $file_path
# file_path=test.log
# read train.log file
cat $file_path | while read line
do
  info=${line:25}
  prefix=${info: 4: 4}
  if [ $prefix = 'Test' ];then
    echo $info >> $output_path
  fi
done
