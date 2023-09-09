#!/bin/bash

# GPU id
GPUID=3

# Sub-model options:

# SUB_MODEL_NAME="ComplEx_FB15k-237_none"
# SUB_MODEL_NAME="ComplEx_FB15k-237_cnt_default" 
# SUB_MODEL_NAME="ComplEx_wn18rr_none"
# SUB_MODEL_NAME="ComplEx_wn18rr_cnt_default" 

# SUB_MODEL_NAME="RotatE_FB15k-237_none" 
# SUB_MODEL_NAME="RotatE_FB15k-237_cnt_default" 
# SUB_MODEL_NAME="RotatE_wn18rr_none"
# SUB_MODEL_NAME="RotatE_wn18rr_cnt_default"
# SUB_MODEL_NAME="RotatE_YAGO3-10_none" 
# SUB_MODEL_NAME="RotatE_YAGO3-10_cnt_default"

# SUB_MODEL_NAME="HAKE_FB15k-237_none"
# SUB_MODEL_NAME="HAKE_FB15k-237_cnt_default"
# SUB_MODEL_NAME="HAKE_wn18rr_none"
# SUB_MODEL_NAME="HAKE_wn18rr_cnt_default"
# SUB_MODEL_NAME="HAKE_YAGO3-10_none"
# SUB_MODEL_NAME="HAKE_YAGO3-10_cnt_default"
                                                                          
# SUB_MODEL_NAME="TransE_FB15k-237_none"
# SUB_MODEL_NAME="TransE_FB15k-237_cnt_default"
# SUB_MODEL_NAME="TransE_wn18rr_none"
# SUB_MODEL_NAME="TransE_wn18rr_cnt_default"

# SUB_MODEL_NAME="DistMult_FB15k-237_none"
# SUB_MODEL_NAME="DistMult_FB15k-237_cnt_default"
SUB_MODEL_NAME="DistMult_wn18rr_none" # 
# SUB_MODEL_NAME="DistMult_wn18rr_cnt_default"

# Directory path to a subsampling model
SUB_MODEL="--subsampling_model ./models/${SUB_MODEL_NAME}"

# Temparature for subsampling
SUB_TEMP="--subsampling_model_temperature 0.5"

# Ratio of the model-based subsampling:
MR="1.0"
MBS_RATIO="--mbs_ratio ${MR}"

# Type of model-based subsampling
# You can choose the following methods:
#
# --mbs_default: Model-based subsampling based on the subsampling in Sun et al., (2019)
# --mbs_freq: Model-based subsampling based on the frequency-based subsampling in Kamigaito et al., (2022)
# --mbs_uniq: Model-based subsampling based on the unique-based subsampling in Kamigaito et al., (2022)

# SUB_TYPE="--mbs_default"
SUB_TYPE="--mbs_freq" 
# SUB_TYPE="--mbs_uniq"

# Suffix of the model directory:
# SUFFIX="mbs_default_${MR}_${SUB_MODEL_NAME}"
SUFFIX="mbs_freq_${MR}_${SUB_MODEL_NAME}"
# SUFFIX="mbs_uniq_${MR}_${SUB_MODEL_NAME}"


# Additional test if the training log files already exsit to avoid overwriting
# Choose a dataset
# DATASET='FB15k-237'
DATASET='wn18rr'
# DATASET='YAGO3-10'
# DATASET='FB15k'
# DATASET='wn18'
#
# Select out the modes you want to train this time
files=(
"./models/HAKE_${DATASET}_${SUFFIX}" 
"./models/RotatE_${DATASET}_${SUFFIX}" 
"./models/ComplEx_${DATASET}_${SUFFIX}" 
"./models/TransE_${DATASET}_${SUFFIX}" 
"./models/DistMult_${DATASET}_${SUFFIX}" 
# "./models/pRotatE_${DATASET}_${SUFFIX}" 
)
for filename in "${files[@]}"
do
  if [ -d "$filename" ]; then
    echo "$filename already exists."
  elif [ ! -d "$filename" ]; then
    echo "let's do it!"
    # # Configuration for HAKE
    #  
    # nohup bash run.sh train HAKE FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 "--modulus_weight 3.5" "--phase_weight 1.0" ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # nohup bash run.sh train HAKE wn18rr ${GPUID} ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 "--modulus_weight 0.5" "--phase_weight 0.5" ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} & 
    # nohup bash run.sh train HAKE YAGO3-10 ${GPUID} ${SUFFIX} 1024 256 500 24.0 1.0 0.0002 180000 4 "--modulus_weight 1.0" "--phase_weight 0.5" ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # #
    # # Configuration for RotatE
    # 
    # nohup bash run.sh train RotatE FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 -de ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # nohup bash run.sh train RotatE wn18rr ${GPUID} ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 -de ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} & 
    # nohup bash run.sh train RotatE YAGO3-10 ${GPUID} ${SUFFIX} 1024 400 500 24.0 1.0 0.0002 100000 4 -de ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # nohup bash run.sh train RotatE FB15k ${GPUID} ${SUFFIX} 1024 256 1000 24.0 1.0 0.0001 150000 16 -de ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # nohup bash run.sh train RotatE wn18 ${GPUID} ${SUFFIX} 512 1024 500 12.0 0.5 0.0001 80000 8 -de ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # #
    # # Configuration for ComplEx
    # 
    # nohup bash run.sh train ComplEx FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} & 
    # nohup bash run.sh train ComplEx wn18rr ${GPUID} ${SUFFIX} 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} & 
    # nohup bash run.sh train ComplEx FB15k ${GPUID} ${SUFFIX} 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # nohup bash run.sh train ComplEx wn18 ${GPUID} ${SUFFIX} 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # #
    # Configuration for TransE
    # 
    # nohup bash run.sh train TransE FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # nohup bash run.sh train TransE wn18rr ${GPUID} ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} & 
    # nohup bash run.sh train TransE FB15k ${GPUID} ${SUFFIX} 1024 256 1000 24.0 1.0 0.0001 150000 16 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} & 
    # nohup bash run.sh train TransE wn18 ${GPUID} ${SUFFIX} 512 1024 500 12.0 0.5 0.0001 80000 8 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # #
    # # Configuration for DistMult
    # 
    # nohup bash run.sh train DistMult FB15k-237 ${GPUID} ${SUFFIX} 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} & 
    # nohup bash run.sh train DistMult wn18rr ${GPUID} ${SUFFIX} 512 1024 1000 200.0 1.0 0.002 80000 8 -r 0.000005 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} & 
    # nohup bash run.sh train DistMult FB15k ${GPUID} ${SUFFIX} 1024 256 2000 500.0 1.0 0.001 150000 16 -r 0.000002 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} & 
    # nohup bash run.sh train DistMult wn18 ${GPUID} ${SUFFIX} 512 1024 1000 200.0 1.0 0.001 80000 8 -r 0.00001 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    #
    # # Configuration for pRotatE
    # 
    # nohup bash run.sh train pRotatE FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # nohup bash run.sh train pRotatE wn18rr ${GPUID} ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # nohup bash run.sh train pRotatE FB15k ${GPUID} ${SUFFIX} 1024 256 1000 24.0 1.0 0.0001 150000 16 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    # nohup bash run.sh train pRotatE wn18 ${GPUID} ${SUFFIX} 512 1024 500 12.0 0.5 0.0001 80000 8 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} &
    #
  fi
done

# check available GPUs
gpus=$(nvidia-smi -L | wc -l)
echo "Found $gpus available GPUs"