#!/bin/bash
declare -g memory_required
declare -g GPUID
declare -g gpu
declare -g gpu_capacity
declare -g process_count

gpu=0
gpu_capacity=3
process_count=0

# manual_seed, default=1
# SEED='-seed=43' 
# SEED='-seed=12345'  
# SEED='-seed=67890' 

# Sub-model options:

# SUB_MODEL_NAME="ComplEx_FB15k-237_none:$SEED" #* 
# SUB_MODEL_NAME="ComplEx_FB15k-237_cnt_default:$SEED" #*
# SUB_MODEL_NAME="ComplEx_wn18rr_none:$SEED" #* 
# SUB_MODEL_NAME="ComplEx_wn18rr_cnt_default:$SEED" #* 

# SUB_MODEL_NAME="RotatE_FB15k-237_none:$SEED" 
# SUB_MODEL_NAME="RotatE_FB15k-237_cnt_default:$SEED" #*
# SUB_MODEL_NAME="RotatE_wn18rr_none:$SEED" 
# SUB_MODEL_NAME="RotatE_wn18rr_cnt_default:$SEED"
# SUB_MODEL_NAME="RotatE_YAGO3-10_none:$SEED" #*
# SUB_MODEL_NAME="RotatE_YAGO3-10_cnt_default:$SEED" #*

# SUB_MODEL_NAME="HAKE_FB15k-237_none:$SEED"
# SUB_MODEL_NAME="HAKE_FB15k-237_--cnt_default:$SEED"
# SUB_MODEL_NAME="HAKE_wn18rr_none:$SEED"
# SUB_MODEL_NAME="HAKE_wn18rr_cnt_default:$SEED"
# SUB_MODEL_NAME="HAKE_YAGO3-10_none:$SEED" #*
# SUB_MODEL_NAME="HAKE_YAGO3-10_cnt_default:$SEED" #*
                                                                          
# SUB_MODEL_NAME="TransE_FB15k-237_none:$SEED" #*
# SUB_MODEL_NAME="TransE_FB15k-237_cnt_default:$SEED" #*
# SUB_MODEL_NAME="TransE_wn18rr_none:$SEED"
# SUB_MODEL_NAME="TransE_wn18rr_cnt_default:$SEED" 

# SUB_MODEL_NAME="DistMult_FB15k-237_none:$SEED" 
# SUB_MODEL_NAME="DistMult_FB15k-237_cnt_default:$SEED" #* 
# SUB_MODEL_NAME="DistMult_wn18rr_none:$SEED" #* 
# SUB_MODEL_NAME="DistMult_wn18rr_cnt_default:$SEED"

# Directory path to a subsampling model
# SUB_MODEL="--subsampling_model ./models/${SUB_MODEL_NAME}"


# Temparature for subsampling, default=0.5
#
# SUB_TEMP="-stp=0.5" # default
# SUB_TEMP="-stp=2" 
# SUB_TEMP="-stp=1" 
# SUB_TEMP="-stp=0.1" 
# SUB_TEMP="-stp=0.05" 
# SUB_TEMP="-stp=0.01" 
#
# Negative tempretures are not selected at all
#
# SUB_TEMP="-stp=-0.01"
# SUB_TEMP="-stp=-0.05" 
# SUB_TEMP="-stp=-0.1" 
# SUB_TEMP="-stp=-0.5" # 
# SUB_TEMP="-stp=-1" #
# SUB_TEMP="-stp=-2"

# Ratio of the model-based subsampling:
# MR="1.0"
# MR="0.9"
# MR="0.7"
# MR="0.5"
# MR="0.3"
# MR="0.1"
# MBS_RATIO="--mbs_ratio ${MR}"

# Type of model-based subsampling
# You can choose the following methods:
#
# --mbs_default: Model-based subsampling based on the subsampling in Sun et al., (2019)
# --mbs_freq: Model-based subsampling based on the frequency-based subsampling in Kamigaito et al., (2022)
# --mbs_uniq: Model-based subsampling based on the unique-based subsampling in Kamigaito et al., (2022)


# SUB_TYPE="--cnt_default"
# SUB_TYPE="--cnt_freq" 
# SUB_TYPE="--cnt_uniq" 

# SUB_TYPE="--mbs_default"
# SUB_TYPE="--mbs_freq" 
# SUB_TYPE="--mbs_uniq" 

# Dump freq or not
# DUMP="--dump_freqs"
# DUMP="" 

# Dump score or not
DUMP_SCORE="--dump_scores"

# Suffix of the model directory: 
# SUFFIX="${SUB_TYPE}_${MR}_${SUB_MODEL_NAME}:${SUB_TEMP}:$SEED"
SUFFIX="--mbs_uniq_0.3_RotatE_YAGO3-10_none:-seed=43:-stp=0.5:-seed=43"

# Define required memory
memory_required=10000
# memory_required=11264
# memory_required=13653 # for YAGO3-10 and HAKE 

find_available_gpu() {
# Check available GPUs
GPUIDs=$(nvidia-smi -L | awk '{print $2}' | tr -d :)

# Find a GPU with enough memory
for GPUID in $GPUIDs
do
  memory_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPUID})
  if [ $memory_free -ge $memory_required ]; then 
    # echo "Found available GPU ${GPUID} with more than ${memory_required}MiB free memory." 
    echo "${GPUID}"
    break
  fi
done
}


# Define the experiments to be run
exps=(
    # # Configuration for RotatE
    # 
    # "bash run.sh train RotatE FB15k-237 %s ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 -de ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &" 
    # "bash run.sh train RotatE wn18rr %s ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 -de ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &" 
    # "bash run.sh train RotatE YAGO3-10 %s ${SUFFIX} 1024 400 500 24.0 1.0 0.0002 100000 4 -de ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &"
    # "bash run.sh train RotatE FB15k %s ${SUFFIX} 1024 256 1000 24.0 1.0 0.0001 150000 16 -de ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &"
    # "bash run.sh train RotatE wn18 %s ${SUFFIX} 512 1024 500 12.0 0.5 0.0001 80000 8 -de ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &"
    #
    # # Configuration for TransE
    # 
    # "bash run.sh train TransE FB15k-237 %s ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &"
    # "bash run.sh train TransE wn18rr %s ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &"
    # "bash run.sh train TransE FB15k %s ${SUFFIX} 1024 256 1000 24.0 1.0 0.0001 150000 16 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &" 
    # "bash run.sh train TransE wn18 %s ${SUFFIX} 512 1024 500 12.0 0.5 0.0001 80000 8 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &"
    #
    # # Configuration for HAKE
    #  
    # "bash run.sh train HAKE FB15k-237 %s ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 --modulus_weight 3.5 --phase_weight 1.0 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &"
    # "bash run.sh train HAKE wn18rr %s ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 --modulus_weight 0.5 --phase_weight 0.5 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &" 
    "bash run.sh train HAKE YAGO3-10 %s ${SUFFIX} 1024 256 500 24.0 1.0 0.0002 180000 4 --modulus_weight 1.0 --phase_weight 0.5 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &"
    # 
    # # Configuration for ComplEx
    # 
    # "bash run.sh train ComplEx FB15k-237 %s ${SUFFIX} 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &"
    # "bash run.sh train ComplEx wn18rr %s ${SUFFIX} 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &" 
    # "bash run.sh train ComplEx FB15k %s ${SUFFIX} 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &"
    # "bash run.sh train ComplEx wn18 %s ${SUFFIX} 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &"
    # 
    # # Configuration for DistMult
    # 
    # "bash run.sh train DistMult FB15k-237 %s ${SUFFIX} 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &" 
    # "bash run.sh train DistMult wn18rr %s ${SUFFIX} 512 1024 1000 200.0 1.0 0.002 80000 8 -r 0.000005 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} ${DUMP} ${DUMP_SCORE} &"
    # "bash run.sh train DistMult FB15k %s ${SUFFIX} 1024 256 2000 500.0 1.0 0.001 150000 16 -r 0.000002 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &" 
    # "bash run.sh train DistMult wn18 %s ${SUFFIX} 512 1024 1000 200.0 1.0 0.001 80000 8 -r 0.00001 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &"
    #
    # # Configuration for pRotatE
    # 
    # "bash run.sh train pRotatE FB15k-237 %s ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &"
    # "bash run.sh train pRotatE wn18rr %s ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &"
    # "bash run.sh train pRotatE FB15k %s ${SUFFIX} 1024 256 1000 24.0 1.0 0.0001 150000 16 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &"
    # "bash run.sh train pRotatE wn18 %s ${SUFFIX} 512 1024 500 12.0 0.5 0.0001 80000 8 ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO} ${SEED} &"
)


# GPUID=$(find_available_gpu)
# gpu=${GPUID}

# Loop over the files and run the experiments if it doesn't exist
for exp in "${exps[@]}"
do
  # calculate GPU capacity of the number of runnable experiments and check if we need to switch GPU
  # memory_left=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${gpu})
  # gpu_capacity=$(echo "scale=0; $memory_left/$memory_required" | bc)
  # process_count=$(nvidia-smi --id=${gpu} --query-compute-apps=pid --format=csv | wc -l)
  # process_count=$((process_count-1))
  if [ ${process_count} -lt ${gpu_capacity} ]; then
    gpu=${gpu}
  else
    gpu=$((gpu+8))
  fi
  exp=$(printf "${exp}" "${gpu}")
  # extract save directory from the experiment command
  model=$(echo "${exp}" | awk '{for (i=1; i<=NF; i++) if ($i=="train") print $(i+1)}')
  dataset=$(echo "${exp}" | awk '{for (i=1; i<=NF; i++) if ($i=="train") print $(i+2)}')
  save_dir="./models/${model}_${dataset}_${SUFFIX}" 
  if [ -d "$save_dir" ]; then
    echo "$save_dir already exists."
    echo "${exp}" # 暂时添加这三行
    eval "${exp}" # 暂时添加这三行
    process_count=$((process_count+1)) # 暂时添加这三行
  else
    # echo "Running experiment for $save_dir on GPU ${gpu}..."
    echo "${exp}"
    eval "${exp}"
    # sleep 15s
    process_count=$((process_count+1))
  fi
done

echo "process_count: $process_count"

# sleep 20s
# nvidia-smi




# Test if the training log files already exsit to avoid overwriting
# #
# # Choose a dataset
# DATASET='FB15k-237'
# DATASET='wn18rr'
# DATASET='YAGO3-10'
# DATASET='FB15k'
# DATASET='wn18'
# #
# # Tell the server which models you want to run 
# files=(
# "./models/HAKE_${DATASET}_${SUFFIX}" 
# "./models/RotatE_${DATASET}_${SUFFIX}" 
# "./models/ComplEx_${DATASET}_${SUFFIX}" 
# "./models/TransE_${DATASET}_${SUFFIX}" 
# "./models/DistMult_${DATASET}_${SUFFIX}" 
# "./models/pRotatE_${DATASET}_${SUFFIX}" 
# )

# # Loop over the files and see if it already exists
# for file in "${files[@]}"
# do
#   if [ -d "$file" ]; then
#     echo "$file already exists."
#   else
#     echo "Let's do it!"
#   fi
# done