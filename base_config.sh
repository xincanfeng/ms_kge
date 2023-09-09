# Type of count-based subsampling
#
# You can choose the following methods:
#
# --cnt_default: Subsampling in Sun et al., (2019)
# --cnt_freq: Frequency-based subsampling in Kamigaito et al., (2022)
# --cnt_uniq: Unique-based subsampling in Kamigaito et al., (2022)

# GPU id
GPUID=3

# manual_seed, default=1
SEED='-seed=43' 
SEED='-seed=12345' 
SEED='-seed=67890' 

SUB_TYPE="--cnt_default"
# SUB_TYPE="--cnt_freq"
SUB_TYPE="--cnt_uniq"

# Suffix of the model directory
SUFFIX="${SUB_TYPE}:$SEED"

# Best Configuration for ComplEx
#
# nohup bash run.sh train ComplEx FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001 ${SUB_TYPE} $SEED &
# nohup bash run.sh train ComplEx wn18rr ${GPUID} ${SUFFIX} 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005 ${SUB_TYPE} $SEED &
# nohup bash run.sh train ComplEx FB15k ${GPUID} ${SUFFIX} 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002 ${SUB_TYPE} $SEED &
# nohup bash run.sh train ComplEx wn18 ${GPUID} ${SUFFIX} 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001 ${SUB_TYPE} $SEED &
#
# Best Configuration for DistMult
#
# nohup bash run.sh train DistMult FB15k-237 ${GPUID} ${SUFFIX} 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001 ${SUB_TYPE} $SEED &
# nohup bash run.sh train DistMult wn18rr ${GPUID} ${SUFFIX} 512 1024 1000 200.0 1.0 0.002 80000 8 -r 0.000005 ${SUB_TYPE} $SEED &
# nohup bash run.sh train DistMult FB15k ${GPUID} ${SUFFIX} 1024 256 2000 500.0 1.0 0.001 150000 16 -r 0.000002 ${SUB_TYPE} $SEED &
# nohup bash run.sh train DistMult wn18 ${GPUID} ${SUFFIX} 512 1024 1000 200.0 1.0 0.001 80000 8 -r 0.00001 ${SUB_TYPE} $SEED &
#
# Best Configuration for HAKE
#
# nohup bash run.sh train HAKE FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 "--modulus_weight 3.5" "--phase_weight 1.0" ${SUB_TYPE} $SEED &
# nohup bash run.sh train HAKE wn18rr ${GPUID} ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 "--modulus_weight 0.5" "--phase_weight 0.5" ${SUB_TYPE} $SEED &
# nohup bash run.sh train HAKE YAGO3-10 ${GPUID} ${SUFFIX} 1024 256 500 24.0 1.0 0.0002 180000 4 "--modulus_weight 1.0" "--phase_weight 0.5" ${SUB_TYPE} $SEED &
#
# Best Configuration for RotatE
#
# nohup bash run.sh train RotatE FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 -de ${SUB_TYPE} $SEED &
# nohup bash run.sh train RotatE wn18rr ${GPUID} ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 -de ${SUB_TYPE} $SEED &
nohup bash run.sh train RotatE YAGO3-10 ${GPUID} ${SUFFIX} 1024 400 500 24.0 1.0 0.0002 100000 4 -de ${SUB_TYPE} $SEED &
# nohup bash run.sh train RotatE FB15k ${GPUID} ${SUFFIX} 1024 256 1000 24.0 1.0 0.0001 150000 16 -de ${SUB_TYPE} $SEED &
# nohup bash run.sh train RotatE wn18 ${GPUID} ${SUFFIX} 512 1024 500 12.0 0.5 0.0001 80000 8 -de ${SUB_TYPE} $SEED &
#
# Best Configuration for TransE
#
# nohup bash run.sh train TransE FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 ${SUB_TYPE} $SEED &
# nohup bash run.sh train TransE wn18rr ${GPUID} ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 ${SUB_TYPE} $SEED &
# nohup bash run.sh train TransE FB15k ${GPUID} ${SUFFIX} 1024 256 1000 24.0 1.0 0.0001 150000 16 ${SUB_TYPE} $SEED &
# nohup bash run.sh train TransE wn18 ${GPUID} ${SUFFIX} 512 1024 500 12.0 0.5 0.0001 80000 8 ${SUB_TYPE} $SEED &
#
# Best Configuration for pRotatE
#
# nohup bash run.sh train pRotatE FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 ${SUB_TYPE} $SEED &
# nohup bash run.sh train pRotatE wn18rr ${GPUID} ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 ${SUB_TYPE} $SEED &
# nohup bash run.sh train pRotatE FB15k ${GPUID} ${SUFFIX} 1024 256 1000 24.0 1.0 0.0001 150000 16 ${SUB_TYPE} $SEED &
# nohup bash run.sh train pRotatE wn18 ${GPUID} ${SUFFIX} 512 1024 500 12.0 0.5 0.0001 80000 8 ${SUB_TYPE} $SEED &
#