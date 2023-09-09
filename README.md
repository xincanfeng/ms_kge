
# Model-based Subsampling for Knowledge Graph Embedding

**Introduction**

This is a modified version of [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)  to run model-based subsampling.
This implementation supports [Automatic Mixed Precision (AMP)](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) to reduce training time.

**Implemented features**

Models:
 - [x] RotatE
 - [x] pRotatE
 - [x] TransE
 - [x] ComplEx
 - [x] DistMult

Evaluation Metrics:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)
 - [x] AUC-PR (for Countries data sets)

Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling

**Usage**

Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

**Train**

To train a model with the model-based subsampling, you need to train another model used for subsampling.
`init_config.sh` lists the configurations to train models used for model-based subsampling.
You can use these settings to train your model.
The following code is a part of the script.
```
# GPU id
GPUID=0
# Suffix of the model directory
SUFFIX="none"

# Best Configuration for HAKE
#
bash run.sh train HAKE wn18rr ${GPUID} ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 "--modulus_weight 0.5" "--phase_weight 0.5"
...
```
After that, you can use model-based subsampling based on the trained model.
You can see example settings in `config.sh`.
The following code is an example for running model-based subsampling based on the unique-based subsampling in Kamigaito et al., (2022).
```
# Directory path to a subsampling model
SUB_MODEL="--subsampling_model ./models/[SET_YOUR_MODEL_NAME_HERE]"

# Temparature for subsampling
SUB_TEMP="--subsampling_model_temperature 0.5"

# Type of model-based subsampling
#
# You can choose the following methods:
#
# --mbs_default: Model-based subsampling based on the subsampling in Sun et al., (2019)
# --mbs_freq: Model-based subsampling based on the frequency-based subsampling in Kamigaito et al., (2022)
# --mbs_uniq: Model-based subsampling based on the unique-based subsampling in Kamigaito et al., (2022)
#

#SUB_TYPE="--mbs_default"
#SUB_TYPE="--mbs_freq"
SUB_TYPE="--mbs_uniq"

# Ratio of the model-based subsampling
MR="1.0"
MBS_RATIO="--mbs_ratio ${MR}"

# Suffix of the model directory

#SUFFIX="mbs_default_${MR}"
#SUFFIX="mbs_freq_${MR}"
SUFFIX="mbs_uniq_${MR}"

# GPU id
GPUID=0

# Configuration for HAKE
#
bash run.sh train HAKE wn18rr ${GPUID} ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 "--modulus_weight 0.5" "--phase_weight 0.5" ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE} ${MBS_RATIO}
...
```

In addition to the above settings, `base_config.sh` lists the configurations for training baseline models.
The following code is a part of the script.
```
# Type of count-based subsampling
#
# You can choose the following methods:
#
# --cnt_default: Subsampling in Sun et al., (2019)
# --cnt_freq: Frequency-based subsampling in Kamigaito et al., (2022)
# --cnt_uniq: Unique-based subsampling in Kamigaito et al., (2022)

SUB_TYPE="--cnt_default"
#SUB_TYPE="--cnt_freq"
#SUB_TYPE="--cnt_uniq"

# Suffix of the model directory

SUFFIX="cnt_default"
#SUFFIX="cnt_freq"
#SUFFIX="cnt_uniq"

# GPU id
GPUID=0

# Best Configuration for HAKE
#
bash run.sh train HAKE wn18rr ${GPUID} ${SUFFIX} 512 1024 500 6.0 0.5 0.00005 80000 8 "--modulus_weight 0.5" "--phase_weight 0.5" ${SUB_TYPE}
...
```


   Check argparse configuration at codes/run.py for more arguments and more details.

After the training, you can see the evaluation scores based on the best validation model at the last part of `./models/[YOUR_MODEL_NAME]/train.log`.

**Test**

You can run test by using a specific checkpoint as follows:
```
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python -u $CODE_PATH/run.py --do_test --cuda -init ${CHECK_POINT_FILE}
```
