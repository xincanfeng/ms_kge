import numpy as np
from scipy import stats
import pandas as pd

# To run the significance test, we need to consider the average scores for each query, which is dumped into a csv file.
# 两组配对或相关的数据样本

# 这里列出你的文件列表，例如：
file_list = [
            'models/RotatE_FB15k-237_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/RotatE_FB15k-237_--mbs_default_1.0_ComplEx_FB15k-237_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/RotatE_FB15k-237_--mbs_default_0.9_ComplEx_FB15k-237_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/RotatE_FB15k-237_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/RotatE_FB15k-237_--mbs_freq_1.0_ComplEx_FB15k-237_none:-seed=43:-stp=0.1:-seed=43/file_dump_score.tsv',
            'models/RotatE_FB15k-237_--mbs_freq_0.7_ComplEx_FB15k-237_none:-seed=43:-stp=0.1:-seed=43/file_dump_score.tsv',
            'models/RotatE_FB15k-237_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/RotatE_FB15k-237_--mbs_uniq_1.0_ComplEx_FB15k-237_none:-seed=43:-stp=0.1:-seed=43/file_dump_score.tsv',
            'models/RotatE_FB15k-237_--mbs_uniq_0.5_ComplEx_FB15k-237_none:-seed=43:-stp=0.1:-seed=43/file_dump_score.tsv',

            'models/TransE_FB15k-237_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/TransE_FB15k-237_--mbs_default_1.0_ComplEx_FB15k-237_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/TransE_FB15k-237_--mbs_default_0.9_ComplEx_FB15k-237_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/TransE_FB15k-237_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/TransE_FB15k-237_--mbs_freq_1.0_RotatE_FB15k-237_cnt_default:-seed=43:-stp=0.1:-seed=43/file_dump_score.tsv',
            'models/TransE_FB15k-237_--mbs_freq_0.7_RotatE_FB15k-237_cnt_default:-seed=43:-stp=0.1:-seed=43/file_dump_score.tsv',
            'models/TransE_FB15k-237_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/TransE_FB15k-237_--mbs_uniq_1.0_RotatE_FB15k-237_cnt_default:-seed=43:-stp=0.1:-seed=43/file_dump_score.tsv',
            'models/TransE_FB15k-237_--mbs_uniq_0.7_RotatE_FB15k-237_cnt_default:-seed=43:-stp=0.1:-seed=43/file_dump_score.tsv',

            'models/HAKE_FB15k-237_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/HAKE_FB15k-237_--mbs_default_1.0_ComplEx_FB15k-237_none:-seed=43:-stp=1:-seed=43/file_dump_score.tsv',
            'models/HAKE_FB15k-237_--mbs_default_0.5_ComplEx_FB15k-237_none:-seed=43:-stp=1:-seed=43/file_dump_score.tsv',
            'models/HAKE_FB15k-237_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/HAKE_FB15k-237_--mbs_freq_1.0_ComplEx_FB15k-237_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/HAKE_FB15k-237_--mbs_freq_0.5_ComplEx_FB15k-237_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/HAKE_FB15k-237_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/HAKE_FB15k-237_--mbs_uniq_1.0_RotatE_FB15k-237_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/HAKE_FB15k-237_--mbs_uniq_0.3_RotatE_FB15k-237_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',

            'models/ComplEx_FB15k-237_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/ComplEx_FB15k-237_--mbs_default_1.0_ComplEx_FB15k-237_none:-stp=1:-seed=43/file_dump_score.tsv',
            'models/ComplEx_FB15k-237_--mbs_default_0.5_ComplEx_FB15k-237_none:-seed=43:-stp=1:-seed=43/file_dump_score.tsv',
            'models/ComplEx_FB15k-237_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/ComplEx_FB15k-237_--mbs_freq_1.0_DistMult_FB15k-237_cnt_default:-seed=12345:-stp=0.5:-seed=12345/file_dump_score.tsv',
            'models/ComplEx_FB15k-237_--mbs_freq_0.1_DistMult_FB15k-237_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/ComplEx_FB15k-237_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/ComplEx_FB15k-237_--mbs_uniq_1.0_ComplEx_FB15k-237_cnt_default:-seed=12345:-stp=0.5:-seed=12345/file_dump_score.tsv',
            'models/ComplEx_FB15k-237_--mbs_uniq_0.1_ComplEx_FB15k-237_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',

            'models/DistMult_FB15k-237_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/DistMult_FB15k-237_--mbs_default_1.0_ComplEx_FB15k-237_none:-seed=43:-stp=1:-seed=43/file_dump_score.tsv',
            'models/DistMult_FB15k-237_--mbs_default_0.7_ComplEx_FB15k-237_none:-seed=12345:-stp=1:-seed=12345/file_dump_score.tsv',
            'models/DistMult_FB15k-237_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/DistMult_FB15k-237_--mbs_freq_1.0_DistMult_FB15k-237_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/DistMult_FB15k-237_--mbs_freq_0.1_DistMult_FB15k-237_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/DistMult_FB15k-237_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/DistMult_FB15k-237_--mbs_uniq_1.0_ComplEx_FB15k-237_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/DistMult_FB15k-237_--mbs_uniq_0.1_ComplEx_FB15k-237_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',

            'models/RotatE_wn18rr_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/RotatE_wn18rr_--mbs_default_1.0_ComplEx_wn18rr_none:-seed=43:-stp=1:-seed=43/file_dump_score.tsv',
            'models/RotatE_wn18rr_--mbs_default_0.5_ComplEx_wn18rr_none:-seed=43:-stp=1:-seed=43/file_dump_score.tsv',
            'models/RotatE_wn18rr_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/RotatE_wn18rr_--mbs_freq_1.0_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/RotatE_wn18rr_--mbs_freq_0.3_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/RotatE_wn18rr_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/RotatE_wn18rr_--mbs_uniq_1.0_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/RotatE_wn18rr_--mbs_uniq_0.5_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',

            'models/TransE_wn18rr_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/TransE_wn18rr_--mbs_default_1.0_ComplEx_wn18rr_cnt_default:-seed=43:-stp=2:-seed=43/file_dump_score.tsv',
            'models/TransE_wn18rr_--mbs_default_0.9_ComplEx_wn18rr_cnt_default:-seed=43:-stp=2:-seed=43/file_dump_score.tsv',
            'models/TransE_wn18rr_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/TransE_wn18rr_--mbs_freq_1.0_ComplEx_wn18rr_cnt_default:-seed=43:-stp=2:-seed=43/file_dump_score.tsv',
            'models/TransE_wn18rr_--mbs_freq_0.9_ComplEx_wn18rr_cnt_default:-seed=43:-stp=2:-seed=43/file_dump_score.tsv',
            'models/TransE_wn18rr_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/TransE_wn18rr_--mbs_uniq_1.0_ComplEx_wn18rr_cnt_default:-seed=43:-stp=1:-seed=43/file_dump_score.tsv',
            'models/TransE_wn18rr_--mbs_uniq_0.9_ComplEx_wn18rr_cnt_default:-seed=43:-stp=1:-seed=43/file_dump_score.tsv',

            'models/HAKE_wn18rr_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/HAKE_wn18rr_--mbs_default_1.0_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/HAKE_wn18rr_--mbs_default_0.1_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/HAKE_wn18rr_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/HAKE_wn18rr_--mbs_freq_1.0_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/HAKE_wn18rr_--mbs_freq_0.9_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/HAKE_wn18rr_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/HAKE_wn18rr_--mbs_uniq_1.0_DistMult_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/HAKE_wn18rr_--mbs_uniq_0.7_DistMult_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',

            'models/ComplEx_wn18rr_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/ComplEx_wn18rr_--mbs_default_1.0_ComplEx_wn18rr_none:-seed=43:-stp=2:-seed=43/file_dump_score.tsv',
            'models/ComplEx_wn18rr_--mbs_default_0.7_ComplEx_wn18rr_none:-seed=43:-stp=2:-seed=43/file_dump_score.tsv',
            'models/ComplEx_wn18rr_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/ComplEx_wn18rr_--mbs_freq_1.0_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/ComplEx_wn18rr_--mbs_freq_0.9_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/ComplEx_wn18rr_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/ComplEx_wn18rr_--mbs_uniq_1.0_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/ComplEx_wn18rr_--mbs_uniq_0.9_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',

            'models/DistMult_wn18rr_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/DistMult_wn18rr_--mbs_default_1.0_ComplEx_wn18rr_none:-seed=43:-stp=2:-seed=43/file_dump_score.tsv',
            'models/DistMult_wn18rr_--mbs_default_0.7_ComplEx_wn18rr_none:-seed=43:-stp=2:-seed=43/file_dump_score.tsv',
            'models/DistMult_wn18rr_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/DistMult_wn18rr_--mbs_freq_1.0_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/DistMult_wn18rr_--mbs_freq_0.9_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/DistMult_wn18rr_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/DistMult_wn18rr_--mbs_uniq_1.0_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/DistMult_wn18rr_--mbs_uniq_0.9_ComplEx_wn18rr_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',

            'models/RotatE_YAGO3-10_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/RotatE_YAGO3-10_--mbs_default_1.0_RotatE_YAGO3-10_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/RotatE_YAGO3-10_--mbs_default_0.7_RotatE_YAGO3-10_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/RotatE_YAGO3-10_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/RotatE_YAGO3-10_--mbs_freq_1.0_HAKE_YAGO3-10_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/RotatE_YAGO3-10_--mbs_freq_0.5_HAKE_YAGO3-10_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/RotatE_YAGO3-10_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/RotatE_YAGO3-10_--mbs_uniq_1.0_RotatE_YAGO3-10_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/RotatE_YAGO3-10_--mbs_uniq_0.5_RotatE_YAGO3-10_cnt_default:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',

            'models/HAKE_YAGO3-10_--cnt_default:-seed=43/file_dump_score.tsv',
            'models/HAKE_YAGO3-10_--mbs_default_1.0_HAKE_YAGO3-10_none:-seed=43:-stp=0.1:-seed=43/file_dump_score.tsv',
            'models/HAKE_YAGO3-10_--mbs_default_0.5_HAKE_YAGO3-10_none:-seed=43:-stp=0.1:-seed=43/file_dump_score.tsv',
            'models/HAKE_YAGO3-10_--cnt_freq:-seed=43/file_dump_score.tsv',
            'models/HAKE_YAGO3-10_--mbs_freq_1.0_RotatE_YAGO3-10_none/file_dump_score.tsv',
            'models/HAKE_YAGO3-10_--mbs_freq_0.1_RotatE_YAGO3-10_none:-seed=12345:-stp=0.5:-seed=12345/file_dump_score.tsv',
            'models/HAKE_YAGO3-10_--cnt_uniq:-seed=43/file_dump_score.tsv',
            'models/HAKE_YAGO3-10_--mbs_uniq_1.0_RotatE_YAGO3-10_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
            'models/HAKE_YAGO3-10_--mbs_uniq_0.3_RotatE_YAGO3-10_none:-seed=43:-stp=0.5:-seed=43/file_dump_score.tsv',
             ] 

def process_files(files):
    data_group = []
    for file in files:
        data = pd.read_csv(file, sep='\t')
        data_group.append(data.iloc[:, 2].to_numpy()) # 读取第三列数据
    return data_group

# 创建或打开一个txt文件以写入输出
with open('wilcoxon_test_output.txt', 'w') as f:
    for i in range(0, len(file_list), 3):
        files = file_list[i:i+3]
        
        if len(files) == 3:
            CBS, MBS, MIX = process_files(files)
            
            # 进行Wilcoxon符号秩检验
            result_mbs = stats.wilcoxon(CBS, MBS)
            result_mix = stats.wilcoxon(CBS, MIX)
            
            # 创建输出字符串
            output_str = f"Group {i//3+1}\n"
            output_str += f"MBS Statistics: {result_mbs.statistic}, p-value: {result_mbs.pvalue}\n"
            output_str += f"MIX Statistics: {result_mix.statistic}, p-value: {result_mix.pvalue}\n"
            
            output_str += "---------MBS results---------\n"
            output_str += "MBS: " + ("Reject the null hypothesis\n" if result_mbs.pvalue < 0.01 else "Fail to reject the null hypothesis\n")
            
            output_str += "---------MIX results---------\n"
            output_str += "MIX: " + ("Reject the null hypothesis\n" if result_mix.pvalue < 0.01 else "Fail to reject the null hypothesis\n")
            output_str += "\n"
            
            # 将输出字符串写入文件
            f.write(output_str)
