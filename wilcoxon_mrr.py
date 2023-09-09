import numpy as np
from scipy import stats

# 两组配对或相关的数据样本
data = [
29.2	,	29.2	,	29.2	,
27.9	,	27.8	,	27.9	,
29.1	,	29.1	,	29.1	,
        ]

CBS = np.array([data[0], data[1], data[2]])
MBS = np.array([data[3], data[4], data[5]])
MIX = np.array([data[6], data[7], data[8]])

# 进行Wilcoxon符号秩检验
result_mbs = stats.wilcoxon(CBS, MBS)
result_mix = stats.wilcoxon(CBS, MIX)

# 输出检验的结果
print(f"MBS Statistics: {result_mbs.statistic}, p-value: {result_mbs.pvalue}")
print(f"MIX Statistics: {result_mix.statistic}, p-value: {result_mix.pvalue}")

# 根据p值做出决定
print("---------MBS results---------")
if result_mbs.pvalue < 0.1:
    print("MBS: Reject the null hypothesis")
else:
    print("MBS: Fail to reject the null hypothesis")

print("---------MIX results---------")
if result_mbs.pvalue < 0.1:
    print("MIX: Reject the null hypothesis")
else:
    print("MIX: Fail to reject the null hypothesis")
