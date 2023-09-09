import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 创建一个新的figure对象，不包含坐标轴和其他元素
fig = plt.figure(figsize=(96, 2))

# 创建两个patch对象，代表你的两类标签
red_patch = mpatches.Patch(color='red', label='Count')
blue_patch = mpatches.Patch(color='blue', label='Model')

# 添加这些patch对象到图例中
leg = fig.legend(handles=[red_patch, blue_patch], 
                 fontsize=80, 
                 loc='center', 
                 ncol=2, 
                 frameon=False)

# 保存图像
fig.savefig("graphs/legend.png")
