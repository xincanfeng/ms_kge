import torch
import os
import model

# 步骤 2: 初始化模型
model = model.KGEModel(model_name='HAKE', nentity='你的实体数', nrelation='你的关系数', hidden_dim='你的隐藏维度数', gamma='你的gamma值')

# 注意: 请根据你的模型参数和训练时的设置填写以上参数

# 步骤 3: 加载检查点
checkpoint = torch.load(os.path.join(model_path, 'checkpoint'))
model.load_state_dict(checkpoint['model_state_dict'])

# 如果你在检查点中还保存了优化器状态，你也可以这样加载它：
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 步骤 4: 将模型置于评估模式
model.eval()

# 此时，你可以使用模型进行计算，例如：
# with torch.no_grad():
#     output = model(input_data)
# 其中input_data是你要传递给模型的输入数据
