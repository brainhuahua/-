import torch

print(torch.cuda.is_available())  # True 表示可用
print(torch.cuda.device_count())  # 返回可用的 GPU 数量
print(torch.cuda.get_device_name(0))  # 第一个 GPU 的名称
