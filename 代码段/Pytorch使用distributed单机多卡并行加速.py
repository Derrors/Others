# 使用 torch.distributed 加速并行训练

import torch

# 1) 初始化
torch.distributed.init_process_group(backend="nccl")

# 2） 配置每个进程的 GPU
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# 3）使用 DistributedSampler
train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

# 4) 封装之前要把模型移到对应的 GPU
model = ...
model.to(device)

# 5) 封装
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (x, y) in enumerate(train_loader):
      # 6) 将数据移到 GPU上
      x = x.to(device)
      y = y.to(device)
      ...
      output = model(images)
      loss = criterion(output, y)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

# 7) 需要通过命令行启动
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py