# Node Classification
## Torch Geometric 依赖安装

### 查看一下本地的torch和cuda的版本
```python
import torch
print(torch.__version__)
print(torch.version.cuda)
```

### 找到版本对应的依赖下载

[链接地址](https://pytorch-geometric.com/whl/)

需要下载的依赖包如下：
- torch_geometric
- torch_sparse
- torch_scatter
- ~~torch_spline_conv~~
- ~~torch_cluster~~

### 安装依赖包

将下载好的 `whl` 文件进行安装。