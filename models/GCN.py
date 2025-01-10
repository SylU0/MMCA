import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool


# 定义全连接边索引
def create_full_edge_index(num_nodes):
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


def prepare_data(x):
    B, L, C = x.shape
    edge_index = create_full_edge_index(L)  # 形状 [2, E]

    # 转换为 Data 对象
    data_list = []
    for i in range(B):
        node_features = x[i]  # [L, C]
        data = Data(x=node_features, edge_index=edge_index)
        data_list.append(data)

    # 批处理
    batch = Batch.from_data_list(data_list)

    return batch


# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = nn.Linear(out_channels, num_classes)
        self.act = nn.GELU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        # 图级池化
        x = global_mean_pool(x, batch)  # [B, out_channels]
        # 分类器
        x = self.fc(x)  # [B, num_classes]
        return x


def test():
    B, L, C = 2, 9, 1920
    x = torch.randn(B, L, C)
    model = GCN(in_channels=C, hidden_channels=C, out_channels=C, num_classes=200)

    # 准备图数据
    batch = prepare_data(x)

    # 前向传播
    output = model(batch)
    print(output.shape)


if __name__ == "__main__":
    test()
