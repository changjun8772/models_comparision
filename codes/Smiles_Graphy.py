import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings('ignore')


class _MolecularGNNEncoder(nn.Module):
    """
    分子图编码器
    """

    def __init__(self, num_node_features, hidden_dim=128, embedding_dim=256):
        super(_MolecularGNNEncoder, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, batch):
        # 消息传递和特征提取
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)

        # 生成分子级别的固定长度嵌入
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding


class Smiles_Graphy:

    def __init__(self,
                 smiles_list,  # SMILES列表
                 embedding_dim=256,  # 生成的特征向量的维度
                 default_features=['AtomicNum', 'Degree', 'FormalCharge', 'Hybridization', 'IsAromatic', 'NumHs',
                                   'NumRadicalElectrons', 'IsInRing']  # 要加入的特征
                 ):
        """
        SMILES转分子图特征类
        @param smiles_list: 有效的smiles列表，误必保证所有的smiles有效
        @param embedding_dim: 生成的特征的维度
        """
        self.smiles_list = smiles_list
        self.embedding_dim = embedding_dim
        self.feature: pd.DataFrame = None
        self.__default_features = default_features

    def __smiles_to_graph(self, smiles):
        """
        将SMILES字符串转换为PyG图数据对象
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"无法解析SMILES: {smiles}")
            return None

        try:
            # 获取原子特征
            atom_features = []
            for atom in mol.GetAtoms():
                features = []
                if 'AtomicNum' in self.__default_features:
                    features.append(float(atom.GetAtomicNum()))  # 原子序数
                if 'Degree' in self.__default_features:
                    features.append(float(atom.GetDegree()))  # 度（连接数）
                if 'FormalCharge' in self.__default_features:
                    features.append(float(atom.GetFormalCharge()))  # 形式电荷
                if 'Hybridization' in self.__default_features:
                    features.append(float(atom.GetHybridization().real))  # 杂化方式
                if 'IsAromatic' in self.__default_features:
                    features.append(float(atom.GetIsAromatic()))  # 是否芳香
                if 'NumHs' in self.__default_features:
                    features.append(float(atom.GetTotalNumHs()))  # 连接的H原子数
                if 'NumRadicalElectrons' in self.__default_features:
                    features.append(float(atom.GetNumRadicalElectrons()))  # 自由基电子数
                if 'IsInRing' in self.__default_features:
                    features.append(float(atom.IsInRing()))  # 是否在环

                atom_features.append(features)

            # 获取键信息构建边
            edge_index = []

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                # 添加双向边（无向图）
                edge_index.append([i, j])
                edge_index.append([j, i])

            if len(edge_index) == 0:
                # 处理单原子分子
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            x = torch.tensor(atom_features, dtype=torch.float)

            return Data(x=x, edge_index=edge_index)
        except Exception as e:
            print(f"处理SMILES {smiles} 时出错: {e}")
            return None

    def __initialize_pretrained_encoder(self, num_node_features=8, embedding_dim=256):
        encoder = _MolecularGNNEncoder(
            num_node_features=num_node_features,
            hidden_dim=128,
            embedding_dim=embedding_dim
        )

        # 加载预训练权重
        # encoder.load_state_dict(torch.load('pretrained_encoder.pth'))

        return encoder

    def __extract_embeddings_from_smiles_list(self, smiles_list, embedding_dim=256):
        """
        从SMILES列表提取图嵌入特征
        """
        # 转换SMILES为图数据
        graphs = []
        valid_smiles = []
        failed_smiles = []

        for smiles in tqdm(smiles_list):
            graph_data = self.__smiles_to_graph(smiles)
            if graph_data is not None:
                graphs.append(graph_data)
                valid_smiles.append(smiles)
            else:
                failed_smiles.append(smiles)

        if failed_smiles:
            print(f"失败的SMILES: {failed_smiles}，请检查Smiles")

        if len(graphs) == 0:
            raise ValueError("没有成功转换任何分子！")

        # 初始化编码器
        num_node_features = graphs[0].x.shape[1]  # 从数据中自动获取特征维度
        encoder = self.__initialize_pretrained_encoder(num_node_features, embedding_dim)
        encoder.eval()

        # 提取嵌入特征
        all_embeddings = []

        with torch.no_grad():
            for graph in graphs:
                # 为单个分子创建batch
                graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
                embedding = encoder(graph.x, graph.edge_index, graph.batch)
                all_embeddings.append(embedding.cpu().numpy())

        # 转换为numpy数组
        embeddings_array = np.vstack(all_embeddings)
        return embeddings_array, valid_smiles, failed_smiles

    def __create_feature_dataframe(self, embeddings, smiles_list, feature_names=None):
        """
        创建包含特征向量的DataFrame
        """
        if feature_names is None:
            feature_names = [f'mol_feature_{i}' for i in range(embeddings.shape[1])]

        feature_df = pd.DataFrame(embeddings, columns=feature_names)
        feature_df['smiles'] = smiles_list
        feature_df.index.name = 'molecule_id'

        return feature_df

    def Get_Features(self) -> pd.DataFrame:
        # 提取特征
        embeddings, valid_smiles, failed_smiles = self.__extract_embeddings_from_smiles_list(
            self.smiles_list,
            embedding_dim=self.embedding_dim
        )
        feature_df = self.__create_feature_dataframe(embeddings, valid_smiles)
        self.feature = feature_df
        return feature_df

# 如何使用
# example_smiles = [
#     'CCO',  # 乙醇
#     'C1=CC=CC=C1',  # 苯
#     'CC(=O)O',  # 乙酸
#     'CCOC',  # 乙醚
#     'CCN(CC)CC',  # 三乙胺
#     'C1CCCCC1',  # 环己烷
#     'C1=CC=C(C=C1)O',  # 苯酚
#     'CC(C)C',  # 异丁烷
#     'C1=CN=C(N=C1)N',  # 鸟嘌呤
#     'CCOCC',  # 乙二醇二甲醚
# ]

# graphy = Smiles_Graphy(example_smiles)
# graphy.Get_Features()
