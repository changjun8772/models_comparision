import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from collections import defaultdict
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.optim import Adam
from torch.utils.data import random_split
import os


# 原子特征
def atom_features(atom):
    """
    将RDKit原子对象转换为特征向量，使用指定的8个特征
    """
    features = []

    # 1. 原子序数 (AtomicNum)
    atomic_num = atom.GetAtomicNum()
    features.append(atomic_num)

    # 2. 度 (Degree) - 原子的连接数
    degree = atom.GetDegree()
    features.append(degree)

    # 3. 形式电荷 (FormalCharge)
    formal_charge = atom.GetFormalCharge()
    features.append(formal_charge)

    # 4. 杂化方式 (Hybridization)
    hybridization = atom.GetHybridization()

    # 将杂化方式转换为数值
    hybrid_map = {
        rdchem.HybridizationType.SP: 1,
        rdchem.HybridizationType.SP2: 2,
        rdchem.HybridizationType.SP3: 3,
        rdchem.HybridizationType.SP3D: 4,
        rdchem.HybridizationType.SP3D2: 5,
    }
    features.append(hybrid_map.get(hybridization, 0))

    # 5. 芳香性 (IsAromatic)
    is_aromatic = 1 if atom.GetIsAromatic() else 0
    features.append(is_aromatic)

    # 6. 氢原子数量 (NumHs)
    num_hs = atom.GetTotalNumHs()
    features.append(num_hs)

    # 7. 自由基电子数 (NumRadicalElectrons)
    num_radical_electrons = atom.GetNumRadicalElectrons()
    features.append(num_radical_electrons)

    # 8. 是否在环中 (IsInRing)
    is_in_ring = 1 if atom.IsInRing() else 0
    features.append(is_in_ring)

    return np.array(features, dtype=np.float32)


# 键特征
def bond_features(bond):
    """
    将RDKit键对象转换为特征向量
    """
    bt = bond.GetBondType()
    bond_features = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
    ]
    return np.array(bond_features, dtype=np.float32)


# SMILES到图数据转换
def smiles_to_graph(smiles):
    """
    将SMILES字符串转换为PyG图数据
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无法解析SMILES: {smiles}")
        return None

    # 添加氢原子
    mol = Chem.AddHs(mol)

    # 原子特征
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_features(atom))
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)

    # 边（键）
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # 添加两个方向
        edge_indices += [[i, j], [j, i]]

        # 边特征
        bond_feats = bond_features(bond)
        edge_attrs += [bond_feats, bond_feats]

    if len(edge_indices) == 0:
        edge_indices = torch.empty((2, 0), dtype=torch.long)
        edge_attrs = torch.empty((0, 4), dtype=torch.float)
    else:
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attrs = torch.tensor(np.array(edge_attrs), dtype=torch.float)

    return Data(x=x, edge_index=edge_indices, edge_attr=edge_attrs)


# 数据集类
class MolecularDataset(Dataset):
    def __init__(self, smiles_list, targets=None, transform=None, pre_transform=None):
        super(MolecularDataset, self).__init__(transform, pre_transform)

        if hasattr(smiles_list, 'tolist'):
            self.smiles_list = smiles_list.tolist()
        else:
            self.smiles_list = list(smiles_list)

        if targets is not None:
            if hasattr(targets, 'tolist'):
                self.targets = targets.tolist()
            else:
                self.targets = list(targets)
        else:
            self.targets = None

        self.data_list = []
        for i, smiles in enumerate(self.smiles_list):
            data = smiles_to_graph(smiles)
            if data is not None:
                if self.targets is not None:
                    target_value = self.targets[i]
                    if hasattr(target_value, 'item'):
                        target_value = target_value.item()
                    data.y = torch.tensor([target_value], dtype=torch.float)
                data.smiles = smiles
                self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# GAT模型
class GATModel(nn.Module):
    def __init__(self,
                 num_node_features,
                 num_edge_features=4,
                 hidden_dim=128,
                 num_heads=8,
                 num_layers=3,
                 dropout=0.2,
                 num_classes=1):
        super(GATModel, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 第一个GAT层
        self.conv1 = GATConv(
            num_node_features,
            hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=num_edge_features,
            dropout=dropout
        )

        # 中间GAT层
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=num_edge_features,
                    dropout=dropout
                )
            )

        # 最后一个GAT层
        self.conv_last = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            edge_dim=num_edge_features,
            dropout=dropout,
            concat=False
        )

        # 全局池化和分类层
        self.pool = global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.conv_layers:
            x = F.elu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.conv_last(x, edge_index, edge_attr))
        x = self.pool(x, batch)
        # 分类
        out = self.classifier(x)

        return out


# 训练函数
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []
    val_aucs = []

    best_val_auc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        batch_count = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)
            target = batch.y.view(-1, 1)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        train_loss /= batch_count
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_batch_count = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                out = model(batch)
                target = batch.y.view(-1, 1)
                loss = criterion(out, target)

                val_loss += loss.item()
                val_batch_count += 1
                all_preds.extend(out.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())

        val_loss /= val_batch_count
        val_losses.append(val_loss)

        # 计算AUC
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()

        if len(np.unique(all_targets)) > 1:
            val_auc = roc_auc_score(all_targets, all_preds)
        else:
            val_auc = 0.5  # 如果只有一类，AUC设为0.5

        val_aucs.append(val_auc)

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': val_aucs,
        'best_val_auc': best_val_auc
    }


# 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_smiles = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            out = model(batch)

            all_preds.extend(out.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
            all_smiles.extend(batch.smiles)

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    # 计算各种指标
    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_preds)
    else:
        auc = 0.5

    predictions_binary = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_targets, predictions_binary)
    f1 = f1_score(all_targets, predictions_binary)

    results = {
        'auc': auc,
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': all_preds,
        'targets': all_targets,
        'smiles': all_smiles
    }

    return results


# 特征统计函数 - 用于验证特征提取是否正确
def analyze_features(dataset):
    """分析数据集中特征的基本统计信息"""
    all_features = []
    for i in range(len(dataset)):
        data = dataset.get(i)
        all_features.append(data.x.numpy())

    all_features = np.vstack(all_features)

    stats = {
        'mean': np.mean(all_features, axis=0),
        'std': np.std(all_features, axis=0),
        'min': np.min(all_features, axis=0),
        'max': np.max(all_features, axis=0)
    }

    feature_names = ['AtomicNum', 'Degree', 'FormalCharge', 'Hybridization',
                     'IsAromatic', 'NumHs', 'NumRadicalElectrons', 'IsInRing']

    print("Feature Statistics:")
    for i, name in enumerate(feature_names):
        print(f"{name}: mean={stats['mean'][i]:.2f}, std={stats['std'][i]:.2f}, "
              f"min={stats['min'][i]:.2f}, max={stats['max'][i]:.2f}")

    return stats


# ====模型保存和加载 =====
def save_gat_model(model, training_config, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_node_features': training_config.get('num_node_features', 8),
            'num_edge_features': training_config.get('num_edge_features', 4),
            'hidden_dim': training_config.get('hidden_dim', 128),
            'num_heads': training_config.get('num_heads', 8),
            'num_layers': training_config.get('num_layers', 3),
            'dropout': training_config.get('dropout', 0.2),
            'num_classes': training_config.get('num_classes', 1)
        },
        'training_config': {
            'best_val_auc': training_config.get('best_val_auc', 0),
            'feature_names': training_config.get('feature_names',
                                                 ['AtomicNum', 'Degree', 'FormalCharge', 'Hybridization',
                                                  'IsAromatic', 'NumHs', 'NumRadicalElectrons', 'IsInRing'])
        }
    }

    # 保存模型
    torch.save(save_data, filepath)


def load_gat_model(filepath, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_config = checkpoint.get('model_config', {
                'num_node_features': 8,
                'num_edge_features': 4,
                'hidden_dim': 128,
                'num_heads': 8,
                'num_layers': 3,
                'dropout': 0.2,
                'num_classes': 1
            })

            model = GATModel(**model_config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("以完整格式加载模型成功")
            return model, checkpoint

        elif isinstance(checkpoint, dict):
            model = GATModel(
                num_node_features=8,
                num_edge_features=4,
                hidden_dim=128,
                num_heads=8,
                num_layers=3,
                dropout=0.2,
                num_classes=1
            ).to(device)

            model.load_state_dict(checkpoint)
            model.eval()
            print("以状态字典格式加载模型成功")
            return model, {'model_config': '默认配置'}

        else:
            print("未知的文件格式")
            return None, None

    except Exception as e:
        print(f"通用加载失败: {e}")
        return None, None


def predict_with_loaded_model(model, smiles_list, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    predictions = []

    with torch.no_grad():
        for smiles in smiles_list:
            # 将SMILES转换为图数据
            data = smiles_to_graph(smiles)
            if data is None:
                predictions.append(None)
                continue

            # 添加batch维度
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
            data = data.to(device)

            # 预测
            output = model(data)
            prediction = output.cpu().numpy()[0][0]
            predictions.append(prediction)

    return predictions


# === 主函数 ===

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据
    data = pd.concat([pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')])
    smiles_list = data["Smiles"].to_list()
    targets = data['Activity'].tolist()

    # 确保targets是列表而不是pandas Series
    if hasattr(targets, 'tolist'):
        targets = targets.tolist()

    # 创建数据集
    dataset = MolecularDataset(smiles_list, targets)
    print(f"成功加载 {len(dataset)} 个分子")

    # 分析特征
    # analyze_features(dataset)

    # 分割数据集 - 使用PyG的random_split
    from torch_geometric.data import DataLoader

    # 手动分割索引
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建子集
    train_dataset = [dataset.get(i) for i in train_indices]
    val_dataset = [dataset.get(i) for i in val_indices]
    test_dataset = [dataset.get(i) for i in test_indices]

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 获取特征维度
    sample_data = dataset.get(0)
    num_node_features = sample_data.x.shape[1]
    num_edge_features = sample_data.edge_attr.shape[1] if sample_data.edge_attr is not None else 0

    print(f"Node features: {num_node_features}, Edge features: {num_edge_features}")

    # 创建模型
    model = GATModel(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=128,
        num_heads=8,
        num_layers=3,
        dropout=0.2,
        num_classes=1
    ).to(device)

    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')

    # 训练模型
    training_results = train_model(
        model, train_loader, val_loader, device,
        num_epochs=100, lr=0.001
    )

    # 评估模型
    test_results = evaluate_model(model, test_loader, device)

    print(f'Test Results:')
    print(f'AUC: {test_results["auc"]:.4f}')
    print(f'Accuracy: {test_results["accuracy"]:.4f}')
    print(f'F1 Score: {test_results["f1_score"]:.4f}')

    # ===保存模型 ===
    training_config = {
        'num_node_features': num_node_features,
        'num_edge_features': num_edge_features,
        'hidden_dim': 128,
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.2,
        'num_classes': 1,
        'best_val_auc': training_results['best_val_auc'],
        'feature_names': ['AtomicNum', 'Degree', 'FormalCharge', 'Hybridization',
                          'IsAromatic', 'NumHs', 'NumRadicalElectrons', 'IsInRing']
    }

    # 保存模型
    save_gat_model(training_results['model'], training_config,
                   r"D:\Research\ML_data\colorectal_cancer\best_gat_model.pth")

    # === 加载模型并进行预测示例 ===
    print("\n" + "=" * 50)
    print("加载模型进行预测示例...")
    print("=" * 50)

    # 加载模型
    loaded_model, loaded_config = load_gat_model(r"D:\Research\ML_data\colorectal_cancer\best_gat_model.pth", device)

    # 使用加载的模型进行预测
    test_smiles = ['CC(=O)O', 'CCO', 'c1ccccc1']
    predictions = predict_with_loaded_model(loaded_model, test_smiles, device)

    print("预测结果示例:")
    for smiles, pred in zip(test_smiles, predictions):
        if pred is not None:
            activity = "Active" if pred > 0.5 else "Inactive"
            print(f"{smiles}: {pred:.4f} ({activity})")
        else:
            print(f"{smiles}: 无法解析")

    return training_results, test_results, loaded_model

# training_results, test_results, loaded_model = main()
