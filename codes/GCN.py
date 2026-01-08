# %%
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.optim import Adam
import os


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


# SMILES到图转换
def smiles_to_graph(smiles):
    """
    将SMILES字符串转换为PyG图数据
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无法解析SMILES: {smiles}")
        return None

    # 氢原子显式添加
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

        # 添加两个方向（无向图）
        edge_indices += [[i, j], [j, i]]

        # 边特征
        bond_feats = bond_features(bond)
        edge_attrs += [bond_feats, bond_feats]

    if len(edge_indices) == 0:
        # 处理单原子分子
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

        # 确保smiles_list和targets是列表或numpy数组，而不是pandas Series
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

        # 预加载所有数据
        self.data_list = []
        for i, smiles in enumerate(self.smiles_list):
            data = smiles_to_graph(smiles)
            if data is not None:
                if self.targets is not None:
                    # 确保目标值是标量，而不是数组
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


# GCN模型定义
class GCNModel(nn.Module):
    def __init__(self,
                 num_node_features,
                 hidden_dim=128,
                 num_layers=3,
                 dropout=0.2,
                 num_classes=1):
        super(GCNModel, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 第一个GCN层
        self.conv1 = GCNConv(num_node_features, hidden_dim)

        # 中间GCN层
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        # 最后一个GCN层
        self.conv_last = GCNConv(hidden_dim, hidden_dim)

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
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 应用GCN层
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.conv_layers:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.conv_last(x, edge_index))

        # 全局池化
        x = self.pool(x, batch)

        # 分类
        out = self.classifier(x)

        return out


# 训练函数
def train_gcn_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    train_losses = []
    val_losses = []
    val_aucs = []

    best_val_auc = 0
    best_model_state = None
    no_improvement_count = 0
    patience = 20

    print("开始训练GCN模型...")

    for epoch in range(num_epochs):
        # 训练阶段
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

            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        # 更新学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_auc)
        new_lr = optimizer.param_groups[0]['lr']

        # 如果学习率发生变化，打印信息
        if new_lr != old_lr:
            print(f"学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")

        current_lr = new_lr

        # 检查性能提升
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            no_improvement_count = 0
            improvement_msg = "↑"  # 提升标记
        else:
            no_improvement_count += 1
            improvement_msg = "→"  # 无提升标记

        # 每epoch都打印进度
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f} {improvement_msg}, '
              f'LR: {current_lr:.6f}, No improvement: {no_improvement_count}/{patience}')

        # 早停检查
        if no_improvement_count >= patience:
            print(f"早停触发! 连续 {patience} 个epoch验证集AUC没有提升。")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"加载最佳模型，验证集AUC: {best_val_auc:.4f}")

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': val_aucs,
        'best_val_auc': best_val_auc
    }


# 评估函数
def evaluate_gcn_model(model, test_loader, device):
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


def save_gcn_model(model, training_config, filepath):
    """
    保存GCN模型和训练配置
    """
    try:
        if not filepath:
            filepath = "gcn_model.pth"

        if not filepath.endswith('.pth'):
            filepath += '.pth'

        save_data = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_node_features': int(training_config.get('num_node_features', 8)),
                'hidden_dim': int(training_config.get('hidden_dim', 128)),
                'num_layers': int(training_config.get('num_layers', 3)),
                'dropout': float(training_config.get('dropout', 0.2)),
                'num_classes': int(training_config.get('num_classes', 1))
            },
            'training_config': {
                'best_val_auc': float(training_config.get('best_val_auc', 0)),
                'feature_names': list(training_config.get('feature_names',
                                                          ['AtomicNum', 'Degree', 'FormalCharge', 'Hybridization',
                                                           'IsAromatic', 'NumHs', 'NumRadicalElectrons', 'IsInRing']))
            }
        }

        # 保存模型
        torch.save(save_data, filepath)
        print(f"GCN模型已保存到: {filepath}")
        print(f"验证集最佳AUC: {training_config.get('best_val_auc', 'N/A')}")
        return True

    except Exception as e:
        print(f"保存失败: {e}")
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': training_config
            }, f"weights_{filepath}")

            torch.save(model.state_dict(), f"pure_weights_{filepath}")

            print(f"模型权重已备份保存到: weights_{filepath} 和 pure_weights_{filepath}")
            return True
        except Exception as e2:
            print(f"备用保存也失败: {e2}")
            return False


def load_gcn_model(filepath, device=None):
    """
    加载GCN模型
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 检查文件是否存在
    if not os.path.exists(filepath):
        # 尝试查找备份文件
        backup_files = [
            f"weights_{filepath}",
            f"pure_weights_{filepath}",
            filepath.replace('.pth', '_backup.pth')
        ]

        for backup in backup_files:
            if os.path.exists(backup):
                filepath = backup
                print(f"使用备份文件: {filepath}")
                break
        else:
            raise FileNotFoundError(f"模型文件不存在: {filepath} 和所有备份文件")

    checkpoint = None

    methods = [
        lambda: torch.load(filepath, map_location=device, weights_only=False),
        lambda: torch.load(filepath, map_location=device),
        lambda: __import__('pickle').load(open(filepath, 'rb'))
    ]

    for i, method in enumerate(methods):
        try:
            checkpoint = method()
            print(f"使用方法 {i + 1} 加载成功")
            break
        except Exception as e:
            print(f"方法 {i + 1} 失败: {e}")
            continue

    if checkpoint is None:
        raise RuntimeError("所有加载方法都失败了")

    # 处理不同的检查点格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_config = checkpoint.get('model_config', {
                'num_node_features': 8,
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'num_classes': 1
            })
            state_dict = checkpoint['model_state_dict']
        else:
            model_config = {
                'num_node_features': 8,
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'num_classes': 1
            }
            state_dict = checkpoint
    else:
        model_config = {
            'num_node_features': 8,
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'num_classes': 1
        }
        state_dict = checkpoint

    model = GCNModel(
        num_node_features=model_config['num_node_features'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        num_classes=model_config['num_classes']
    )

    # 加载模型权重
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"GCN模型已从 {filepath} 加载")
    print(f"模型配置: {model_config}")

    if isinstance(checkpoint, dict) and 'training_config' in checkpoint:
        print(f"最佳验证AUC: {checkpoint['training_config'].get('best_val_auc', 'N/A')}")

    return model, checkpoint


# 预测函数
def predict_with_gcn_model(model, smiles_list, device=None):
    """
    使用加载的GCN模型进行预测
    """
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


# 模型加载函数（适用于所有模型）
def universal_model_loader(filepath, model_class, default_config, device=None):
    """
    通用模型加载函数
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")

    checkpoint = None

    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except:
        try:
            checkpoint = torch.load(filepath, map_location=device)
        except:
            try:
                import pickle
                with open(filepath, 'rb') as f:
                    checkpoint = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"所有加载方法都失败: {e}")

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_config = checkpoint.get('model_config', default_config)
            state_dict = checkpoint['model_state_dict']
        else:
            model_config = default_config
            state_dict = checkpoint
    else:
        model_config = default_config
        state_dict = checkpoint

    # 创建模型并加载权重
    model = model_class(**model_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"模型已从 {filepath} 加载")
    return model, checkpoint


# 使用通用加载器的GCN加载函数
def load_gcn_model_universal(filepath, device=None):
    """
    使用通用加载器加载GCN模型
    """
    default_config = {
        'num_node_features': 8,
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'num_classes': 1
    }
    return universal_model_loader(filepath, GCNModel, default_config, device)


# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据
    data = pd.concat([pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')])
    smiles_list = data["Smiles"].to_list()
    targets = data['Activity'].tolist()

    if hasattr(targets, 'tolist'):
        targets = targets.tolist()

    # 创建数据集
    dataset = MolecularDataset(smiles_list, targets)

    print(f"成功加载 {len(dataset)} 个分子")

    # 分割数据集
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

    print(f"Node features: {num_node_features}")

    # 创建GCN模型
    model = GCNModel(
        num_node_features=num_node_features,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        num_classes=1
    ).to(device)

    print(f'GCN模型创建完成，参数数量: {sum(p.numel() for p in model.parameters())}')

    # 训练模型
    training_results = train_gcn_model(
        model, train_loader, val_loader, device,
        num_epochs=100, lr=0.001
    )

    # 评估模型
    test_results = evaluate_gcn_model(model, test_loader, device)

    print(f'GCN Test Results:')
    print(f'AUC: {test_results["auc"]:.4f}')
    print(f'Accuracy: {test_results["accuracy"]:.4f}')
    print(f'F1 Score: {test_results["f1_score"]:.4f}')

    # 保存模型
    training_config = {
        'num_node_features': num_node_features,
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'num_classes': 1,
        'best_val_auc': training_results['best_val_auc'],
        'feature_names': ['AtomicNum', 'Degree', 'FormalCharge', 'Hybridization',
                          'IsAromatic', 'NumHs', 'NumRadicalElectrons', 'IsInRing']
    }

    save_gcn_model(model, training_config, "best_gcn_model.pth")

    # 测试加载和预测
    print("\n测试模型加载和预测...")
    try:
        loaded_model, loaded_config = load_gcn_model("best_gcn_model.pth", device)

        test_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1']
        predictions = predict_with_gcn_model(loaded_model, test_smiles, device)

        print("预测结果示例:")
        for smiles, pred in zip(test_smiles, predictions):
            if pred is not None:
                activity = "Active" if pred > 0.5 else "Inactive"
                print(f"  {smiles}: {pred:.4f} ({activity})")
            else:
                print(f"  {smiles}: 无法解析")
    except Exception as e:
        print(f"模型加载测试失败: {e}")
        print("尝试其他加载方法...")

        try:
            state_dict = torch.load("best_gcn_model.pth", map_location=device, weights_only=False)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

            loaded_model = GCNModel(
                num_node_features=8,
                hidden_dim=128,
                num_layers=3,
                dropout=0.2,
                num_classes=1
            ).to(device)
            loaded_model.load_state_dict(state_dict)
            loaded_model.eval()

            print("备用加载方法成功!")

            # 测试预测
            test_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1']
            predictions = predict_with_gcn_model(loaded_model, test_smiles, device)

            print("预测结果示例:")
            for smiles, pred in zip(test_smiles, predictions):
                if pred is not None:
                    activity = "Active" if pred > 0.5 else "Inactive"
                    print(f"  {smiles}: {pred:.4f} ({activity})")
                else:
                    print(f"  {smiles}: 无法解析")
        except Exception as e2:
            print(f"所有加载方法都失败: {e2}")

    return training_results, test_results

# training_results, test_results = main()