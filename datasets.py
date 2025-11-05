import torch
import torchvision
import numpy as np
import h5py
import os
from torch.utils.data import DataLoader, Subset, TensorDataset
# from sklearn.model_selection import train_test_split

class ImbalancedDataset:
    def __init__(self, dataset_name="TBM_0.01", rho=0.01, batch_size=64, seed=42):
        """
        初始化数据集处理类
        :param dataset_name: 数据集名称 
        :param rho: 不平衡因子 (用于某些数据集)
        :param batch_size: DataLoader 批次大小
        :param seed: 随机种子(确保可复现)
        """
        self.dataset_name = dataset_name
        self.rho = rho
        self.batch_size = batch_size
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 加载并预处理数据
        self.train_data, self.test_data = self.load_raw_data()
        
        # 添加存储类别样本数的字典
        self.class_counts = {}
        
        # 预处理数据(针对多分类)
        self._preprocess_data()

    def load_raw_data(self):
        """加载原始数据集(需扩展时在此添加新数据集)"""
        if self.dataset_name in ["TBM_0.01", "TBM_0.01_class", "TBM_0.01_K", "TBM_0.01_M"]:
            # 所有TBM数据集使用统一的文件路径
            print(f"正在加载TBM训练集...")
            train_data, train_labels = self._load_h5_file('/datasets/TBM/train_data/data/train_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05_ratio0.01_head10000.h5')
            
            print(f"正在加载TBM测试集...")
            test_data, test_labels = self._load_h5_file('/datasets/TBM/train_data/data/test_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05.h5')
            
            # 创建训练集和测试集
            train_set = self._create_dataset_from_arrays(train_data, train_labels)
            test_set = self._create_dataset_from_arrays(test_data, test_labels)
            
            return train_set, test_set
        elif self.dataset_name == "TBM_0.001":
            # 所有TBM数据集使用统一的文件路径
            print(f"正在加载TBM训练集...")
            train_data, train_labels = self._load_h5_file('/datasets/TBM/train_data/data/train_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05_ratio0.001_head10000.h5')
            
            print(f"正在加载TBM测试集...")
            test_data, test_labels = self._load_h5_file('/datasets/TBM/train_data/data/test_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05.h5')
            
            # 创建训练集和测试集
            train_set = self._create_dataset_from_arrays(train_data, train_labels)
            test_set = self._create_dataset_from_arrays(test_data, test_labels)
            
            return train_set, test_set
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
    def _load_h5_file(self, file_path):
        """从h5文件中加载数据和标签"""
        with h5py.File(file_path, 'r') as h5f:
            data = h5f['data'][:]
            labels = h5f['labels'][:]
        return data, labels
        
    def _create_dataset_from_arrays(self, data, labels):
        """从NumPy数组创建一个类似torchvision数据集的对象"""
        # 创建一个具有类似torchvision数据集接口的对象
        dataset = type('', (), {})()
        dataset.data = data
        dataset.targets = labels
        return dataset

    def _filter_and_remap_classes(self, data, labels, class_list, label_mapping):
        """
        过滤特定类别并重新映射标签
        :param data: 原始数据
        :param labels: 原始标签
        :param class_list: 要保留的类别列表
        :param label_mapping: 标签映射字典
        :return: 过滤后的数据和重新映射的标签
        """
        # 创建掩码，选择指定类别
        mask = np.isin(labels, class_list)
        filtered_data = data[mask]
        filtered_labels = labels[mask]
        
        # 重新映射标签
        remapped_labels = np.array([label_mapping[label] for label in filtered_labels])
        
        return filtered_data, remapped_labels

    def _downsample_to_min_class(self, data, labels):
        """
        将所有类别降采样到最少样本数
        :param data: 数据
        :param labels: 标签
        :return: 降采样后的数据和标签
        """
        unique_classes = np.unique(labels)
        class_counts = {cls: np.sum(labels == cls) for cls in unique_classes}
        min_count = min(class_counts.values())
        
        sampled_data = []
        sampled_labels = []
        
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            sampled_indices = np.random.choice(cls_indices, size=min_count, replace=False)
            sampled_data.append(data[sampled_indices])
            sampled_labels.append(labels[sampled_indices])
        
        return np.concatenate(sampled_data), np.concatenate(sampled_labels)

    def _preprocess_data(self):
        """
        核心预处理：处理多分类数据
        - 保留原始标签(0-8)
        - 计算每个类别的样本数
        """
        # 获取标签数据 - 处理不同数据集的标签格式
        if isinstance(self.train_data.targets, list):
            train_labels = np.array(self.train_data.targets)
        elif isinstance(self.train_data.targets, np.ndarray):
            train_labels = self.train_data.targets
        else:
            train_labels = self.train_data.targets.numpy()
            
        if isinstance(self.test_data.targets, list):
            test_labels = np.array(self.test_data.targets)
        elif isinstance(self.test_data.targets, np.ndarray):
            test_labels = self.test_data.targets
        else:
            test_labels = self.test_data.targets.numpy()
        
        # 获取训练数据
        if not isinstance(self.train_data.data, np.ndarray):
            train_data = self.train_data.data
        else:
            train_data = self.train_data.data
            
        if not isinstance(self.test_data.data, np.ndarray):
            test_data = self.test_data.data
        else:
            test_data = self.test_data.data
        
        # 根据dataset_name进行不同的处理
        if self.dataset_name == "TBM_0.01_class":
            # 合并为三类: 0(正常), K类(0,2,3,5,6,8), M类(1,4,7)
            # 标签映射: 0->0, K类->1, M类->2
            print("处理TBM_0.01_class: 合并为0, K, M三类并降采样...")
            
            # 创建标签映射
            label_mapping = {
                0: 0,  # 正常类
                2: 1, 3: 1, 5: 1, 6: 1, 8: 1,  # K类
                1: 2, 4: 2, 7: 2  # M类
            }
            
            # 处理训练集
            train_labels_mapped = np.array([label_mapping[label] for label in train_labels])
            train_data, train_labels_mapped = self._downsample_to_min_class(train_data, train_labels_mapped)
            
            # 处理测试集
            test_labels_mapped = np.array([label_mapping[label] for label in test_labels])
            
            train_labels = train_labels_mapped
            test_labels = test_labels_mapped
            
        elif self.dataset_name == "TBM_0.01_K":
            # 只保留K类(2,3,5,6,8)
            print("处理TBM_0.01_K: 只保留K类...")
            k_classes = [2, 3, 5, 6, 8]
            label_mapping = {2: 0, 3: 1, 5: 2, 6: 3, 8: 4}
            
            train_data, train_labels = self._filter_and_remap_classes(
                train_data, train_labels, k_classes, label_mapping
            )
            test_data, test_labels = self._filter_and_remap_classes(
                test_data, test_labels, k_classes, label_mapping
            )
            
        elif self.dataset_name == "TBM_0.01_M":
            # 只保留M类(1,4,7)
            print("处理TBM_0.01_M: 只保留M类...")
            m_classes = [1, 4, 7]
            label_mapping = {1: 0, 4: 1, 7: 2}
            
            train_data, train_labels = self._filter_and_remap_classes(
                train_data, train_labels, m_classes, label_mapping
            )
            test_data, test_labels = self._filter_and_remap_classes(
                test_data, test_labels, m_classes, label_mapping
            )
        
        # 计算训练集中每个类别的样本数
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)
        self.class_counts = {cls: count for cls, count in zip(unique_classes, class_counts)}
        
        # 计算最少样本的类别
        min_class = min(self.class_counts.items(), key=lambda x: x[1])
        self.min_class = min_class[0]
        self.min_count = min_class[1]
        
        # 计算每个类别的奖励权重 (最少样本数/该类样本数)
        self.reward_weights = {cls: self.min_count / count for cls, count in self.class_counts.items()}
        
        # 确保训练数据是torch张量
        if not isinstance(train_data, torch.Tensor):
            train_data = torch.tensor(train_data)
            
        self.train_data = TensorDataset(train_data, torch.tensor(train_labels))
        
        # 处理测试集
        if not isinstance(test_data, torch.Tensor):
            test_data = torch.tensor(test_data)
            
        self.test_data = TensorDataset(test_data, torch.tensor(test_labels))

    def get_dataloaders(self):
        """
        生成训练和测试 DataLoader
        :return: (train_loader, test_loader)
        """
        
        train_loader = DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        return train_loader, test_loader
        
    def get_full_dataset(self):
        """
        直接返回完整的训练和测试数据集
        :return: (train_data, train_labels, test_data, test_labels)
        """
        train_data = self.train_data.tensors[0]
        train_labels = self.train_data.tensors[1]
        test_data = self.test_data.tensors[0]
        test_labels = self.test_data.tensors[1]
        return train_data, train_labels, test_data, test_labels

    def get_class_distribution(self):
        """返回处理后的类别分布(用于验证)"""
        train_labels = self.train_data.tensors[1].numpy()
        test_labels = self.test_data.tensors[1].numpy()
        
        # 获取所有唯一的类别标签
        all_classes = sorted(np.unique(np.concatenate([train_labels, test_labels])))
        
        # 计算训练集中每个类别的数量
        train_counts = np.bincount(train_labels, minlength=max(all_classes)+1)
        test_counts = np.bincount(test_labels, minlength=max(all_classes)+1)
        
        # 返回按标签分组的计数
        return {
            "train": train_counts,
            "test": test_counts,
            "classes": all_classes,
            "reward_weights": self.reward_weights if hasattr(self, 'reward_weights') else None,
            "min_class": self.min_class if hasattr(self, 'min_class') else None,
            "min_count": self.min_count if hasattr(self, 'min_count') else None
        }