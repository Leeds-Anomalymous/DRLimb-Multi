import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
from datetime import datetime
import os

def compute_group_accuracy(y_true, y_pred, group_indices):
    """
    计算特定组的准确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        group_indices: 该组包含的类别索引
    
    Returns:
        float: 该组的准确率
    """
    # 找出属于该组的样本
    group_mask = np.isin(y_true, group_indices)
    
    if not np.any(group_mask):
        return 0.0  # 如果没有该组的样本，返回0
        
    # 计算该组的准确率
    group_true = y_true[group_mask]
    group_pred = y_pred[group_mask]
    
    return accuracy_score(group_true, group_pred)

def compute_metrics(y_true, y_pred):
    """计算各类评估指标"""
    # 总体准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 头部类别[0,1,2]的准确率
    head_acc = compute_group_accuracy(y_true, y_pred, [0, 1, 2])
    
    # 中部类别[3,4,5]的准确率
    mid_acc = compute_group_accuracy(y_true, y_pred, [3, 4, 5])
    
    # 尾部类别[6,7,8]的准确率
    tail_acc = compute_group_accuracy(y_true, y_pred, [6, 7, 8])
    
    return {
        'accuracy': accuracy,
        'head_acc': head_acc,
        'mid_acc': mid_acc,
        'tail_acc': tail_acc
    }

def plot_confusion_matrix(y_true, y_pred, save_path=None, model_type=None, dataset_name=None, 
                         training_ratio=None, rho=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    
    # 增大方格内数字字体并保持其它参数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(9),  # 9个类别
                yticklabels=range(9),  # 9个类别
                annot_kws={'fontsize':16})
    
    # 放大坐标轴标签字体和刻度字体
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到 {save_path}")
    
    # 关闭图形以释放内存，不显示窗口
    plt.close()

def evaluate_model(model, test_loader, save_dir='/root/autodl-tmp/checkpoints', dataset_name=None, training_ratio=None, rho=None, dataset_obj=None, run_number=None, model_type=None, reward_multiplier=1.0, discount_factor=0.1):
    """
    评估模型性能并计算相关指标
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        save_dir: 保存结果的目录
        dataset_name: 数据集名称
        training_ratio: 训练完成比例
        rho: 不平衡率
        dataset_obj: 数据集对象，用于获取样本数量统计
        run_number: 运行次数编号，用于文件命名
        model_type: 模型类型名称
        reward_multiplier: 奖励倍数
        discount_factor: 折扣因子
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            # 根据模型类型进行不同的数据预处理
            if 'TBM' in str(dataset_name) or 'TBM' in str(model_type):
                # 针对TBM数据的处理
                # 如果数据是4D，需要调整为3D [batch_size, channels, length]
                if len(data.shape) == 4:  # [batch_size, 1, 3, 1024]
                    data = data.squeeze(1)  # 移除额外的维度，变为[batch_size, 3, 1024]
                
                # 如果维度顺序是[batch_size, length, channels]，需要转置
                if data.shape[1] != 3 and data.shape[2] == 3:
                    data = data.transpose(1, 2)  # 转换为[batch_size, channels, length]
                
                data = data.float().to(device)
            else:
                # 图像数据需要添加通道维度
                if len(data.shape) == 3:  # (N, 28, 28)
                    data = data.unsqueeze(1)  # 添加通道维度 -> (N, 1, 28, 28)
                # 修正通道顺序（如果需要）
                if data.shape[1] != 3 and data.shape[-1] == 3:
                    data = data.permute(0, 3, 1, 2)  # NHWC -> NCHW
                data = data.float().to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    metrics = compute_metrics(all_labels, all_preds)
    
    # 打印结果
    print("\n===== 模型评估结果 =====")
    print(f"总体准确率: {metrics['accuracy']:.4f}")
    print(f"头部准确率 [0,1,2]: {metrics['head_acc']:.4f}")
    print(f"中部准确率 [3,4,5]: {metrics['mid_acc']:.4f}")
    print(f"尾部准确率 [6,7,8]: {metrics['tail_acc']:.4f}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取数据集统计信息
    max_class_count = 0
    min_class_count = float('inf')
    test_samples_count = 0
    
    if dataset_obj is not None:
        try:
            # 获取类别分布
            class_distribution = dataset_obj.get_class_distribution()
            train_counts = class_distribution['train']
            
            # 获取最多样本的类别（通常是正常样本，即类别0）
            max_class_count = train_counts[0]  # 假设类别0有最多样本
            
            # 获取最少样本的类别（通常是类别8）
            min_class_count = min(train_counts)
            
            # 测试集总样本数
            test_samples_count = sum(class_distribution['test'])
            
            print(f"最多样本类别数量: {max_class_count}")
            print(f"最少样本类别数量: {min_class_count}")
            print(f"测试集样本总数: {test_samples_count}")
        except Exception as e:
            print(f"获取数据集统计信息时出错: {e}")
    
    # 准备DataFrame数据
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        '评估时间': [current_time],
        '数据集名称': [dataset_name if dataset_name else 'Unknown'],
        '模型类型': [model_type if model_type else 'Unknown'],
        '训练完成比例': [training_ratio if training_ratio is not None else 'Unknown'],
        '不平衡率rho': [rho if rho is not None else 'Unknown'],
        '奖励倍数': [reward_multiplier if reward_multiplier is not None else 1.0],
        '折扣因子': [discount_factor if discount_factor is not None else 0.1],
        '最多(正常)样本数': [max_class_count],
        '最少样本数': [min_class_count],
        '测试集样本数': [test_samples_count],
        '总体准确率': [metrics['accuracy']],
        '头部准确率': [metrics['head_acc']],
        '中部准确率': [metrics['mid_acc']],
        '尾部准确率': [metrics['tail_acc']]
    }
    
    new_df = pd.DataFrame(new_data)
    
    # Excel文件路径
    excel_path = os.path.join(save_dir, 'evaluation_results.xlsx')
    
    # 检查是否存在现有文件
    if os.path.exists(excel_path):
        try:
            # 读取现有数据（包含标题行）
            existing_df = pd.read_excel(excel_path, header=0)
            # 合并数据
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"读取现有Excel文件时出错: {e}")
            print("将创建新文件")
            combined_df = new_df
    else:
        combined_df = new_df
    
    # 保存到Excel文件（包含标题行）
    try:
        combined_df.to_excel(excel_path, index=False, header=True)
        print(f"评估结果已保存到 {excel_path}")
        print(f"当前文件包含 {len(combined_df)} 条评估记录")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
    
    # 绘制混淆矩阵
    # 生成带数据集名称、模型类型、不平衡率、奖励倍数、折扣因子、训练完成比例和序号的文件名
    dataset_str = dataset_name if dataset_name else 'Unknown'
    model_str = model_type if model_type else 'Unknown'
    rho_str = f"rho{rho}" if rho is not None else 'rhoUnknown'
    reward_str = f"reward{reward_multiplier}" if reward_multiplier is not None else 'reward1.0'
    gamma_str = f"gamma{discount_factor}" if discount_factor is not None else 'gamma0.1'
    ratio_str = f"{training_ratio}" if training_ratio is not None else 'Unknown'
    
    cm_filename = f'{dataset_str}_{model_str}_{rho_str}_{reward_str}_{gamma_str}_训练完成比{ratio_str}_第{run_number}次_cm.png'
    cm_path = os.path.join(save_dir, cm_filename)
    
    plot_confusion_matrix(all_labels, all_preds, save_path=cm_path, model_type=model_type,
                         dataset_name=dataset_name, training_ratio=training_ratio, rho=rho)
    
    return metrics