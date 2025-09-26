import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
from datetime import datetime
import os

def compute_gmean(y_true, y_pred):
    """
    计算G-mean: sqrt(recall * specificity)
    sensitivity = recall of positive class
    specificity = recall of negative class
    """
    # 少数类(0)的召回率
    recall = recall_score(y_true, y_pred, pos_label=0)
    
    # 多数类(1)的召回率
    specificity = recall_score(y_true, y_pred, pos_label=1)
    
    # 计算G-mean
    g_mean = np.sqrt(recall * specificity)
    
    return g_mean

def compute_metrics(y_true, y_pred):
    """计算F1分数和G-mean"""
    # F1-score (针对少数类)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # G-mean
    g_mean = compute_gmean(y_true, y_pred)
    
    # 准确率
    accuracy = (y_true == y_pred).sum() / len(y_true)
    
    # 计算每个类别的准确率
    class_0_acc = ((y_true == 0) & (y_pred == 0)).sum() / (y_true == 0).sum()
    class_1_acc = ((y_true == 1) & (y_pred == 1)).sum() / (y_true == 1).sum()
    
    return {
        'accuracy': accuracy,
        'class_0_acc': class_0_acc,
        'class_1_acc': class_1_acc,
        'f1_minority': f1[0],  # 少数类F1
        'f1_majority': f1[1],  # 多数类F1
        'f1_macro': f1.mean(),  # 宏平均F1
        'g_mean': g_mean
    }

def plot_confusion_matrix(y_true, y_pred, save_path=None, model_type=None, dataset_name=None, 
                         training_ratio=None, rho=None, train_pos_count=None, train_neg_count=None,
                         test_pos_count=None, test_neg_count=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    # 增大方格内数字字体并保持其它参数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Minority (0)', 'Majority (1)'],
                yticklabels=['Minority (0)', 'Majority (1)'],
                annot_kws={'fontsize':24})
    # 放大坐标轴标签字体和刻度字体
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # 构建详细的标题
    title_parts = []
    
    # 基础标题
    if model_type:
        title_parts.append(f'Confusion Matrix - {model_type}')
    else:
        title_parts.append('Confusion Matrix')
    
    # 添加详细信息
    if dataset_name:
        title_parts.append(f'Dataset: {dataset_name}')
    if training_ratio is not None:
        title_parts.append(f'Training Ratio: {training_ratio}')
    if rho is not None:
        title_parts.append(f'Imbalance Ratio (rho): {rho}')
    
    # 添加样本数信息
    sample_info = []
    if train_pos_count is not None and train_neg_count is not None:
        sample_info.append(f'Train: Pos={train_pos_count}, Neg={train_neg_count}')
    if test_pos_count is not None and test_neg_count is not None:
        sample_info.append(f'Test: Pos={test_pos_count}, Neg={test_neg_count}')
    
    if sample_info:
        title_parts.extend(sample_info)
    
    # 组合标题，使用换行符分隔
    #full_title = '\n'.join(title_parts)
    # 不显示标题（已移除 plt.title 调用）
    # plt.title(full_title, fontsize=8, pad=20)

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
            if 'TBM' in str(dataset_name) or model_type == 'TBM_conv1d':
                # 打印数据形状以便调试
                # print(f"原始数据形状: {data.shape}")
                
                # 针对TBM数据的处理
                # 如果数据是4D，需要调整为3D [batch_size, channels, length]
                if len(data.shape) == 4:  # [batch_size, 1, 3, 1024]
                    data = data.squeeze(1)  # 移除额外的维度，变为[batch_size, 3, 1024]
                
                # 如果维度顺序是[batch_size, length, channels]，需要转置
                if data.shape[1] != 3 and data.shape[2] == 3:
                    data = data.transpose(1, 2)  # 转换为[batch_size, channels, length]
                
                # 打印处理后的形状
                # print(f"处理后数据形状: {data.shape}")
                
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
    print(f"少数类准确率: {metrics['class_0_acc']:.4f}")
    print(f"多数类准确率: {metrics['class_1_acc']:.4f}")
    print(f"少数类F1-score: {metrics['f1_minority']:.4f}")
    print(f"多数类F1-score: {metrics['f1_majority']:.4f}")
    print(f"宏平均F1-score: {metrics['f1_macro']:.4f}")
    print(f"G-mean: {metrics['g_mean']:.4f}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取数据集统计信息
    train_positive_count = 0
    train_negative_count = 0
    test_positive_count = 0
    test_negative_count = 0
    
    if dataset_obj is not None:
        try:
            # 获取类别分布
            class_distribution = dataset_obj.get_class_distribution()
            train_positive_count = int(class_distribution['train'][0])  # 正类(标签0)数量
            train_negative_count = int(class_distribution['train'][1])  # 负类(标签1)数量
            test_positive_count = int(class_distribution['test'][0])    # 正类(标签0)数量
            test_negative_count = int(class_distribution['test'][1])    # 负类(标签1)数量
            
            print(f"训练集 - 正类样本数: {train_positive_count}, 负类样本数: {train_negative_count}")
            print(f"测试集 - 正类样本数: {test_positive_count}, 负类样本数: {test_negative_count}")
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
        '折扣因子': [discount_factor if discount_factor is not None else 0.1],  # 添加折扣因子
        '训练集正类样本数': [train_positive_count],
        '训练集负类样本数': [train_negative_count],
        '测试集正类样本数': [test_positive_count],
        '测试集负类样本数': [test_negative_count],
        '总体准确率': [metrics['accuracy']],
        '少数类准确率': [metrics['class_0_acc']],
        '多数类准确率': [metrics['class_1_acc']],
        '少数类F1-score': [metrics['f1_minority']],
        '多数类F1-score': [metrics['f1_majority']],
        '宏平均F1-score': [metrics['f1_macro']],
        'G-mean': [metrics['g_mean']]
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
    
    cm_filename = f'{dataset_str}_{model_str}_{rho_str}_{reward_str}_{gamma_str}_训练完成比{ratio_str}_第{run_number}次.png'
    cm_path = os.path.join(save_dir, cm_filename)
    
    plot_confusion_matrix(all_labels, all_preds, save_path=cm_path, model_type=model_type,
                         dataset_name=dataset_name, training_ratio=training_ratio, rho=rho,
                         train_pos_count=train_positive_count, train_neg_count=train_negative_count,
                         test_pos_count=test_positive_count, test_neg_count=test_negative_count)
    
    return metrics
