TEST_ONLY = False  # 设置为 True 时只进行评估，不进行训练
HIERARCHICAL_MODE = True  # 设置为 True 时使用层次化训练和测试

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datasets import ImbalancedDataset
from Model import BiLSTM, Transformer
from evaluate import evaluate_model,evaluate_model2
import pandas as pd

def set_random_seed(seed):
    """设置所有随机数种子以确保实验的可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机数种子: {seed}")

class MyRL():
    def __init__(self, input_shape, rho=0.01, model_type='BiLSTM', reward_multiplier=1.0, discount_factor=0.1, num_classes=9):
        self.discount_factor = discount_factor
        self.mem_size = 50000
        self.rho = rho
        self.reward_multiplier = reward_multiplier
        self.num_classes = num_classes
        self.t_max = 120000
        self.eta = 0.05
        self.learning_rate = 0.00025
        self.batch_size = 64
        self.ratio = 1
        
        self.loss_history = []
        
        # 根据模型类型选择网络
        if model_type == 'BiLSTM':
            self.q_net = BiLSTM(input_shape, output_dim=num_classes)
            self.target_net = BiLSTM(input_shape, output_dim=num_classes)
        elif model_type == 'Transformer':
            self.q_net = Transformer(input_shape, output_dim=num_classes)
            self.target_net = Transformer(input_shape, output_dim=num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.replay_memory = deque(maxlen=self.mem_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_net.to(self.device)
        self.target_net.to(self.device)
        
        self.step_count = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (self.t_max*self.ratio)
        
        self.model_type = model_type
        self.reward_weights = {}

    def set_reward_weights(self, reward_weights):
        """设置各类别的奖励权重"""
        self.reward_weights = reward_weights
        print(f"已设置奖励权重: {self.reward_weights}")

    def compute_reward(self, action, label):
        """多分类问题的奖励函数"""
        weight = self.reward_weights[label]
        terminal = False
        update = True
        
        if action == label:
            reward = weight * self.reward_multiplier
        else:
            reward = -weight * self.reward_multiplier
            terminal = True
            if action < label:
                update = False

        return reward, terminal, update

    def replay_experience(self, update_target=True):
        """从经验回放缓冲区采样并训练网络"""
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack(next_states).to(self.device)
        terminals = torch.tensor(terminals, dtype=torch.bool, device=self.device).unsqueeze(1)

        current_q = self.q_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.discount_factor * next_q * (~terminals)
        
        loss = F.mse_loss(current_q, target_q)
        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if update_target:
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.eta * param.data + (1.0 - self.eta) * target_param.data)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def train(self, dataset):
        """训练DQN分类器"""
        dist_info = dataset.get_class_distribution()
        if dist_info["reward_weights"] is not None:
            self.set_reward_weights(dist_info["reward_weights"])
            
        self.step_count = 0
        episode = 0
        
        total_pbar = tqdm(total=self.t_max, desc="Training Progress", unit="step")
        
        while self.step_count < self.t_max:
            episode += 1
            train_loader, _ = dataset.get_dataloaders()
            
            for batch_idx, (states, labels) in enumerate(train_loader):
                if self.step_count >= self.t_max:
                    break
                
                states = states.float().to(self.device)
                labels = labels.to(self.device)
                
                batch_size = states.size(0)
                for i in range(batch_size - 1):
                    current_state = states[i:i+1]
                    current_label = labels[i].item()
                    next_state = states[i+1:i+2]
                    
                    if random.random() < self.epsilon:
                        action = random.randint(0, self.num_classes - 1)
                    else:
                        with torch.no_grad():
                            q_values = self.q_net(current_state)
                        action = q_values.argmax().item()

                    reward, terminal, update = self.compute_reward(action, current_label)
                    
                    self.replay_memory.append((
                        current_state.squeeze(0).cpu().clone().detach(),
                        action,
                        reward,
                        next_state.squeeze(0).cpu().clone().detach(),
                        terminal
                    ))
                    
                    if len(self.replay_memory) >= 64:
                        self.replay_experience(update_target=update)
                        self.step_count += 1
                        total_pbar.update(1)
                        total_pbar.set_postfix({
                            'Episode': episode,
                            'Epsilon': f'{self.epsilon:.4f}',
                            'Memory': len(self.replay_memory)
                        })
        
        total_pbar.close()
        print("训练完成!")

    def plot_loss(self, save_path):
        """绘制训练过程中的损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('Training Steps', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(False)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"损失曲线已保存到 {save_path}")


class HierarchicalClassifier:
    """层次化分类器：组合class、K、M三个模型"""
    def __init__(self, class_model, k_model, m_model, device):
        self.class_model = class_model  # 分类0/K/M
        self.k_model = k_model          # K类细分
        self.m_model = m_model          # M类细分
        self.device = device
        
        # 定义标签映射
        self.k_label_mapping = {0: 2, 1: 3, 2: 5, 3: 6, 4: 8}
        self.m_label_mapping = {0: 1, 1: 4, 2: 7}
        
    def predict(self, x):
        """
        层次化预测
        Args:
            x: 输入数据 [batch_size, ...]
        Returns:
            predictions: 预测的原始标签 (0-8)
        """
        with torch.no_grad():
            # 第一层：分类为0/K/M
            class_output = self.class_model(x)
            class_pred = class_output.argmax(dim=1)
            
            # 初始化最终预测结果
            final_pred = torch.zeros_like(class_pred)
            
            # 对于每个样本进行处理
            for i in range(len(class_pred)):
                if class_pred[i] == 0:
                    # 预测为正常类(0)
                    final_pred[i] = 0
                elif class_pred[i] == 1:
                    # 预测为K类，使用K模型细分
                    k_output = self.k_model(x[i:i+1])
                    k_pred = k_output.argmax(dim=1).item()
                    final_pred[i] = self.k_label_mapping[k_pred]
                elif class_pred[i] == 2:
                    # 预测为M类，使用M模型细分
                    m_output = self.m_model(x[i:i+1])
                    m_pred = m_output.argmax(dim=1).item()
                    final_pred[i] = self.m_label_mapping[m_pred]
            
            return final_pred

def train_hierarchical_models(save_dir, model_type, input_shape, rho, 
                              reward_multiplier, discount_factor, run_number, datasets_name):
    """
    训练三个层次化模型
    Returns:
        三个模型的路径: (class_model_path, k_model_path, m_model_path)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 训练class分类器 (0/K/M)
    print(f"\n{'='*70}")
    print(f"第{run_number}次训练 - 步骤1: 训练class分类器 (0/K/M)")
    print(f"{'='*70}")
    
    class_dataset = ImbalancedDataset(dataset_name=f"{datasets_name}_class", rho=rho, batch_size=64)
    class_classifier = MyRL(input_shape, rho=rho, model_type=model_type,
                           reward_multiplier=reward_multiplier, discount_factor=discount_factor,
                           num_classes=3)  # 0, K, M
    class_classifier.train(class_dataset)
    
    class_model_filename = f'{datasets_name}_class_{model_type}_reward{reward_multiplier}_gamma{discount_factor}_第{run_number}次.pth'
    class_model_path = os.path.join(save_dir, class_model_filename)
    torch.save(class_classifier.q_net.state_dict(), class_model_path)
    print(f"Class模型已保存到 {class_model_path}")
    
    # 评估class分类器
    _, class_test_loader = class_dataset.get_dataloaders()
    evaluate_model2(class_classifier.q_net, class_test_loader, save_dir=save_dir,
                  dataset_name=f"{datasets_name}_class", training_ratio=1, rho=rho,
                  dataset_obj=class_dataset, run_number=run_number, model_type=model_type,
                  reward_multiplier=reward_multiplier, discount_factor=discount_factor)
    
    # 2. 训练K类分类器
    print(f"\n{'='*70}")
    print(f"第{run_number}次训练 - 步骤2: 训练K类分类器")
    print(f"{'='*70}")
    
    k_dataset = ImbalancedDataset(dataset_name=f"{datasets_name}_K", rho=rho, batch_size=64)
    k_classifier = MyRL(input_shape, rho=rho, model_type=model_type,
                       reward_multiplier=reward_multiplier, discount_factor=discount_factor,
                       num_classes=5)  # K类有6个子类
    k_classifier.train(k_dataset)
    
    k_model_filename = f'{datasets_name}_K_{model_type}_reward{reward_multiplier}_gamma{discount_factor}_第{run_number}次.pth'
    k_model_path = os.path.join(save_dir, k_model_filename)
    torch.save(k_classifier.q_net.state_dict(), k_model_path)
    print(f"K模型已保存到 {k_model_path}")
    
    # 评估K类分类器
    _, k_test_loader = k_dataset.get_dataloaders()
    evaluate_model2(k_classifier.q_net, k_test_loader, save_dir=save_dir,
                  dataset_name=f"{datasets_name}_K", training_ratio=1, rho=rho,
                  dataset_obj=k_dataset, run_number=run_number, model_type=model_type,
                  reward_multiplier=reward_multiplier, discount_factor=discount_factor)
    
    # 3. 训练M类分类器
    print(f"\n{'='*70}")
    print(f"第{run_number}次训练 - 步骤3: 训练M类分类器")
    print(f"{'='*70}")
    
    m_dataset = ImbalancedDataset(dataset_name=f"{datasets_name}_M", rho=rho, batch_size=64)
    m_classifier = MyRL(input_shape, rho=rho, model_type=model_type,
                       reward_multiplier=reward_multiplier, discount_factor=discount_factor,
                       num_classes=3)  # M类有3个子类
    m_classifier.train(m_dataset)
    
    m_model_filename = f'{datasets_name}_M_{model_type}_reward{reward_multiplier}_gamma{discount_factor}_第{run_number}次.pth'
    m_model_path = os.path.join(save_dir, m_model_filename)
    torch.save(m_classifier.q_net.state_dict(), m_model_path)
    print(f"M模型已保存到 {m_model_path}")
    
    # 评估M类分类器
    _, m_test_loader = m_dataset.get_dataloaders()
    evaluate_model2(m_classifier.q_net, m_test_loader, save_dir=save_dir,
                  dataset_name=f"{datasets_name}_M", training_ratio=1, rho=rho,
                  dataset_obj=m_dataset, run_number=run_number, model_type=model_type,
                  reward_multiplier=reward_multiplier, discount_factor=discount_factor)
    
    return class_model_path, k_model_path, m_model_path


def main():
    # 配置参数
    save_dir = '/workspace/RL/DRLimb-Multi/multi_class_results_hierarchical'
    datasets_name = "TBM_0.01"
    os.makedirs(save_dir, exist_ok=True)
    
    model_type = 'Transformer'
    input_shape = (1024, 3)
    rho = 0.01
    reward_multiplier = 1
    discount_factor = 0.01
    num_runs = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if HIERARCHICAL_MODE:
        print("="*70)
        print("层次化训练和测试模式")
        print("="*70)
        
        if not TEST_ONLY:
            # 训练模式：训练所有三个模型
            for run in range(1, num_runs + 1):
                print(f"\n{'#'*70}")
                print(f"开始第 {run} 次层次化训练")
                print(f"{'#'*70}")
                
                class_path, k_path, m_path = train_hierarchical_models(
                    save_dir, model_type, input_shape, rho,
                    reward_multiplier, discount_factor, run, datasets_name
                )
                
                # 4. 组合测试
                print(f"\n{'='*70}")
                print(f"第{run}次训练 - 步骤4: 组合模型测试 ({datasets_name}完整测试集)")
                print(f"{'='*70}")
                
                # 加载三个模型
                if model_type == 'BiLSTM':
                    class_model = BiLSTM(input_shape, output_dim=3)
                    k_model = BiLSTM(input_shape, output_dim=5)
                    m_model = BiLSTM(input_shape, output_dim=3)
                elif model_type == 'Transformer':
                    class_model = Transformer(input_shape, output_dim=3)
                    k_model = Transformer(input_shape, output_dim=5)
                    m_model = Transformer(input_shape, output_dim=3)
                
                class_model.load_state_dict(torch.load(class_path, weights_only=True), strict=False)
                k_model.load_state_dict(torch.load(k_path, weights_only=True), strict=False)
                m_model.load_state_dict(torch.load(m_path, weights_only=True), strict=False)
                
                class_model.to(device)
                k_model.to(device)
                m_model.to(device)
                
                class_model.eval()
                k_model.eval()
                m_model.eval()
                
                # 创建层次化分类器
                hierarchical_classifier = HierarchicalClassifier(class_model, k_model, m_model, device)
                
                # 加载完整测试集
                full_dataset = ImbalancedDataset(dataset_name=datasets_name, rho=rho, batch_size=64)
                _, full_test_loader = full_dataset.get_dataloaders()
                
                # 评估层次化模型
                evaluate_model(
                    hierarchical_classifier,
                    full_test_loader,
                    save_dir=save_dir,
                    dataset_name=datasets_name,
                    training_ratio=1,
                    rho=rho,
                    dataset_obj=full_dataset,
                    run_number=run,
                    model_type=f"{model_type}_Hierarchical",
                    reward_multiplier=reward_multiplier,
                    discount_factor=discount_factor
                )
        else:
            # 测试模式：加载已训练的模型
            print("测试模式：加载已有模型进行层次化测试")
            
            for run in range(1, num_runs + 1):
                print(f"\n{'='*70}")
                print(f"第{run}次测试 - 加载层次化模型")
                print(f"{'='*70}")
                
                class_filename = f'{datasets_name}_class_{model_type}_reward{reward_multiplier}_gamma{discount_factor}_第{run}次.pth'
                k_filename = f'{datasets_name}_K_{model_type}_reward{reward_multiplier}_gamma{discount_factor}_第{run}次.pth'
                m_filename = f'{datasets_name}_M_{model_type}_reward{reward_multiplier}_gamma{discount_factor}_第{run}次.pth'
                
                class_path = os.path.join(save_dir, class_filename)
                k_path = os.path.join(save_dir, k_filename)
                m_path = os.path.join(save_dir, m_filename)
                
                if not (os.path.exists(class_path) and os.path.exists(k_path) and os.path.exists(m_path)):
                    print(f"警告: 第{run}次训练的模型文件不完整，跳过")
                    continue
                
                # 加载三个模型
                if model_type == 'BiLSTM':
                    class_model = BiLSTM(input_shape, output_dim=3)
                    k_model = BiLSTM(input_shape, output_dim=5)
                    m_model = BiLSTM(input_shape, output_dim=3)
                elif model_type == 'Transformer':
                    class_model = Transformer(input_shape, output_dim=3)
                    k_model = Transformer(input_shape, output_dim=5)
                    m_model = Transformer(input_shape, output_dim=3)
                
                class_model.load_state_dict(torch.load(class_path, weights_only=True), strict=False)
                k_model.load_state_dict(torch.load(k_path, weights_only=True), strict=False)
                m_model.load_state_dict(torch.load(m_path, weights_only=True), strict=False)
                
                class_model.to(device)
                k_model.to(device)
                m_model.to(device)
                
                class_model.eval()
                k_model.eval()
                m_model.eval()
                
                # 创建层次化分类器
                hierarchical_classifier = HierarchicalClassifier(class_model, k_model, m_model, device)
                
                # 加载完整测试集
                full_dataset = ImbalancedDataset(dataset_name=datasets_name, rho=rho, batch_size=64)
                _, full_test_loader = full_dataset.get_dataloaders()
                
                # 评估层次化模型
                evaluate_model(
                    hierarchical_classifier, 
                    full_test_loader,
                    save_dir=save_dir,
                    dataset_name=datasets_name,
                    training_ratio=1,
                    rho=rho,
                    dataset_obj=full_dataset,
                    run_number=run,
                    model_type=f"{model_type}_Hierarchical",
                    reward_multiplier=reward_multiplier,
                    discount_factor=discount_factor
                )
    else:
        # 原有的直接训练模式
        print("直接训练模式 (非层次化)")
        # ... 保留原有代码 ...

if __name__ == "__main__":
    main()