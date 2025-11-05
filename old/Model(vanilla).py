import torch
import torch.nn as nn
import math

class Q_Net_image(nn.Module):
    def __init__(self, input_shape, output_dim=10): 
        super(Q_Net_image, self).__init__()
        channels, height, width = input_shape 
        
        # Layer 1
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 先不初始化全连接层，在第一次前向传播时动态创建
        self.fc1 = None
        self.relu3 = nn.ReLU()
        self.fc2 = None
        self.output_dim = output_dim

    def _get_conv_output(self, shape):
        x = torch.zeros(1, *shape) 
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return int(torch.prod(torch.tensor(x.shape[1:]))) # 计算卷积输出维度

    def _initialize_fc_layers(self, conv_output_size):
        """动态初始化全连接层"""
        if self.fc1 is None:
            self.fc1 = nn.Linear(conv_output_size, 256)
            self.fc2 = nn.Linear(256, self.output_dim)
            # 确保新层在正确的设备上
            device = next(self.parameters()).device
            self.fc1 = self.fc1.to(device)
            self.fc2 = self.fc2.to(device)

    def forward(self, x):
        # 确保输入和模型权重在同一设备
        x = x.to(next(self.parameters()).device)
        
        # 根据输入形状判断数据类型并进行相应处理
        if len(x.shape) == 3 and x.shape[1] > 1 and x.shape[2] > 1:
            # TBM数据形状为 [batch_size, len_window, feature_dim]
            # 需要转换为 [batch_size, channels, len_window, feature_dim]
            x = x.unsqueeze(1)  # 添加通道维度 [batch_size, 1, len_window, feature_dim]
            x = x.repeat(1, 3, 1, 1)  # 复制为3通道 [batch_size, 3, len_window, feature_dim]
        elif len(x.shape) == 3:  
            # MNIST/CIFAR10数据 - 单个样本时
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            # 扁平化的输入数据，重塑为所需的形状
            x = x.unsqueeze(0).unsqueeze(0)  # [batch_size, 1, height*width]
        
        x = self.conv1(x)
        x = self.relu1(x)
        
        # 智能池化策略：根据维度大小选择合适的池化方式
        if x.shape[2] < 2 or x.shape[3] < 2:
            # 维度太小时使用自适应池化，保证至少有1个输出
            target_h = max(1, x.shape[2] // 2)
            target_w = max(1, x.shape[3] // 2)
            x = nn.AdaptiveMaxPool2d((target_h, target_w))(x)
        else:
            # 维度足够时使用普通池化（效率更高）
            x = self.pool1(x)
            
        x = self.conv2(x)
        x = self.relu2(x)
        
        # 第二次池化使用相同策略
        if x.shape[2] < 2 or x.shape[3] < 2:
            target_h = max(1, x.shape[2] // 2)
            target_w = max(1, x.shape[3] // 2)
            x = nn.AdaptiveMaxPool2d((target_h, target_w))(x)
        else:
            x = self.pool2(x)
            
        # 展平 + 全连接
        x = x.reshape(x.size(0), -1)
        
        # 动态初始化全连接层
        if self.fc1 is None:
            self._initialize_fc_layers(x.shape[1])
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d_1layer(nn.Module):
    """单层卷积的TBM模型 - 用于参数敏感性分析"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d_1layer, self).__init__()
        len_window, feature_dim = input_shape 
        
        # 只有一层卷积
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小 - 只经过一次池化
        conv_output_size = (len_window // 2) * 32
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 处理输入维度
        # 检查是否是4D输入 [batch, channels, height, width]
        if len(x.shape) == 4:
            batch_size, channels, _, seq_len = x.shape
            # 重塑为3D输入 [batch, channels, seq_len]
            x = x.reshape(batch_size, channels, seq_len)
        
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
        
        # 单个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d(nn.Module):
    """双层卷积的TBM模型 - 用于参数敏感性分析（原始版本）"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d, self).__init__()
        len_window, feature_dim = input_shape 
        
        # Layer 1
        ''''output_length = (1024 + 2*2 - 5) // 1 + 1
              = (1028 - 5) // 1 + 1
              = 1023 + 1
              = 1024'''
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小
        conv_output_size = (len_window // 4) * 32  # 经过两次池化层，长度变为原来的1/4
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 处理输入维度
        # 检查是否是4D输入 [batch, channels, height, width]
        if len(x.shape) == 4:
            batch_size, channels, _, seq_len = x.shape
            # 重塑为3D输入 [batch, channels, seq_len]
            x = x.reshape(batch_size, channels, seq_len)
        
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:  # 如果输入是[batch, len_window, feature_dim]
            x = x.transpose(1, 2)  # 转换为[batch, feature_dim, len_window]
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d_3layer(nn.Module):
    """三层卷积的TBM模型 - 用于参数敏感性分析"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d_3layer, self).__init__()
        len_window, feature_dim = input_shape 
        
        # Layer 1
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小 - 经过三次池化
        conv_output_size = (len_window // 8) * 64
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 处理输入维度
        # 检查是否是4D输入 [batch, channels, height, width]
        if len(x.shape) == 4:
            batch_size, channels, _, seq_len = x.shape
            # 重塑为3D输入 [batch, channels, seq_len]
            x = x.reshape(batch_size, channels, seq_len)
        
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d_4layer(nn.Module):
    """四层卷积的TBM模型 - 用于参数敏感性分析"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d_4layer, self).__init__()
        len_window, feature_dim = input_shape 
        
        # Layer 1
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 4
        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小 - 经过四次池化
        conv_output_size = (len_window // 16) * 64
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 处理输入维度
        # 检查是否是4D输入 [batch, channels, height, width]
        if len(x.shape) == 4:
            batch_size, channels, _, seq_len = x.shape
            # 重塑为3D输入 [batch, channels, seq_len]
            x = x.reshape(batch_size, channels, seq_len)
        
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 第四个卷积块
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d_5layer(nn.Module):
    """五层卷积的TBM模型 - 用于参数敏感性分析"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d_5layer, self).__init__()
        len_window, feature_dim = input_shape 
        
        # Layer 1
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 4
        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 5
        self.conv5 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小 - 经过五次池化
        conv_output_size = (len_window // 32) * 128
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 处理输入维度
        # 检查是否是4D输入 [batch, channels, height, width]
        if len(x.shape) == 4:
            batch_size, channels, _, seq_len = x.shape
            # 重塑为3D输入 [batch, channels, seq_len]
            x = x.reshape(batch_size, channels, seq_len)
        
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 第四个卷积块
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # 第五个卷积块
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d_6layer(nn.Module):
    """六层卷积的TBM模型 - 用于参数敏感性分析"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d_6layer, self).__init__()
        len_window, feature_dim = input_shape 
        
        # Layer 1
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 4
        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 5
        self.conv5 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 6
        self.conv6 = nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小 - 经过六次池化
        conv_output_size = (len_window // 64) * 128
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu7 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 处理输入维度
        # 检查是否是4D输入 [batch, channels, height, width]
        if len(x.shape) == 4:
            batch_size, channels, _, seq_len = x.shape
            # 重塑为3D输入 [batch, channels, seq_len]
            x = x.reshape(batch_size, channels, seq_len)
        
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 第四个卷积块
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # 第五个卷积块
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        # 第六个卷积块
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu7(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d_7layer(nn.Module):
    """七层卷积的TBM模型 - 用于参数敏感性分析"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d_7layer, self).__init__()
        len_window, feature_dim = input_shape 
        
        # Layer 1
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 4
        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 5
        self.conv5 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 6
        self.conv6 = nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 7
        self.conv7 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小 - 经过七次池化
        conv_output_size = (len_window // 128) * 256
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu8 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 处理输入维度
        # 检查是否是4D输入 [batch, channels, height, width]
        if len(x.shape) == 4:
            batch_size, channels, _, seq_len = x.shape
            # 重塑为3D输入 [batch, channels, seq_len]
            x = x.reshape(batch_size, channels, seq_len)
        
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 第四个卷积块
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # 第五个卷积块
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        # 第六个卷积块
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        
        # 第七个卷积块
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.pool7(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu8(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d_8layer(nn.Module):
    """八层卷积的TBM模型 - 用于参数敏感性分析"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d_8layer, self).__init__()
        len_window, feature_dim = input_shape 
        
        # Layer 1
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 4
        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 5
        self.conv5 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 6
        self.conv6 = nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 7
        self.conv7 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 8
        self.conv8 = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2)
        self.relu8 = nn.ReLU()
        self.pool8 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小 - 经过八次池化
        conv_output_size = (len_window // 256) * 256
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu9 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 处理输入维度
        # 检查是否是4D输入 [batch, channels, height, width]
        if len(x.shape) == 4:
            batch_size, channels, _, seq_len = x.shape
            # 重塑为3D输入 [batch, channels, seq_len]
            x = x.reshape(batch_size, channels, seq_len)
        
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 第四个卷积块
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # 第五个卷积块
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        # 第六个卷积块
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        
        # 第七个卷积块
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.pool7(x)
        
        # 第八个卷积块
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.pool8(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu9(x)
        x = self.fc2(x)
        
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResNet32_1D(nn.Module):
    def __init__(self, input_shape, output_dim=2):
        super(ResNet32_1D, self).__init__()
        len_window, feature_dim = input_shape
        input_channels = feature_dim
        seq_length = len_window
        num_classes = output_dim
        
        self.in_channels = 16
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 16, 5, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 5, stride=2)
        
        self.seq_length = seq_length // 4 
        self.avgpool = nn.AdaptiveAvgPool1d(self.seq_length)
        self.fc = nn.Linear(64 * self.seq_length, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 处理输入维度
        if len(x.shape) == 4:
            batch_size, channels, _, seq_len = x.shape
            x = x.reshape(batch_size, channels, seq_len)
        
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_shape, output_dim=2):
        super(LSTM, self).__init__()
        len_window, feature_dim = input_shape
        input_size = feature_dim
        hidden_size = 64
        num_layers = 2
        num_classes = output_dim
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, bidirectional=False
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 处理输入维度: [64, 3, 1, 1024] -> [64, 1024, 3]
        if len(x.shape) == 4:
            batch_size, channels, height, width = x.shape
            x = x.squeeze(2).permute(0, 2, 1)  # [64, 3, 1024] -> [64, 1024, 3]
        elif len(x.shape) == 3 and x.shape[1] == 3:
            x = x.permute(0, 2, 1)  # [batch, 3, seq_len] -> [batch, seq_len, 3]
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_shape, output_dim=2):
        super(BiLSTM, self).__init__()
        len_window, feature_dim = input_shape
        input_size = feature_dim
        hidden_size = 64
        num_layers = 2
        num_classes = output_dim
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, bidirectional=True
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 因为双向
        
    def forward(self, x):
        # 处理输入维度: [64, 3, 1, 1024] -> [64, 1024, 3]
        if len(x.shape) == 4:
            batch_size, channels, height, width = x.shape
            x = x.squeeze(2).permute(0, 2, 1)  # [64, 3, 1024] -> [64, 1024, 3]
        elif len(x.shape) == 3 and x.shape[1] == 3:
            x = x.permute(0, 2, 1)  # [batch, 3, seq_len] -> [batch, seq_len, 3]
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播BiLSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, input_shape, output_dim=2):
        super(Transformer, self).__init__()
        len_window, feature_dim = input_shape
        input_dim = feature_dim
        seq_length = len_window
        num_classes = output_dim
        d_model = 256
        nhead = 2
        num_encoder_layers = 3
        dim_feedforward = 512
        dropout = 0.1
        
        # 将输入特征映射到模型维度
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_length)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.d_model = d_model
        self.seq_length = seq_length
        
    def forward(self, x):
        # 处理输入维度: [64, 3, 1, 1024] -> [64, 1024, 3]
        if len(x.shape) == 4:
            batch_size, channels, height, width = x.shape
            x = x.squeeze(2).permute(0, 2, 1)  # [64, 3, 1024] -> [64, 1024, 3]
        elif len(x.shape) == 3 and x.shape[1] == 3:
            x = x.permute(0, 2, 1)  # [batch, 3, seq_len] -> [batch, seq_len, 3]
        
        # x shape: [batch_size, seq_length, features]
        # 映射到模型维度
        x = self.embedding(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 使用序列的平均值进行分类
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # 分类层
        x = self.classifier(x)
        return x