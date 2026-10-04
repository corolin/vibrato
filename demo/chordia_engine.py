"""
Chordia 推理引擎 - 高精度情感动力学推理系统
Author: Chordia Team
Version: 1.0
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Union, Tuple

# 配置日志（从环境变量读取日志级别）
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Chordia")


class ChordiaModel(nn.Module):
    """
    Chordia MLP 模型架构定义
    结构: 7 -> 512 -> 256 -> 128 -> 3
    """
    def __init__(self, input_dim: int = 7, output_dim: int = 3):
        super(ChordiaModel, self).__init__()

        self.network = nn.Sequential(
            # Layer 1: 7 -> 512
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 4: 128 -> 3
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class ChordiaEngine:
    """
    Chordia 高精度情感动力学推理引擎

    封装了基于 MLP 的 PAD 增量预测逻辑、人格缩放以及压力值派生计算。

    Attributes:
        model: 神经网络模型
        device: 计算设备 (CPU/GPU)
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化引擎并加载模型权重

        Args:
            model_path: 模型权重文件 (.pth) 路径
            device: 指定计算设备 ('cuda', 'cpu', 或 'auto')
        """
        # 1. 设备选择策略
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2. 初始化模型架构
        self.model = ChordiaModel(input_dim=7, output_dim=3)

        # 3. 加载权重
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # 提取 model_state_dict 并加载
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(state_dict)
                logger.info(f"成功加载模型状态字典 (epoch: {checkpoint.get('epoch', 'N/A')})")
            else:
                # 兼容直接保存的 state_dict
                self.model.load_state_dict(checkpoint)
                logger.info("成功加载模型状态字典")

            self.model.eval()
            logger.info(f"Chordia engine initialized on {self.device}")

            # 打印模型信息
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"模型总参数量: {total_params:,}")

        except Exception as e:
            logger.error(f"Failed to load Chordia model from {model_path}: {e}")
            raise

    def predict(
        self,
        user_pad: List[float],  # 用户 PAD 情感状态，包含愉悦度(P)、激活度(A)和支配度(D)三个维度
        vitality: float,
        current_pad: List[float],
        personality_scale: float = 1.0
    ) -> Dict[str, Union[List[float], Dict[str, float]]]:
        """
        执行前向推断并返回处理后的情感指标

        Args:
            user_pad: 用户 PAD 状态 [P, A, D] (范围: [-1, 1])
            vitality: AI 生理活力值 (范围: [0, 100])
            current_pad: AI 当前 PAD 状态 [P, A, D] (范围: [-1, 1])
            personality_scale: 人格缩放系数 (默认 1.0)

        Returns:
            包含 delta_pad 和 derived_metrics 的字典:
            {
                "delta_pad": [ΔP, ΔA, ΔD],
                "metrics": {
                    "raw_delta": [ΔP, ΔA, ΔD],
                    "scale_factor": float
                }
            }
        """
        # 1. 输入验证
        if len(user_pad) != 3 or len(current_pad) != 3:
            raise ValueError("user_pad 和 current_pad 必须是长度为 3 的列表 [P, A, D]")

        if not (0 <= vitality <= 100):
            logger.warning(f"vitality 值 {vitality} 超出常规范围 [0, 100]，建议检查输入")

        # 2. 数据预处理与归一化
        # 将 vitality 从 [0, 100] 线性映射至 [-1, 1] 以匹配 MLP 输入分布
        norm_vitality = (vitality / 50.0) - 1.0

        # 构造 7 维特征向量: [user_P, user_A, user_D, norm_vitality, ai_P, ai_A, ai_D]
        input_feature = user_pad + [norm_vitality] + current_pad

        # 转换为 Tensor 并增加 batch 维度
        input_tensor = torch.FloatTensor(input_feature).to(self.device).unsqueeze(0)

        # 3. 模型推理 (禁用梯度计算以优化性能)
        with torch.no_grad():
            raw_output = self.model(input_tensor).cpu().numpy().flatten()

        # 4. 执行人格缩放 (Personality Scaling)
        # 这一步决定了情感响应的烈度
        scaled_delta_pad = raw_output * personality_scale
        dp, da, dd = scaled_delta_pad

        # 5. 计算置信度 (Confidence)
        # 基于原始输出的幅度计算置信度：输出绝对值越大，模型越确信
        # 使用 sigmoid 函数将幅度映射到 [0, 1] 区间
        output_magnitude = np.mean(np.abs(raw_output))
        confidence = float(1.0 / (1.0 + np.exp(-5.0 * (output_magnitude - 0.1))))

        # 6. 输出日志
        logger.debug(f"Chordia 预测:")
        logger.debug(f"  原始输出: {raw_output.tolist()}")
        logger.debug(f"  缩放后 ΔPAD: [{dp:.4f}, {da:.4f}, {dd:.4f}]")
        logger.debug(f"  置信度: {confidence:.2%}")

        return {
            "delta_pad": [float(x) for x in scaled_delta_pad],
            "metrics": {
                "raw_delta": [float(x) for x in raw_output],
                "scale_factor": float(personality_scale),
                "confidence": float(confidence),
                "output_magnitude": float(output_magnitude)
            }
        }
