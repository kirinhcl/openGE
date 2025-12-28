"""
基于梯度的可解释性方法 (Gradient-based Interpretability Methods)

实现多种基于梯度的模型解释方法:
    - 积分梯度 (Integrated Gradients)
    - 显著性图 (Saliency Maps)
    - SmoothGrad
    - Gradient × Input
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union


class GradientExplainer:
    """
    基于梯度的模型解释器。
    
    提供多种梯度方法来解释模型预测。
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        初始化梯度解释器。
        
        Args:
            model: 训练好的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def integrated_gradients(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        target: int = 0,
        steps: int = 50,
        baseline: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        计算积分梯度。
        
        积分梯度是一种满足公理性质的归因方法，通过对从基线到输入的
        路径上的梯度进行积分来计算每个特征的归因。
        
        Args:
            inputs: 输入张量 [batch_size, n_features] 或 [n_features]
            target: 目标类别/输出索引
            steps: 积分步数
            baseline: 基线输入 (默认为零向量)
            
        Returns:
            归因分数
        """
        # 转换输入
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(self.device)
        
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        
        # 设置基线
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        elif isinstance(baseline, np.ndarray):
            baseline = torch.tensor(baseline, dtype=torch.float32)
        baseline = baseline.to(self.device)
        
        if baseline.dim() == 1:
            baseline = baseline.unsqueeze(0)
        
        # 生成插值路径
        scaled_inputs = [
            baseline + (float(i) / steps) * (inputs - baseline)
            for i in range(steps + 1)
        ]
        
        # 计算每个步骤的梯度
        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.clone().detach().requires_grad_(True)
            
            output = self.model(scaled_input)
            if isinstance(output, tuple):
                output = output[0]
            
            # 处理目标输出
            if output.dim() > 1 and output.shape[1] > 1:
                output = output[:, target]
            
            output.sum().backward()
            gradients.append(scaled_input.grad.detach().clone())
        
        # 积分 (使用梯形法则)
        gradients = torch.stack(gradients, dim=0)
        avg_gradients = (gradients[:-1] + gradients[1:]) / 2
        integrated_gradients = avg_gradients.mean(dim=0)
        
        # 归因 = 积分梯度 × (输入 - 基线)
        attributions = integrated_gradients * (inputs - baseline)
        
        return attributions.detach()
    
    def saliency_map(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        target: int = 0,
        absolute: bool = True
    ) -> np.ndarray:
        """
        计算基于梯度的显著性图。
        
        显著性图显示输入的微小变化如何影响输出。
        
        Args:
            inputs: 输入张量
            target: 目标类别/输出索引
            absolute: 是否取绝对值
            
        Returns:
            显著性图
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(self.device)
        
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        
        inputs = inputs.clone().detach().requires_grad_(True)
        
        output = self.model(inputs)
        if isinstance(output, tuple):
            output = output[0]
        
        if output.dim() > 1 and output.shape[1] > 1:
            output = output[:, target]
        
        output.sum().backward()
        saliency = inputs.grad.detach()
        
        if absolute:
            saliency = torch.abs(saliency)
        
        return saliency.cpu().numpy()
    
    def smoothgrad(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        n_samples: int = 50,
        noise_level: float = 0.1,
        target: int = 0
    ) -> np.ndarray:
        """
        计算平滑梯度以获得更鲁棒的归因。
        
        通过在输入周围添加噪声并平均梯度来减少噪声。
        
        Args:
            inputs: 输入张量
            n_samples: 噪声样本数
            noise_level: 噪声标准差
            target: 目标类别/输出索引
            
        Returns:
            平滑后的梯度归因
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(self.device)
        
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        
        # 计算噪声标准差
        stdev = noise_level * (inputs.max() - inputs.min()).item()
        if stdev == 0:
            stdev = noise_level
        
        all_gradients = []
        
        for _ in range(n_samples):
            # 添加噪声
            noise = torch.randn_like(inputs) * stdev
            noisy_input = (inputs + noise).clone().detach().requires_grad_(True)
            
            output = self.model(noisy_input)
            if isinstance(output, tuple):
                output = output[0]
            
            if output.dim() > 1 and output.shape[1] > 1:
                output = output[:, target]
            
            output.sum().backward()
            all_gradients.append(noisy_input.grad.detach().clone())
        
        # 平均梯度
        smoothgrad = torch.stack(all_gradients, dim=0).mean(dim=0)
        
        return smoothgrad.cpu().numpy()
    
    def gradient_times_input(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        target: int = 0
    ) -> np.ndarray:
        """
        计算 Gradient × Input。
        
        Args:
            inputs: 输入张量
            target: 目标类别/输出索引
            
        Returns:
            Gradient × Input 归因
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(self.device)
        
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        
        inputs_for_grad = inputs.clone().detach().requires_grad_(True)
        
        output = self.model(inputs_for_grad)
        if isinstance(output, tuple):
            output = output[0]
        
        if output.dim() > 1 and output.shape[1] > 1:
            output = output[:, target]
        
        output.sum().backward()
        gradient = inputs_for_grad.grad.detach()
        
        # Gradient × Input
        attributions = gradient * inputs
        
        return attributions.cpu().numpy()
    
    def guided_backprop(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        target: int = 0
    ) -> np.ndarray:
        """
        计算引导反向传播 (Guided Backpropagation)。
        
        仅反向传播正梯度通过正激活。
        
        Args:
            inputs: 输入张量
            target: 目标类别/输出索引
            
        Returns:
            引导反向传播结果
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(self.device)
        
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        
        # 注册钩子以修改 ReLU 的反向传播
        handles = []
        
        def guided_relu_hook(module, grad_in, grad_out):
            if isinstance(grad_in[0], torch.Tensor):
                return (torch.clamp(grad_in[0], min=0.0),)
            return grad_in
        
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                handle = module.register_full_backward_hook(guided_relu_hook)
                handles.append(handle)
        
        try:
            inputs = inputs.clone().detach().requires_grad_(True)
            
            output = self.model(inputs)
            if isinstance(output, tuple):
                output = output[0]
            
            if output.dim() > 1 and output.shape[1] > 1:
                output = output[:, target]
            
            output.sum().backward()
            guided_grads = inputs.grad.detach()
            
        finally:
            # 移除钩子
            for handle in handles:
                handle.remove()
        
        return guided_grads.cpu().numpy()
    
    def compare_methods(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        methods: Optional[list] = None,
        target: int = 0
    ) -> Dict:
        """
        比较多种梯度方法的结果。
        
        Args:
            inputs: 输入特征
            methods: 要比较的方法列表
            target: 目标类别/输出索引
            
        Returns:
            各方法的归因结果
        """
        if methods is None:
            methods = ['saliency', 'integrated_gradients', 'smoothgrad', 
                       'gradient_times_input']
        
        results = {}
        
        if 'saliency' in methods:
            results['saliency'] = self.saliency_map(inputs, target)
        
        if 'integrated_gradients' in methods:
            ig = self.integrated_gradients(inputs, target)
            results['integrated_gradients'] = ig.cpu().numpy()
        
        if 'smoothgrad' in methods:
            results['smoothgrad'] = self.smoothgrad(inputs, target=target)
        
        if 'gradient_times_input' in methods:
            results['gradient_times_input'] = self.gradient_times_input(inputs, target)
        
        if 'guided_backprop' in methods:
            results['guided_backprop'] = self.guided_backprop(inputs, target)
        
        # 计算方法间的相关性
        method_names = list(results.keys())
        n_methods = len(method_names)
        correlation_matrix = np.zeros((n_methods, n_methods))
        
        for i, m1 in enumerate(method_names):
            for j, m2 in enumerate(method_names):
                v1 = results[m1].flatten()
                v2 = results[m2].flatten()
                if len(v1) > 1:
                    correlation_matrix[i, j] = np.corrcoef(v1, v2)[0, 1]
                else:
                    correlation_matrix[i, j] = 1.0 if i == j else 0.0
        
        results['correlation_matrix'] = correlation_matrix
        results['method_names'] = method_names
        
        return results
