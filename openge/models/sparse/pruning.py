"""Model pruning and sparsification utilities."""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Tuple, Optional
import numpy as np


class ModelPruner:
    """
    Utilities for model pruning and sparsification.
    
    Supports various pruning methods for reducing model size
    and improving interpretability.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize pruner.
        
        Args:
            model: PyTorch model to prune
        """
        self.model = model
        self.pruning_history = []
    
    def magnitude_pruning(
        self, 
        sparsity_level: float,
        prune_bias: bool = False,
        layers_to_prune: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Prune weights by magnitude (unstructured pruning).
        
        Args:
            sparsity_level: Target sparsity level (0-1)
            prune_bias: Whether to prune bias terms
            layers_to_prune: List of layer names to prune (None = all)
            
        Returns:
            Dictionary with sparsity achieved per layer
        """
        sparsity_report = {}
        
        for name, module in self.model.named_modules():
            if layers_to_prune is not None and name not in layers_to_prune:
                continue
            
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Prune weights
                prune.l1_unstructured(module, name='weight', amount=sparsity_level)
                
                # Calculate actual sparsity
                weight = module.weight
                total = weight.numel()
                zeros = (weight == 0).sum().item()
                sparsity_report[f"{name}.weight"] = zeros / total
                
                # Prune bias if requested
                if prune_bias and module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=sparsity_level)
                    bias = module.bias
                    sparsity_report[f"{name}.bias"] = (bias == 0).sum().item() / bias.numel()
        
        self.pruning_history.append({
            'method': 'magnitude',
            'sparsity_level': sparsity_level,
            'report': sparsity_report
        })
        
        return sparsity_report
    
    def structured_pruning(
        self, 
        sparsity_level: float,
        dim: int = 0,
        norm: int = 1
    ) -> Dict[str, float]:
        """
        Perform structured pruning (channels/filters).
        
        Args:
            sparsity_level: Target sparsity level
            dim: Dimension to prune along (0=output, 1=input)
            norm: Norm type for importance scoring
            
        Returns:
            Dictionary with sparsity info per layer
        """
        sparsity_report = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                prune.ln_structured(
                    module, 
                    name='weight', 
                    amount=sparsity_level, 
                    n=norm, 
                    dim=dim
                )
                
                weight = module.weight
                # Count zeroed channels/filters
                if dim == 0:
                    channel_norms = weight.view(weight.size(0), -1).norm(dim=1)
                else:
                    channel_norms = weight.view(-1, weight.size(-1)).norm(dim=0)
                
                zeros = (channel_norms == 0).sum().item()
                total = channel_norms.numel()
                sparsity_report[f"{name}"] = zeros / total
        
        self.pruning_history.append({
            'method': 'structured',
            'sparsity_level': sparsity_level,
            'report': sparsity_report
        })
        
        return sparsity_report
    
    def random_pruning(self, sparsity_level: float) -> Dict[str, float]:
        """
        Random unstructured pruning (baseline).
        
        Args:
            sparsity_level: Target sparsity level
            
        Returns:
            Dictionary with sparsity info per layer
        """
        sparsity_report = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                prune.random_unstructured(module, name='weight', amount=sparsity_level)
                
                weight = module.weight
                total = weight.numel()
                zeros = (weight == 0).sum().item()
                sparsity_report[f"{name}.weight"] = zeros / total
        
        return sparsity_report
    
    def global_pruning(
        self, 
        sparsity_level: float,
        importance_fn: str = 'magnitude'
    ) -> Dict[str, float]:
        """
        Global pruning across all layers based on global importance.
        
        Args:
            sparsity_level: Target global sparsity level
            importance_fn: Importance scoring function ('magnitude', 'gradient')
            
        Returns:
            Dictionary with sparsity info per layer
        """
        # Collect all prunable parameters
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            return {}
        
        # Apply global pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity_level
        )
        
        return self.get_sparsity_report()
    
    def iterative_pruning(
        self,
        target_sparsity: float,
        num_iterations: int = 10,
        fine_tune_fn: Optional[callable] = None
    ) -> List[Dict[str, float]]:
        """
        Iterative pruning with optional fine-tuning between iterations.
        
        Args:
            target_sparsity: Final target sparsity
            num_iterations: Number of pruning iterations
            fine_tune_fn: Optional function to fine-tune after each iteration
            
        Returns:
            List of sparsity reports for each iteration
        """
        reports = []
        sparsity_per_iter = 1 - (1 - target_sparsity) ** (1 / num_iterations)
        
        for i in range(num_iterations):
            # Prune
            report = self.magnitude_pruning(sparsity_per_iter)
            reports.append(report)
            
            # Fine-tune if provided
            if fine_tune_fn is not None:
                fine_tune_fn(self.model, iteration=i)
        
        return reports
    
    def make_pruning_permanent(self) -> None:
        """Remove pruning reparameterization and make weights permanent."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass  # Not pruned
    
    def get_sparsity_report(self) -> Dict[str, float]:
        """
        Get sparsity statistics for each layer.
        
        Returns:
            Dictionary with sparsity info per layer
        """
        report = {}
        total_params = 0
        total_zeros = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weight = module.weight
                total = weight.numel()
                zeros = (weight == 0).sum().item()
                
                report[name] = {
                    'total_params': total,
                    'zero_params': zeros,
                    'sparsity': zeros / total if total > 0 else 0
                }
                
                total_params += total
                total_zeros += zeros
        
        report['global'] = {
            'total_params': total_params,
            'zero_params': total_zeros,
            'sparsity': total_zeros / total_params if total_params > 0 else 0
        }
        
        return report
    
    def get_importance_scores(
        self, 
        method: str = 'magnitude'
    ) -> Dict[str, torch.Tensor]:
        """
        Get importance scores for each parameter.
        
        Args:
            method: Scoring method ('magnitude', 'gradient', 'taylor')
            
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        importance_scores = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if method == 'magnitude':
                    scores = param.abs()
                elif method == 'gradient' and param.grad is not None:
                    scores = param.grad.abs()
                elif method == 'taylor' and param.grad is not None:
                    scores = (param * param.grad).abs()
                else:
                    scores = param.abs()
                
                importance_scores[name] = scores.detach().cpu()
        
        return importance_scores
    
    def compute_flops_reduction(self) -> float:
        """
        Estimate FLOPs reduction from sparsity.
        
        Returns:
            Estimated FLOPs reduction ratio
        """
        report = self.get_sparsity_report()
        global_sparsity = report.get('global', {}).get('sparsity', 0)
        
        # Approximate: FLOPs reduction ≈ weight sparsity
        # (actual reduction depends on hardware support)
        return global_sparsity


class GradientBasedPruner(ModelPruner):
    """
    Pruner that uses gradient information for importance scoring.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.gradient_scores = {}
    
    def accumulate_gradients(self, dataloader, loss_fn, num_batches: int = 10):
        """
        Accumulate gradient statistics over multiple batches.
        
        Args:
            dataloader: Data loader for computing gradients
            loss_fn: Loss function
            num_batches: Number of batches to accumulate
        """
        self.model.train()
        
        # Initialize gradient accumulators
        gradient_sums = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                gradient_sums[name] = torch.zeros_like(param)
        
        # Accumulate gradients
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            self.model.zero_grad()
            
            # Forward pass
            if isinstance(batch, dict):
                outputs = self.model(**batch)
            else:
                outputs = self.model(*batch[:-1])
            
            # Compute loss and backward
            loss = loss_fn(outputs, batch[-1])
            loss.backward()
            
            # Accumulate
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradient_sums[name] += param.grad.abs()
        
        # Average and store
        for name in gradient_sums:
            self.gradient_scores[name] = gradient_sums[name] / num_batches
    
    def gradient_pruning(self, sparsity_level: float) -> Dict[str, float]:
        """
        Prune based on gradient importance scores.
        
        Args:
            sparsity_level: Target sparsity level
            
        Returns:
            Sparsity report
        """
        if not self.gradient_scores:
            raise ValueError("Must call accumulate_gradients first")
        
        # Collect all importance scores
        all_scores = []
        for name in self.gradient_scores:
            all_scores.append(self.gradient_scores[name].flatten())
        
        all_scores = torch.cat(all_scores)
        
        # Find threshold
        k = int(sparsity_level * all_scores.numel())
        if k > 0:
            threshold = torch.kthvalue(all_scores, k).values.item()
        else:
            threshold = 0
        
        # Apply masks
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                param_name = f"{name}.weight"
                if param_name in self.gradient_scores:
                    mask = (self.gradient_scores[param_name] >= threshold).float()
                    with torch.no_grad():
                        module.weight.mul_(mask)
        
        return self.get_sparsity_report()
