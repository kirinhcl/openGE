# OpenGE: Multi-Crop Trait Prediction with G×E Interactions

OpenGE is a deep learning library designed for **crop trait prediction** using **Genotype (G) × Environment (E) interaction modeling**. The library is built to be crop-agnostic and supports multiple crops with minimal configuration changes.

## Features

✨ **Core Capabilities**
- **G×E Interaction Modeling**: Explicitly captures how genetic and environmental factors interact to influence crop traits
- **Multi-Crop Support**: Easily adapt to different crops through configuration files
- **Modular Architecture**: Flexible combination of encoders (MLP, CNN, Transformer, CNN+Transformer) and fusion strategies
- **Production-Ready**: Complete training, prediction, and evaluation pipelines with progress bars
- **Comprehensive Logging**: Detailed logs with hyperparameters, model architecture, and training history

🧬 **Genetic Encoders**
- **MLP Encoder**: Fast and effective for dense SNP data
- **CNN Encoder**: Captures local linkage patterns in genetic markers
- **Transformer Encoder**: Models long-range dependencies across the genome
- **CNN+Transformer Hybrid**: Combines local pattern detection with global context

🌍 **Environment Encoders**
- **MLP Encoder**: For weather, soil, and environmental covariate (EC) data
- Supports multi-source environmental data integration

🔗 **Fusion Strategies**
- **Attention Fusion**: Learns dynamic weighting of genetic vs environmental contributions
- **Concatenation Fusion**: Simple concatenation with projection

🔍 **Interpretability**
- **Attention Analysis**: Visualize which genetic markers and environmental factors matter most
- **Feature Importance**: Quantify relative contribution of genetic vs environmental factors
- **Gradient-Based Explanations**: Integrated gradients, saliency maps, and SmoothGrad
- **SHAP Explainer**: Game-theoretic model explanations
- **Sparsity Analysis**: Analyze patterns in pruned weight-sparse models

⚡ **Advanced Models**
- **Weight-Sparse Transformers**: Interpretable sparse attention mechanisms
- **Model Pruning**: Magnitude and structured pruning utilities
- **Top-K Attention**: Efficient sparse attention for large genomic datasets

## Installation

### From source (development)
```bash
git clone https://github.com/kirinhcl/openGE.git
cd openGE
pip install -e .
```

### With optional dependencies
```bash
# For interpretability features
pip install -e ".[interpretability]"

# For all features
pip install -e ".[all]"
```

## Quick Start

### 1. Training

Train a model with a configuration file:
```bash
python train.py --config configs/maize_2024.yaml --data-dir ./Training_data
```

Or with custom hyperparameters:
```bash
python train.py --data-dir ./Training_data --epochs 50 --batch-size 64 --lr 0.0005
```

Training outputs are saved to `outputs/run_YYYYMMDD_HHMMSS/` including:
- `best_model.pt` - Model checkpoint with weights and configuration
- `results.json` - Complete results with hyperparameters and architecture
- `training.log` - Detailed training logs
- `test_predictions.npz` - Test set predictions

### 2. Prediction

Make predictions with a trained model:
```bash
python predict.py --model-path outputs/run_xxx/best_model.pt --data-dir ./Testing_data
```

Custom output directory:
```bash
python predict.py --model-path outputs/run_xxx/best_model.pt --data-dir ./Testing_data --output-dir ./my_predictions
```

Predictions are saved to `predictions/pred_YYYYMMDD_HHMMSS/` including:
- `predictions.csv` - Detailed predictions with sample IDs
- `submission.csv` - Competition-ready submission format
- `evaluation_metrics.json` - Performance metrics (if targets available)
- `predictions.npz` - Raw numpy arrays

### 3. Interpretability Analysis

```bash
python interpret.py --model-path outputs/run_xxx/best_model.pt --config configs/maize_2024.yaml --data-dir ./data
```

## Project Structure

```
openge/
├── openge/                    # Main package
│   ├── data/                 # Data loading and preprocessing
│   │   ├── dataset.py       # Generic dataset loader
│   │   ├── preprocess.py    # Preprocessing utilities
│   │   └── loaders/         # Crop-specific loaders
│   │       ├── genetic.py   # SNP/genetic data
│   │       ├── environment.py  # Weather, soil, EC data
│   │       └── phenotype.py # Trait/target data
│   ├── models/              # Model architectures
│   │   ├── encoders.py      # CNN, Transformer, MLP encoders
│   │   ├── fusion.py        # Fusion layers (Concat, Attention, Gating)
│   │   ├── heads.py         # Prediction heads
│   │   ├── gxe.py           # G×E interaction model
│   │   └── sparse/          # Sparse model variants
│   ├── core/                # Training and configuration
│   │   ├── engine.py        # Trainer class
│   │   ├── config.py        # Configuration management
│   │   └── registry.py      # Component registry
│   ├── utils/               # Utilities
│   │   ├── metrics.py       # Evaluation metrics
│   │   └── visualization.py # Plotting functions
│   └── interpretability/    # Explainability modules
│       ├── attention_analysis.py
│       ├── feature_importance.py
│       ├── gradient_methods.py
│       ├── shap_explainer.py
│       └── sparsity_analysis.py
├── configs/                 # Configuration files
│   ├── base.yaml           # Base configuration template
│   ├── maize_2024.yaml     # Maize-specific config
│   └── wheat_2025.yaml     # Wheat-specific config
├── tests/                   # Unit tests
├── train.py                # Main training script
├── predict.py              # Prediction script
├── interpret.py            # Interpretability script
├── setup.py                # Package setup
└── README.md              # This file
```

## Configuration

### Available Configurations
- `base.yaml`: Base configuration template with all available options
- `maize_2024.yaml`: MLP encoder optimized for maize (default)
- `maize_transformer.yaml`: Transformer encoder for maize
- `maize_cnn_transformer.yaml`: Hybrid CNN+Transformer encoder for maize
- `sparse_transformer.yaml`: Weight-sparse transformer for interpretability
- `wheat_2025.yaml`: Configuration for wheat trait prediction

### Crop-Specific Configuration Example
```yaml
crop: "maize"
year: 2024

model:
  # Choose encoder: mlp, cnn, transformer, or cnn_transformer
  genetic_encoder: "mlp"
  genetic_hidden_dim: 256           # Output dimension
  genetic_hidden_dims: [1024, 512, 256]  # MLP internal layers
  
  env_hidden_dim: 128
  env_hidden_dims: [256, 128]
  
  fusion: "attention"               # or "concat"
  head_hidden_dims: [64, 32]
  dropout: 0.2

training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.0005
  weight_decay: 0.0001
  early_stopping_patience: 20
  train_ratio: 0.7
  val_ratio: 0.15

data:
  target_traits: ["Yield_Mg_ha"]
  genetic_missing_threshold: 0.5
  normalization: "standard"
```

### CNN+Transformer Hybrid Example
```yaml
model:
  genetic_encoder: "cnn_transformer"
  genetic_hidden_channels: [64, 128, 256]  # CNN channels
  genetic_kernel_sizes: [7, 5, 3]          # CNN kernels
  cnn_output_dim: 512                      # CNN → Transformer dimension
  genetic_hidden_dim: 256                  # Final output dimension
  genetic_n_heads: 8                       # Transformer heads
  genetic_n_layers: 2                      # Transformer layers
```

## Model Architectures

### G×E Model
The core model combines four components:
1. **Genetic Encoder**: Processes SNP/marker data
   - `mlp`: Multi-layer perceptron (fast, good for dense data)
   - `cnn`: Convolutional neural network (captures local LD patterns)
   - `transformer`: Self-attention mechanism (models long-range dependencies)
   - `cnn_transformer`: Hybrid architecture (local + global patterns)

2. **Environment Encoder**: Processes environmental data (MLP)
   - Weather, soil, and environmental covariate (EC) features
   - Supports multi-source data integration

3. **Fusion Layer**: Combines genetic and environmental representations
   - `attention`: Learned dynamic weighting of G vs E contributions
   - `concat`: Simple concatenation with projection

4. **Prediction Head**: Generates trait predictions
   - Regression head for continuous traits (e.g., yield)
   - Classification head for categorical traits

### Architecture Examples

**MLP Encoder (Default)**
```
Input: 2397 SNP markers
  ↓ Linear(2397 → 1024) + BatchNorm + ReLU + Dropout
  ↓ Linear(1024 → 512) + BatchNorm + ReLU + Dropout  
  ↓ Linear(512 → 256) + BatchNorm + ReLU + Dropout
Output: 256-dim embedding
```

**CNN+Transformer Hybrid**
```
Input: 2397 SNP markers
  ↓ CNN: Conv1D(64, 128, 256 channels) + MaxPool
  ↓ Flatten + Linear → 512-dim
  ↓ Transformer: Multi-head attention (8 heads, 2 layers)
Output: 256-dim embedding
```

**Attention Fusion**
```
Genetic embedding (256-dim) + Environment embedding (128-dim)
  ↓ Cross-attention: Q=genetic, K=env, V=env
  ↓ Residual connection
Output: 256-dim fused representation → Prediction head
```

### Model Sizes
- **MLP (default)**: ~4.4M parameters
- **CNN**: ~2-5M parameters (depending on channels)
- **Transformer**: ~8-15M parameters (depending on layers/heads)
- **CNN+Transformer**: ~82M parameters (hybrid architecture)

## Training

### Command Line Interface

Basic training:
```bash
python train.py --data-dir ./Training_data
```

With configuration file:
```bash
python train.py --config configs/maize_2024.yaml --data-dir ./Training_data
```

Override hyperparameters:
```bash
python train.py --data-dir ./Training_data --epochs 50 --batch-size 64 --lr 0.0005
```

Use GPU:
```bash
python train.py --data-dir ./Training_data --device cuda
```

### Programmatic Usage

```python
from openge.core import Config
from openge.models import GxEModel
import torch.nn as nn

# Load configuration
config = Config("configs/maize_2024.yaml")

# Model is built automatically in train.py
# Or build manually:
from train import build_model, load_data

dataset, preprocessors, data_info = load_data(config.config, data_dir, logger)
model = build_model(config.config, data_info, device="cuda", logger=logger)

# Train
# See train.py for complete training loop with:
# - Progress bars (tqdm)
# - Early stopping
# - Learning rate scheduling
# - Model checkpointing
```

### Output Structure

Training outputs are saved to `outputs/run_YYYYMMDD_HHMMSS/`:
```
outputs/run_20251230_100457/
├── best_model.pt          # Model checkpoint
├── results.json           # Complete results
├── training.log           # Training logs
└── test_predictions.npz   # Test predictions
```

### Results File (`results.json`)

Contains comprehensive training information:
```json
{
  "best_epoch": 45,
  "best_val_r2": 0.5633,
  "test_r2": 0.5758,
  "hyperparameters": {
    "training": {
      "batch_size": 64,
      "learning_rate": 0.0005,
      "optimizer": "AdamW",
      "epochs": 100,
      ...
    },
    "model": {...},
    "data": {...}
  },
  "model_architecture": {
    "total_parameters": 4446017,
    "trainable_parameters": 4446017,
    "genetic_encoder": {...},
    "env_encoder": {...},
    ...
  },
  "history": {
    "train_loss": [...],
    "val_r2": [...],
    ...
  }
}
```

## Making Predictions

### Command Line Interface

Basic prediction:
```bash
python predict.py --model-path outputs/run_xxx/best_model.pt --data-dir ./Testing_data
```

Custom output directory:
```bash
python predict.py \
  --model-path outputs/run_xxx/best_model.pt \
  --data-dir ./Testing_data \
  --output-dir ./my_predictions
```

Custom batch size:
```bash
python predict.py \
  --model-path outputs/run_xxx/best_model.pt \
  --data-dir ./Testing_data \
  --batch-size 128
```

### Output Structure

Predictions are saved to `predictions/pred_YYYYMMDD_HHMMSS/`:
```
predictions/pred_20251230_102258/
├── predictions.csv            # Full predictions with sample IDs
├── submission.csv            # Competition format (Env, Hybrid, Yield_Mg_ha)
├── evaluation_metrics.json   # Metrics (if targets available)
├── predictions.npz           # Raw numpy arrays
└── prediction.log            # Prediction logs
```

### Programmatic Usage

```python
import torch
from predict import load_model, predict

# Load trained model
model, checkpoint = load_model(model_path, device, logger)

# Model automatically reconstructs architecture from checkpoint
# Supports: mlp, cnn, transformer, cnn_transformer encoders

# Make predictions with progress bar
predictions, targets = predict(model, test_loader, device, logger)

# Evaluate
if targets_available:
    from openge.utils.metrics import calculate_rmse, calculate_r2
    rmse = calculate_rmse(targets.flatten(), predictions.flatten())
    r2 = calculate_r2(targets.flatten(), predictions.flatten())
    print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
```

### Evaluation Metrics

When target values are available (e.g., `7_Testing_Observed_Values.csv`), the script automatically calculates:
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **MAE**: Mean Absolute Error
- **Correlation**: Pearson correlation coefficient

## Interpretability Analysis

### Feature Importance
```python
from openge.interpretability import FeatureImportance

fi = FeatureImportance(model)
g_importance, e_importance = fi.genetic_vs_environment_contribution(X_genetic, X_env)
print(f"Genetic: {g_importance:.2%}, Environment: {e_importance:.2%}")
```

### Attention Analysis
```python
from openge.interpretability import AttentionAnalyzer

analyzer = AttentionAnalyzer(model)
attention_heads = analyzer.extract_attention_heads()
head_importance = analyzer.analyze_head_importance()
```

### Gradient-Based Explanations
```python
from openge.interpretability import GradientExplainer

explainer = GradientExplainer(model)
attr = explainer.integrated_gradients(inputs, target=0)
```

## Extending to New Crops

1. **Create crop-specific config** (e.g., `configs/rice_2025.yaml`):
```yaml
crop: "rice"
data:
  genetic_file: "path/to/rice/genotypes.vcf"
  environment_files:
    weather: "path/to/rice/weather.csv"
  phenotype_file: "path/to/rice/phenotypes.csv"
```

2. **Implement crop-specific data loaders** (optional):
```python
from openge.data.loaders import GeneticLoader

class RiceGeneticLoader(GeneticLoader):
    def load_from_vcf(self, filepath):
        # Rice-specific VCF parsing
        pass
```

3. **Train model**:
```bash
python train.py --config configs/rice_2025.yaml --data-dir ./rice_data
```

## Adding New Components

### Register a Custom Encoder
```python
from openge.core import encoder_registry

@encoder_registry.register("custom_encoder")
class CustomEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Implementation
    
    def forward(self, x):
        # Forward pass
        pass
```

### Register a Custom Fusion Layer
```python
from openge.core import fusion_registry

@fusion_registry.register("custom_fusion")
class CustomFusion(nn.Module):
    def forward(self, x1, x2):
        # Fusion logic
        pass
```

## Requirements

Core dependencies:
- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- PyYAML >= 5.4.0
- tqdm >= 4.62.0 (progress bars)

Optional for interpretability:
- SHAP >= 0.40.0 (SHAP explanations)
- Captum >= 0.4.0 (advanced interpretability)
- matplotlib >= 3.4.0 (visualization)

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=openge --cov-report=html
```

## Citation

If you use OpenGE in your research, please cite:

```bibtex
@software{openge2024,
  title = {OpenGE: Open-source library for crop trait prediction with G×E interactions},
  author = {Contributors, OpenGE},
  year = {2024},
  url = {https://github.com/kirinhcl/openGE}
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## References

- Incorporation of genomic information in crop yield prediction models
- Multi-task learning in crop genetics
- Attention mechanisms for genomic prediction
- Sparse neural networks for interpretability

## Acknowledgments

This library was developed for multi-crop genomic prediction research. Special thanks to the contributions from the plant genomics and deep learning communities.

---

**Last Updated**: December 30, 2025
**Version**: 0.2.0
**Status**: Active Development

## Recent Updates

### v0.2.0 (December 2025)
- ✨ Added CNN+Transformer hybrid encoder
- ✨ Enhanced training pipeline with progress bars (tqdm)
- ✨ Improved output structure with timestamped directories
- ✨ Added hyperparameters and model architecture to results
- ✨ Better predict.py with automatic architecture reconstruction
- 🐛 Fixed configuration parameter handling
- 📝 Updated documentation and examples
