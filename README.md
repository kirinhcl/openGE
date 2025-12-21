# OpenGE: Multi-Crop Trait Prediction with G×E Interactions

OpenGE is a deep learning library designed for **crop trait prediction** using **Genotype (G) × Environment (E) interaction modeling**. The library is built to be crop-agnostic and supports multiple crops with minimal configuration changes.

## Features

✨ **Core Capabilities**
- **G×E Interaction Modeling**: Explicitly captures how genetic and environmental factors interact to influence crop traits
- **Multi-Crop Support**: Easily adapt the library to different crops through configuration files
- **Modular Architecture**: Flexible combination of genetic/environment encoders and fusion strategies
- **Production-Ready**: Includes training, prediction, and evaluation pipelines

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

```bash
python train.py --config configs/maize_2024.yaml --data-dir ./Training_data
```

### 2. Prediction

```bash
python predict.py --model-path results/best_model.pt --config configs/maize_2024.yaml --data-dir ./Testing_data
```

### 3. Interpretability Analysis

```bash
python interpret.py --model-path results/best_model.pt --config configs/maize_2024.yaml --data-dir ./data
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

### Base Configuration (`configs/base.yaml`)
Defines default settings for model architecture, training hyperparameters, and data preprocessing.

### Crop-Specific Configurations
- `maize_2024.yaml`: Optimized settings for maize trait prediction
- `wheat_2025.yaml`: Optimized settings for wheat trait prediction

Example crop config:
```yaml
crop: "maize"
year: 2024

model:
  genetic_encoder:
    name: "transformer"
    input_dim: 40000  # Maize SNP count
    hidden_dim: 256
  env_encoder:
    name: "mlp"
    input_dim: 30

training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.001
```

## Model Architectures

### G×E Model
The core model combines:
- **Genetic Encoder**: Processes SNP/marker data (CNN, Transformer, or MLP)
- **Environment Encoder**: Processes weather, soil, EC data (typically MLP)
- **Fusion Layer**: Combines representations (Concatenation, Attention, or Gating)
- **Prediction Head**: Generates trait predictions (Regression or Classification)

### Attention Fusion
The default fusion strategy uses learned attention to weight the contribution of genetic vs environmental factors dynamically.

### Weight-Sparse Transformers
For interpretability, sparse transformers enforce sparsity patterns in:
- Attention heads (top-K attention)
- Feedforward connections (magnitude pruning)

## Training

```python
from openge.core import Config, Trainer
from openge.models import GxEModel

# Load configuration
config = Config("configs/maize_2024.yaml")

# Build model
model = build_model(config)

# Setup trainer
trainer = Trainer(
    model=model,
    criterion=nn.MSELoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    device="cuda"
)

# Train
history = trainer.fit(train_loader, val_loader, epochs=100)
```

## Making Predictions

```python
# Load trained model
model = torch.load("results/best_model.pt")
model.eval()

# Make predictions
predictions, targets = trainer.predict(test_loader)

# Evaluate
from openge.utils import calculate_rmse, calculate_r2
rmse = calculate_rmse(targets, predictions)
r2 = calculate_r2(targets, predictions)
```

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

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- PyYAML >= 5.4.0
- matplotlib >= 3.4.0

Optional:
- SHAP >= 0.40.0 (for SHAP explanations)
- Captum >= 0.4.0 (for advanced interpretability)

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

**Last Updated**: December 2024
**Version**: 0.1.0
**Status**: Active Development
