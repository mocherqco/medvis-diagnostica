# MedVis Diagnostica 🚑🧠

MedVis Diagnostica is an advanced medical image classification platform designed to support diagnostic decision-making in clinical environments. The system integrates state-of-the-art deep learning architectures with dynamic training pipelines and multi-modal learning.

## 🔍 Features
- Flexible configuration via YAML files
- Data augmentation with adjustable parameters
- Transfer learning with multiple architectures (ResNet, EfficientNet, MobileNetV2)
- Dynamic logging and result tracking
- Support for batch-wise data loading and preprocessing

## 🗂️ Project Structure

├── config/ # YAML configuration files
├── data/ # Images and labels
├── logs/ # Training logs
├── models/ # Model architectures and training pipeline
├── notebooks/ # Exploratory data analysis
├── utils/ # Data loading and logging utilities
├── requirements.txt # Python dependencies
└── LICENSE # License information


## 🚀 Getting Started

### Install dependencies
```bash
pip install -r requirements.txt

Run training pipeline
bash
Copy
Edit
python models/train_pipeline.py --config config/config.yaml

Explore data
Launch Jupyter Notebook and open notebooks/exploratory_analysis.ipyn
```
