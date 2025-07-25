
# Real-Fake Image Classification: Demographic Bias Analysis

This repository contains the implementation of a comprehensive study examining demographic biases in real versus synthetic face image classification tasks, focusing on images generated by Stable Diffusion and Generative Adversarial Networks (GANs).

## Overview

This project investigates the presence of demographic biases in CNN-based classification systems that distinguish between real and AI-generated face images. The study evaluates two primary classification tasks:
- **Real vs. Stable Diffusion**: Distinguishing real face images from those generated by Stable Diffusion models
- **Real vs. GAN**: Differentiating between real face images and GAN-generated faces

## Key Features

- **Bias Analysis**: Statistical evaluation of gender-based performance disparities using two-proportion Z-tests
- **Multiple Architectures**: Implementation of both custom CNN and ResNet-18 transfer learning models
- **Interpretability**: Gradient-weighted Class Activation Mapping (Grad-CAM) for model decision visualization
- **Comprehensive Datasets**: Integration of multiple datasets including FairFace for bias measurement

## Model Architectures

### Custom CNN
- Convolutional layers with 32 and 64 filters (3×3 kernels)
- ReLU activation functions
- Max-pooling layers (2×2)
- Fully connected layers (128 units → 2 output classes)
- Early stopping with patience=3

### Transfer Learning (ResNet-18)
- Pre-trained ResNet-18 with frozen parameters
- Custom classification head for binary classification
- Fine-tuned for real/fake image distinction

## Datasets

### Training Data
- **Real Images**: Flickr dataset (77,856 images)
- **GAN Images**: Kaggle dataset (77,856 images, balanced sampling)
- **Stable Diffusion Images**: Custom generated dataset (9,319 training images)
  - Generated using Realistic Vision v6.0B1 and Realistic Stock Photo v2.0
  - Prompts: "Close up face" with controlled "with glasses" variations

### Bias Evaluation
- **Supplementary Set**: 2,000 manually labeled images for bias testing
- **FairFace Dataset**: 86,745 images with demographic labels for external validation

## Key Findings

The study reveals significant demographic biases across different model-dataset combinations:

- **Stable Diffusion Models**: Generally show higher accuracy for female images
- **GAN Models**: Exhibit variable bias patterns depending on architecture
- **Statistical Significance**: Most models show p-values < 0.05 in two-proportion Z-tests

## Installation

```bash
# Clone the repository
git clone https://github.com/verynewusername/RealFakeClassification.git
cd RealFakeClassification

# Install dependencies
pip install torch torchvision
pip install opencv-python
pip install matplotlib
pip install numpy
pip install scikit-learn
```
## Usage

This project provides several scripts and notebooks for different classification tasks.

### Prerequisites

Before you begin, ensure you have Python installed. You will also need to install the necessary libraries. You can do this by running:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

### Classification

You can use the provided scripts to classify images as real or fake.

**Simple Classification:**

To use a simple classifier, run the `Simple-Classify.py` script:

```bash
python Simple-Classify.py
```

**Transfer Learning Classification:**

For a more advanced classifier that uses transfer learning, run the `TransferLearning-Classify.py` script:

```bash
python TransferLearning-Classify.py
```

**Jupyter Notebooks:**

Alternatively, you can use the `SimpleClassify.ipynb` Jupyter Notebook for an interactive classification experience.

### Bias and Fairness Analysis

This repository includes tools to analyze bias in the models.

**Bias Analysis Scripts:**

You can run the `bias.py` and `biasStat.py` scripts to perform bias analysis:

```bash
python bias.py
python biasStat.py
```

The results of the bias analysis, including charts, can be found in the `BiasCharts` directory.

### Visualization

To visualize the model's predictions and understand how it makes decisions, you can use the `gradcam.py` and `visualizer.py` scripts.

```bash
python gradcam.py
python visualizer.py
```

These scripts will generate visualizations that highlight the areas of an image the model focuses on when making a prediction.

## Statistical Analysis

The project implements rigorous statistical testing:

- **Two-Proportion Z-Test**: Evaluates significance of accuracy differences between demographic groups
- **Null Hypothesis**: No difference in classification accuracy between male and female images
- **Alternative Hypothesis**: Significant difference exists between demographic groups

## Image Processing

All images are standardized to:
- **Resolution**: 256×256 pixels
- **Downscaling Method**: LANCZOS interpolation for quality preservation
- **Format**: RGB channels
- **Normalization**: Standard ImageNet preprocessing

## Results Summary

| Model | Dataset | Male Accuracy | Female Accuracy | P-value |
|-------|---------|---------------|-----------------|---------|
| CNN | GAN-Real | 96.84% | 98.30% | 0.032 |
| CNN | SD-Real | 98.94% | 99.69% | 0.001 |
| ResNet-18 | GAN-Real | 91.53% | 84.48% | 0.000002 |
| ResNet-18 | SD-Real | 93.79% | 99.76% | 0.000000 |

## Contributing

This research project welcomes contributions in the following areas:
- Additional generative model evaluation (e.g., DALL-E, Midjourney)
- Extended demographic analysis (age, ethnicity)
- Improved bias mitigation techniques
- Enhanced interpretability methods

## Citation

If you use this code in your research, please cite:

```bibtex
@thesis{sirin2024demographic,
  title={Demographic Biases in Real-Fake Image Classification Tasks},
  author={Şirin, Efe Görkem},
  year={2024},
  school={University of Groningen},
  type={Bachelor's Thesis}
}
```

## License

This project is available for academic and research purposes. Please refer to the license file for detailed terms.


