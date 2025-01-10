# Conditional Generative Adversarial Network (CGAN)

This repository implements a **Conditional Generative Adversarial Network (CGAN)** to generate images conditioned on specific labels. The project is built using **PyTorch** and is designed to demonstrate the core concepts of GANs with conditional input.

---

## 🌟 Key Features

- **Conditional GAN Architecture**:
  - Combines a generator and a discriminator.
  - Conditions both generator and discriminator on class labels.

- **Custom Dataset Integration**:
  - Easily integrate your dataset by modifying the data loading pipeline.
  - Conditional labels are passed during training.

- **Evaluation and Visualization**:
  - Generate and save images conditioned on specific labels.
  - Loss metrics for generator and discriminator are tracked during training.

---

## 📂 Project Structure

- `cgan.py` - Contains the CGAN model definitions (Generator and Discriminator).
- `train.py` - Training pipeline for the CGAN.
- `utils.py` - Utility functions for saving images and visualizations.
- `dataset.py` - Code for loading and processing datasets.
- `README.md` - This file.

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+**
2. **PyTorch 1.12+**
3. **Additional Libraries**:
   - torchvision
   - matplotlib

Install the required libraries using:
```bash
pip install torch torchvision matplotlib

Run Training
python train.py


Evaluate the Model
python generate.py


📊 Loss Functions
adversarial_loss = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))



Contributions towards it are welcome feel free to connect!









