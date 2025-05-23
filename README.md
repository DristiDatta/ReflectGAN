# ReflectGAN: Vegetation-to-Bare Soil Reflectance Translation using GANs

ReflectGAN is a deep learning framework designed to reconstruct bare soil reflectance from vegetation-contaminated satellite observations using a paired Generative Adversarial Network (GAN). This model supports improved Soil Organic Carbon (SOC) estimation by minimizing spectral distortions caused by vegetation cover.

## 🔍 Overview

Traditional SOC estimation models often struggle in vegetated areas due to mixed reflectance signals. ReflectGAN addresses this by learning a direct spectral transformation from vegetated soil reflectance to corresponding bare soil signatures using paired training data. The architecture consists of:

- A **Generator** with residual learning blocks to recover bare soil spectra.
- A **Discriminator** to enforce realistic spectral distributions via adversarial training.
- Support for performance tracking with custom metrics (R², RMSE, RPD).

## 🧠 Model Architecture

- `generator.py`: Defines the Generator network using residual blocks.
- `discriminator.py`: Implements the Discriminator that distinguishes between real and generated reflectance.
- `main.py`: Trains the model, performs spectral translation, and evaluates results.
- `utils.py`: Contains helper functions for metric calculations and result formatting.

## 📂 Data

The model expects a CSV file in `data/paired_dataset_veg2bare.csv` containing:
- **Input features**: Vegetated reflectance (Landsat 8 Bands B1–B7)
- **Target features**: Corresponding bare soil reflectance (B1–B7)
- **Optional**: SOC values and vegetation indices (e.g., NDVI) for extended experiments

Example CSV format:
