
# ReflectGAN: Modeling Vegetation Effects for Soil Carbon Estimation from Satellite Imagery

ReflectGAN is a paired Generative Adversarial Network (GAN) designed to reconstruct bare soil reflectance from vegetation-contaminated satellite observations. The goal is to enhance soil organic carbon (SOC) estimation in vegetated areas by transforming input reflectance into vegetation-corrected spectra that better represent bare soil conditions.

This repository supports the study titled:

**"ReflectGAN: Modeling Vegetation Effects for Soil Carbon Estimation from Satellite Imagery"**  
*(Submitted to IEEE Transactions on Geoscience and Remote Sensing - TGRS)*

---

## 🌱 Key Features

- Paired GAN architecture tailored for spectral correction.
- Residual learning blocks to capture complex spectral transformations.
- Trained on paired vegetated-bare reflectance samples.
- Outputs vegetation-free reflectance to support downstream SOC estimation.
- Includes real-world Landsat 8-based paired dataset.

---

## 📁 Repository Structure

```
ReflectGAN/
│
├── main.py               # Training pipeline and evaluation
├── generator.py          # Generator network with residual blocks
├── discriminator.py      # Discriminator network for binary classification
├── utils.py              # Utility functions for printing and evaluation
│
├── data/
│   └── paired_dataset_veg2bare.csv  # Input paired reflectance dataset (vegetated and bare)
│
├── results/
│   ├── train.csv         # Training split
│   ├── test.csv          # Testing split
│   └── generated_data.csv # Output from generator after training
│
└── README.md             # Project overview
```

---

## 📊 Dataset

The dataset contains 354 paired samples consisting of:

- Vegetated reflectance (Bands B1 to B7)
- Corresponding bare soil reflectance (Bands B1_bare to B7_bare)
- NDVI and ground-truth SOC values

> 📍 Dataset location: `data/paired_dataset_veg2bare.csv`

---

## 🚀 How to Run

1. **Install Dependencies**

Ensure Python 3.8+ is installed. Then install required packages:

```bash
pip install torch scikit-learn pandas numpy
```

2. **Train ReflectGAN**

Run the training script:

```bash
python main.py
```

3. **Output**

- Generated bare soil reflectance will be saved to `results/generated_data.csv`.
- Console output displays side-by-side reflectance comparisons.

---

## 🧠 Model Architecture

- **Generator**  
  - Input: Vegetated reflectance (7 bands)  
  - Layers: Dense layer → 4 Residual Blocks (expand-compress pattern) → Dense layer  
  - Output: Reconstructed bare soil reflectance (7 bands)

- **Discriminator**  
  - Input: Concatenated input-target reflectance (14 features)  
  - Layers: 3-layer MLP with Dropout and BatchNorm  
  - Output: Real/Fake classification (sigmoid)

---


> These are printed using `utils.print_metrics()`.


---

## 📬 Citation

If you find this work helpful, please cite:

> D. Datta, M. Paul, M. Murshed, S. W. Teng, and L. Schmidtke,  
> "**ReflectGAN: Modeling Vegetation Effects for Soil Carbon Estimation from Satellite Imagery**,"  
> *IEEE Transactions on Geoscience and Remote Sensing (Under Review),* 2024.  
> [GitHub Repo](https://github.com/DristiDatta/ReflectGAN)

---

## 🤝 Acknowledgment

This work was supported by the Cooperative Research Centre for High Performance Soils (Soil CRC), funded by the Australian Government’s Cooperative Research Centres Program.
