# Leveraging Graph Neural Networks and Mobility Data for COVID-19 Forecasting

This repository contains the code and supporting materials for the article  
**"Leveraging graph neural networks and mobility data for COVID-19 forecasting"**,  
published in *Applied Soft Computing* (ASOC_115242, accepted April 16, 2026).

The project investigates how **Graph Neural Networks (GNNs)** combined with **mobility data** can improve forecasting of COVID-19 cases. By modeling population movement patterns as graphs, we aim to enhance predictive accuracy compared to traditional epidemiological approaches.

---

## 📂 Repository Structure
├── analisys/                  # Analysis scripts
├── codes/                     # Core source code
├── raw_data/                  # Input datasets (pre-processed)
├── results/                   # Results using cumulative case data
├── results_daily/             # Results using daily case data
├── results_tune_1/            # Hyperparameter tuning experiments
├── environment_mac_m1.yml     # Conda environment for Mac M1
├── environment_windows_11.yml # Conda environment for Windows 11
├── requirements.txt           # Python dependencies
├── LICENSE                    # License (AGPL-3.0)
└── README.md                  # Project documentation

---

## 🚀 Features
- Implementation of **Graph Neural Network models** for epidemiological forecasting.
- Integration of **mobility datasets** to enrich predictions.
- Scripts for training, evaluation, and visualization.
- Ready-to-use environments for both **Windows 11** and **Mac M1** systems.

---

## 📦 Installation
Clone the repository and install dependencies:

```
git clone https://github.com/hodfernando/Leveraging-graph-neural-networks-and-mobility-data-for-COVID-19-forecasting.git
cd Leveraging-graph-neural-networks-and-mobility-data-for-COVID-19-forecasting

pip install -r requirements.txt
```

Or create a Conda environment:

```
conda env create -f environment_windows_11.yml
# or
conda env create -f environment_mac_m1.yml
```

🖥️ Usage
Run the training script with your dataset:

```
python codes/'train_model'.py
```

Additional analysis scripts are available in the analisys/ folder.

📊 Results
results/ → Forecasts based on cumulative COVID-19 case data.

results_daily/ → Forecasts based on daily COVID-19 case data.

results_tune_1/ → Hyperparameter tuning experiments.

Disclaimer:  
The complete set of generated results amounts to approximately 1.3 TB of data.
Due to the impractical size, these full outputs are not included in this repository.
Instead, we provide the codebase, analysis scripts, and figure-generation routines so that results can be reproduced or adapted as needed. Selected figures and examples are included for reference.

📜 License
This project is licensed under the AGPL-3.0. See the LICENSE file for details.

📖 Reference
If you use this repository, please cite the article:

Duarte, Fernando, et al. Leveraging graph neural networks and mobility data for COVID-19 forecasting.
Applied Soft Computing, 2026. Reference: ASOC_115242.

## Citation

If you find this work useful for your research, please consider citing our paper:

**Title:** Leveraging graph neural networks and mobility data for COVID-19 forecasting
**Journal:** *Applied Soft Computing*, 2026  
**DOI:** [10.1016/j.asoc.2026.115242](https://doi.org/10.1016/j.asoc.2026.115242)

You can access the full version here: [ScienceDirect - Full Article](https://www.sciencedirect.com/science/article/pii/S1568494626006903)

### BibTeX
```latex
@article{DUARTE2026ASOC,
  title = {Leveraging graph neural networks and mobility data for COVID-19 forecasting},
  journal = {Applied Soft Computing},
  year = {2026},
  issn = {1568-4946},
  doi = {10.1016/j.asoc.2026.113264},
  url = {[https://doi.org/10.1016/j.asoc.2026.115242](https://doi.org/10.1016/j.asoc.2026.115242)},
  author = {Fernando Duarte and [Other Authors]}
}
