# MISA+MMLatch: Modality-Invariant and -Specific Representations with Feedback for Multimodal Sentiment Analysis

This repository extends the original [MISA](https://arxiv.org/pdf/2005.03545.pdf) model by integrating the **MMLatch** feedback mechanism, enabling improved cross-modal alignment and information flow for multimodal sentiment analysis.

---

## Overview

- **MISA**: Learns both modality-invariant (shared) and modality-specific (private) representations for multimodal sentiment analysis.
- **MMLatch**: Introduces a feedback block that dynamically filters each modality's sequence using information from the other modalities, enhancing alignment and robustness.

---

## Key Features

- **Feedback Integration**: MMLatch feedback block is applied to the sequence representations before fusion, improving cross-modal interactions.
- **Flexible Modalities**: Supports text (BERT or embeddings), visual, and acoustic features.
- **Domain Adversarial Training**: Optionally uses CMD or adversarial loss for domain-invariant learning.
- **Notebook-Friendly**: Easily runnable in Jupyter/VSCode notebooks for experimentation and reproducibility.

---

## Setup

We recommend using the provided Conda environments for reproducibility.

```bash
# For GPU
conda env create -f env_gpu.yml
conda activate misa-code-py39-gpu

# For CPU
conda env create -f env_cpu.yml
conda activate misa-code-py39
```

---

## Data Preparation

1. **Download Datasets**  
   - Place the required datasets (e.g., MOSI, MOSEI, UR_FUNNY) in the `datasets` folder.
   - For BERT-based runs, ensure the correct tokenization and alignment as described in the code.

2. **Glove Embeddings**  
   - Download [GloVe 840B 300d](http://nlp.stanford.edu/data/glove.840B.300d.zip) and set the path in `config.py`.

3. **CMU-MultimodalSDK**  
   - Clone and install the [CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) in the project directory.

---

## Running the Model

### From Python Script

```bash
cd src
python train.py --data mosei --patience 4
```
- Replace `mosei` with `mosi` or `ur_funny` for other datasets.
- Adjust `--patience` and other hyperparameters as needed.

### From Notebook

You can run the model end-to-end in a Jupyter or VSCode notebook.  
See [`MISA_GPU.ipynb`](MISA_GPU.ipynb) for a step-by-step setup, including environment creation, data download, and training.

By changing the variable **branch** that appears twice you can run the different version of MISA+MMLatch.

Versions (and their respective branch):

- Feedback after Feature Extraction (best model) = master, diff_dim
- Feedback after private encoding phase = mpl_seq
- Feedback with static masks after Feat. Extr. = static_masks
- Feedback with static masks after private encoding = static_after_mlp
- MISA Reproduction = reprod_py39

**Notebook workflow:**
1. Clone the repository and set up the environment.
2. Download and prepare datasets.
3. Run the training cell to start model training and evaluation.
4. Use the provided cells to visualize results and analyze performance.

---

## Results

![image](https://github.com/user-attachments/assets/d7c61c73-48ce-427d-9896-1c1e5b7d15b9)

---
## Report

The report of research on MMLatch which is part of a team project with more info about the architecture, the result and other related research, you can find it here (https://github.com/alexisfilippakopoulos/mmlatch). My contribution is integrating MMlatch on MISA. 

## Citation

```
@article{hazarika2020misa,
  title={MISA: Modality-Invariant and-Specific Representations for Multimodal Sentiment Analysis},
  author={Hazarika, Devamanyu and Zimmermann, Roger and Poria, Soujanya},
  journal={arXiv preprint arXiv:2005.03545},
  year={2020}
}
```
@INPROCEEDINGS{9746418,
  author={Paraskevopoulos, Georgios and Georgiou, Efthymios and Potamianos, Alexandras},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Mmlatch: Bottom-Up Top-Down Fusion For Multimodal Sentiment Analysis}, 
  year={2022},
  volume={},
  number={},
  pages={4573-4577},
  doi={10.1109/ICASSP43922.2022.9746418}
}

---

## Notes & Tips

- **Hyperparameters**: All key hyperparameters (hidden size, patience, learning rate, etc.) can be set via command line or notebook arguments.
- **Reproducibility**: Random seeds are set for NumPy and PyTorch.
- **Checkpoints**: Models are saved in the `checkpoints` directory based on best validation performance.
- **Feedback Block**: The MMLatch feedback block is controlled via the `Feedback` class in `mmlatch.py` and is fully integrated into the MISA pipeline.

---

## Contact

For questions or contributions, please open an issue or contact the maintainers.
