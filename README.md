# bnn-speech-denoising

The repository is organized as follows:

**Main_BNN_Notebook.ipynb: The central Jupyter/Colab notebook containing the end-to-end pipeline (data exploration, training, evaluation, and visualizations).**


data_compressed.zip: A compressed archive of the dataset, which includes a subset of the TIMIT acoustic-phonetic corpus and wind noise audio files.


texts/: Documentation and references.

- **project_report_GAMLA.pdf: The final academic project report.**

- AI_usage_declaration.pdf: A declaration of AI tools used for coding assistance and text refinement.

- instructions.pdf: The original course project guidelines.

- papers/: A sub-directory containing the academic papers referenced and cited in the final report.


models/: Contains the saved .pth files of the trained models (DNN, BNN Gaussian, BNN Laplace, and the fine-tuned BNN check).


src/: Contains the modularized Python scripts imported by the main notebook:

- data_loader.py: Audio loading, mixing (SNR), and STFT feature extraction.
- models.py: PyTorch definitions for the DNN, Bayesian Linear Layers, and BNN architectures.
- train.py: Training loops, including ELBO optimization and KL divergence annealing.
- inference.py: Monte Carlo sampling for BNNs and audio reconstruction (ISTFT).
- metrics.py: Calculation of PESQ and STOI scores.
- utils.py: Visualization functions for learning curves, weight sparsity, and predictive variance.



How to Run
Environment: The code is optimized to run on Google Colab, but can be executed locally.

Dependencies: Ensure you have the required libraries installed (primarily torch, librosa, pesq, pystoi, matplotlib, numpy, and pandas).

Bash
pip install torch librosa pesq pystoi matplotlib numpy pandas
Data Extraction: You do not need to manually extract the data. Running the first few cells of Main_BNN_Notebook.ipynb will automatically detect data_compressed.zip and extract it into a local data/ directory.
