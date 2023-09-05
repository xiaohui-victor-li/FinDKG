
# FinDKG Project: Dynamic Knowledge Graph with Large Language Models for Global Finance

This repository houses the codebase and scripts required to replicate the KGTransformer model specifically engineered for Temporal Knowledge Graphs (TKGs). TKGs extend the utility of traditional static knowledge graphs by incorporating a time dimension, thereby capturing dynamic interconnections between entities over a period. This added time complexity calls for specialized modeling techniques, such as KGTransformer, for performing tasks like temporal link prediction and anomaly detection.

# A. FinDKG Dataset

This repository includes the Financial Dynamic Knowledge Graph (FinDKG) dataset, which is specifically tailored to financial domain TKGs. This dataset is uniquely designed to focus on temporal knowledge graphs within the financial sector. It encompasses various types of financial entities and their evolving relationships over time, making it an ideal testbed for temporal link prediction, anomaly detection, and other financial network analysis tasks.

`/FinDKG_dataset` directory contains the specific dataset files. For details, refer to the [README](https://github.com/xiaohui-victor-li/FinDKG/tree/main/FinDKG_dataset) in that directory.  

---

# B. KGTransformer: Temporal Knowledge Graph Learning
KGTransformer is a state-of-the-art Graph Neural Network model explicitly designed for handling TKGs. It incorporates a probabilistic framework and leverages specialized attention mechanisms to better capture temporal relationships. The implementation of the KGTransformer model is seamlessly integrated within the `DKG` library, also part of this repository.

## DKG Python Library

Our DKG (Dynamic Knowledge Graph) Python library is housed within the `/DKG` directory. This library is a comprehensive toolbox offering functionalities to facilitate dynamic knowledge graph data manipulation, model training, and performance evaluation.

**Prerequisites:**
- Python 3.8+
- PyTorch 1.8+
- DGL 0.8+ (Follow installation guide [here](https://www.dgl.ai/pages/start.html))
- CUDA 11.0+ (if using GPU)

DKG library is an extension built on top of the DGL (Deep Graph Library). DGL is an open-source library that offers a versatile platform for implementing graph neural networks (GNNs). With efficient and flexible APIs, DGL has become an industry standard for both academic research and industrial applications in graph learning. Our DKG library leverages DGL's robust architecture to extend its capabilities specifically for dynamic knowledge graphs, featuring KGTransformer.

To get started with DKG, make sure you have the latest version of DKG 0.0.8+:

```python
import DKG
print(DKG.__version__)
```

For faster training, we recommend train the KGTransformer model on a GPU.

## Running the KGTransformer Model

Execute the following command to train and evaluate the default KGTransformer model on the bundled FinDKG dataset:

```bash
python3 train_DKG_run.py
```

You can tweak the model's behavior by modifying the following parameters within the `train_DKG_run.py` script:

```python
graph_mode = "FinDKG"         # Dataset choice: "FinDKG", "ICEWS18", "ICEWS14", "ICEWS_500", "GDELT", "WIKI", "YAGO"
model_ver = "KGTransformer"  # Model version: "GraphTransformer"
model_type ='KGT+RNN'        # Model type: 'KGT+RNN' for GraphTransformer | 'RGCN+RNN' for GraphRNN
epoch_times = 150            # Number of training epochs
random_seed = 1             # Seed for reproducibility
data_root_path = './data'    # Output data path
flag_train = True            # To train the model
flag_eval = True             # To evaluate the model
```

# Reference

For model details, please see the FinDKG website: https://xiaohui-victor-li.github.io/FinDKG/ and the original paper (under review). 


