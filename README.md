# graph2vec

<p align="justify">
graph2vec is an embedding algorithm which learns representations for a set of graphs using an implicit factorization machine. The procedure places graphs in an abstract feature space where graphs with similar structural properties (Weisfehler-Lehman features) are clustered together. Graph2vec has a linear runtime complexity in the number of graphs in the dataset which makes it extremely scalable. This specific implementation supports multi-core data processing in the feature extraction and factorization phases. (So far this is the only implementation which support multi-core processing in every phase).
</p>

This repository provides an implementation for graph2vec as it is described in:
> Community Preserving Network Embedding.
> Xiao Wang, Peng Cui, Jing Wang, Jain Pei, WenWu Zhu, Shiqiang Yang.
> Proceedings of the Thirsty-First AAAI conference on Artificial Intelligence (AAAI-17).

### Requirements

The codebase is implemented in Python 2.7. Package versions used for development are just below.
```
networkx          1.11
tqdm              4.19.5
numpy             1.13.3
pandas            0.20.3
tensorflow-gpu    1.3.0
jsonschema        2.6.0
texttable         1.2.1
python-louvain    0.11
```

### Datasets

The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. A sample graph for the `Facebook Politicians` dataset is included in the  `data/` directory.

### Logging

The models are defined in a way that parameter settings and cluster quality is logged in every single epoch. Specifically we log the followings:

```
1. Hyperparameter settings.     We save each hyperparameter used in the experiment.
2. Cluster quality.             Measured by modularity. We calculate it in every epoch.
3. Runtime.                     We measure the time needed for optimization -- measured by seconds.
```

### Options

Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --input STR                     Input graph path.                                 Default is `data/food_edges.csv`.
  --embedding-output STR          Embeddings path.                                  Default is `output/embeddings/food_embedding.csv`.
  --cluster-mean-output STR       Cluster centers path.                             Default is `output/cluster_means/food_means.csv'`.
  --log-output STR                Log path.                                         Default is `output/logs/food.log`.
  --assignment-output STR         Node-cluster assignment dictionary path.          Default is `output/assignments/food.json`.
  --dump-matrices BOOL            Whether the trained model should be saved.        Default is `True`.
```
#### Model options
```
  --dimensions INT                Number of dimensions.                             Default is 16.
  --clusters INT                  Number of clusters.                               Default is 20.
  --lambd FLOAT                   KKT penalty.			                    Default is 0.2.
  --alpha FLOAT                   Clustering penalty.                               Default is 0.05.
  --beta FLOAT                    Modularity regularization penalty.                Default is 0.05.
  --eta FLOAT                     Similarity mixing parameter.                      Default is 5.0.
  --lower-control FLOAT           Floating point overflow control.                  Default is 10**-15.
  --iteration-number INT          Number of power iterations.                       Default is 200.
  --early-stopping INT            Early stopping round number based on modularity.  Default is 3.
```

### Examples

The following commands learn a graph embedding, cluster centers and writes them to disk. The node representations are ordered by the ID.

Creating an MNMF embedding of the default dataset with the default hyperparameter settings. Saving the embedding, cluster centres and the log file at the default path.

```
python src/main.py
```

Turning off the model saving.

```
python src/main.py --dump-matrices False
```

Creating an embedding of an other dataset the `Facebook Companies`. Saving the output and the log in a custom place.

```
python src/main.py --input data/company_edges.csv  --embedding-output output/embeddings/company_embedding.csv --cluster-mean-output output/cluster_means/company_means.csv
```

Creating a clustered embedding of the default dataset in 128 dimensions and 10 cluster centers.

```
python src/main.py --dimensions 128 --clusters 10
```
