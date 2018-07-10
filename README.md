# graph2vec

<p align="justify">
graph2vec is an embedding algorithm which learns representations for a set of graphs using an implicit factorization machine. The procedure places graphs in an abstract feature space where graphs with similar structural properties (Weisfehler-Lehman features) are clustered together. Graph2vec has a linear runtime complexity in the number of graphs in the dataset which makes it extremely scalable. This specific implementation supports multi-core data processing in the feature extraction and factorization phases. (So far this is the only implementation which support multi-core processing in every phase).
</p>

This repository provides an implementation for graph2vec as it is described in:
> graph2vec: Learning distributed representations of graphs.
> Narayanan, Annamalai and Chandramohan, Mahinthan and Venkatesan, Rajasekar and Chen, Lihui and Liu, Yang
> MLG 2017, 13th International Workshop on Mining and Learning with Graphs (MLGWorkshop 2017).

### Requirements

The codebase is implemented in Python 2.7. Package versions used for development are just below.
```
networkx          1.11
tqdm              4.19.5
numpy             1.13.3
pandas            0.20.3
jsonschema        2.6.0
joblib            0.11
gensim            3.1.0.
numpy             1.14.3
logging           0.4.9.6  
```

### Datasets

The code takes an input folder with json files. Every file is a graph and files have a numeric index as a name. The json files have two keys. The first key called "edges" corresponds to the edge list of the graph. The second key "features" corresponds to the node features. If the second key is not present the WL machine defaults to use the node degree as a feature.  A sample graph dataset from NCI1 is included in the `dataset/` directory.

### Options

Learning of the embedding is handled by the `src/graph2vec.py` script which provides the following command line arguments.

#### Input and output options

```
  --input-path STR                Input folder.                                     Default is `dataset`/.
  --output-path STR               Embeddings path.                                  Default is `features/nci1.csv`.
```
#### Model options
```
  --dimensions INT                Number of dimensions.                             Default is 128.
  --clusters INT                  Number of clusters.                               Default is 20.
  --lambd FLOAT                   KKT penalty.			                    Default is 0.2.
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
