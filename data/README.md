# Datasets

## Install
``` shell
pip install torch
pip install torch_geometric
```

## Usage
- Homogeneous graph datasets: `Cora`, `Citeseer`, `PubMed`
    ```py
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='data/Cora', name='Cora')  # up to your file root
    data = dataset[0]
    ```
- Heterogeneous graph datasets: `Texas`
    ```py
    from torch_geometric.datasets import WebKB

    dataset = WebKB(root='data/Texas', name='Texas')    # up to your file root
    data = dataset[0]
    ```

## Conference
- [Pytorch Geometric Datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)