## Install

```
# install cxgnndl before it
source /opt/spack/share/spack/setup-env.sh
spack load /5juudln # cuda11.3
spack load /7zlelqx # cudnn8.2.4
python setup.py build -j16 develop --user
```

## Run

```
python example/train.py dl/dataset=arxiv dl.type=cxg train.type=dgl # using cxg as the loader, using dgl to define the model
python example/train.py dl/dataset=arxiv dl.type=cxg train.type=cxg train.model=gcn # using cxg as the loader, using cxg to define the model
```
