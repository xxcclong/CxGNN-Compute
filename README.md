# CxGNN-Compute

## Setup

If you use our cluster, just activate the prepared environment.

```bash
source /data/eurosysae/.venv/cxgnn/bin/activate
```

Else, you need to setup the environment and prepare the data:

### Environment

Activate virtual environment:
```bash
mkdir ~/.venv
python3 -m venv ~/.venv/cxgnn
source ~/.venv/cxgnn/bin/activate
```

Install requirements. Make sure [CxGNN-DL](https://github.com/xxcclong/CxGNN-DL) is cloned and put aside with CxGNN-Compute.

```bash
cd CxGNN-Compute
bash install.sh # this will install the prerequisites (e.g., CxGNN-DL) and CxGNN-Compute
```

### Data preparation

All datasets are from [OGB](https://ogb.stanford.edu/). We have pre-processed them for faster read. You can get access to them:

```bash
bash download.sh
```

After it, the directory tree should be like

```
.
|-- CxGNN-Compute
|-- CxGNN-DL
`-- data
```

## Reproduce

Scripts and READMEs for experiments are put in `test/ae/`

## TroubleShooting

If you meet any problem, please contact us through email (hkz20@mails.tsinghua.edu.cn) or HotCRP.