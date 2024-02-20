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

Install requirements. Make sure [CxGNN-DL](https://github.com/xxcclong/CxGNN-DL) and the [modified triton](https://github.com/xxcclong/triton) are cloned and put aside with CxGNN-Compute.

After it, the directory tree should be like

```
.
|-- CxGNN-Compute
|-- CxGNN-DL
`-- triton
```

```bash
cd CxGNN-Compute
bash install.sh # this will install the prerequisites (e.g., CxGNN-DL) and CxGNN-Compute
```

### Data preparation

All datasets are from [OGB](https://ogb.stanford.edu/). We have pre-processed them for faster read. You can get access to them:

```bash
cd CxGNN-Compute
bash download.sh
```

After it, the directory tree should be like

```
.
|-- CxGNN-Compute
|-- CxGNN-DL
|-- triton
`-- data
```

## Reproduce

Scripts and READMEs for experiments are put in `test/ae/`

## TroubleShooting

If you meet any problem, please contact us through email (hkz20@mails.tsinghua.edu.cn) or HotCRP.

* Q: The program blocks when running overall test. A: Check the [overall test readme](test/ae/E1_overall/README.md) to fix the performance bug in PyG.
* Q: There are CUDA OOM errors in overall test. A: Some baseline test will suffer from OOM, their number will not be displayed in the result file.
* Q: I can only run `arxiv` in the accuracy test. A: The node feature data of the other two datasets are too large and not uploaded to the cloud drive. If you are interested in them, please contact me for the full datasets.