import cxgnncomp
import cxgnndl
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None,
            config_path="../../CxGNN-DL/configs",
            config_name="config")
def main(config: DictConfig):
    cxgnncomp.set_timers()
    s = OmegaConf.to_yaml(config)
    log.info(s)
    new_file_name = "new_config.yaml"
    s_dl = OmegaConf.to_yaml(config.dl)
    with open(new_file_name, 'w') as f:
        s = s_dl.replace("-", "  -")  # fix cpp yaml interprete
        f.write(s)
    trainer = cxgnncomp.Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
