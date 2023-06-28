import cxgnncomp as cxgc
import cxgnncomp_backend
import time
import torch
import cxgnndl
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

import logging

log = logging.getLogger(__name__)

base_dir = "/home/huangkz/data/dataset_diskgnn/{}/processed/"


def degree_metric(config, loader):
    ptr = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) +
                    "csr_ptr_undirected.dat",
                    dtype=np.int64)).cuda()
    deg = ptr[1:] - ptr[:-1]  # in degree
    return deg


def profile_metric(config, loader):
    num_node = int(
        open(base_dir.format(config.dl.dataset.name) + "num_nodes.txt").read())
    metric = torch.zeros(num_node, dtype=torch.int64).cuda()
    num_epochs = 1
    for epoch in range(num_epochs):
        for cur_loader in [loader.train_loader]:
            for batch in cur_loader:
                nodes = batch.sub_to_full
                metric[nodes] += 1
        log.info(f"epoch {epoch} metric {torch.sum(metric > 0)}")
    return metric


def pagerank_metric2(config, loader):
    edge_index = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) + "edge_index.dat",
                    dtype=np.int64)).view(2, -1).cuda()
    num_node = int(
        open(base_dir.format(config.dl.dataset.name) + "num_nodes.txt").read())
    metric = torch.zeros(num_node, dtype=torch.int64, device=edge_index.device)
    training_nodes = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) + "split/time/" +
                    "train_idx.dat",
                    dtype=np.int64)).cuda()
    metric[training_nodes] = 1

    log.info("initial training nodes {}".format(torch.sum(metric > 0)))
    for l in range(3):
        metric[edge_index[1][metric[edge_index[0]] > 0]] += 1
        metric[edge_index[0][metric[edge_index[1]] > 0]] += 1
        log.info("layer {} training nodes {}".format(l, torch.sum(metric > 0)))


def pagerank_metric(config, loader):
    ptr = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) +
                    "csr_ptr_undirected.dat",
                    dtype=np.int64))
    idx = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) +
                    "csr_idx_undirected.dat",
                    dtype=np.int64))
    num_node = int(
        open(base_dir.format(config.dl.dataset.name) + "num_nodes.txt").read())
    metric = torch.zeros(num_node, dtype=torch.int64, device=ptr.device)
    deg = ptr[1:] - ptr[:-1]  # in degree
    training_nodes = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) + "split/time/" +
                    "train_idx.dat",
                    dtype=np.int64))
    metric[training_nodes] = 1
    for l in range(3):
        for i in range(training_nodes.shape[0]):
            metric[idx[ptr[training_nodes[i]]:ptr[training_nodes[i] + 1]]] += 1
        training_nodes = (metric > 0).nonzero().squeeze()
        log.info("layer {} training nodes {}".format(l, training_nodes.shape))


def get_cache_map(metric, cache_rate=0.1):
    sorted, indices = torch.sort(metric, descending=True)
    import matplotlib.pyplot as plt
    plt.plot(sorted.cpu().numpy())
    plt.savefig("sorted.png")
    exit()
    num_node = metric.shape[0]
    thres = metric[indices[int(num_node * cache_rate)]]
    print("thres", thres)
    cache_status = metric >= thres
    print("num cached", torch.sum(cache_status))
    log.info("rate touched {}".format(torch.sum(metric > 0) / metric.shape[0]))
    log.info("cache can hit {}".format(
        torch.sum(metric[cache_status]) / torch.sum(metric)))
    return cache_status


def test_cache(cache_status, loader):
    num_epochs = 3
    num_visit = 0
    num_hit = 0
    for epoch in range(num_epochs):
        for cur_loader in [
                # loader.train_loader, loader.val_loader, loader.test_loader
                loader.train_loader
        ]:
            for batch in cur_loader:
                nodes = batch.sub_to_full
                num_visit += nodes.shape[0]
                num_hit += int(torch.sum(cache_status[nodes]))
        log.info("hit rate {} {} {}".format(num_hit / num_visit, num_hit,
                                            num_visit))
        # print(nodes.shape)
    # print("hit rate", num_hit / num_visit, num_hit, num_visit)


@hydra.main(version_base=None,
            config_path="../../../CxGNN-DL/configs",
            config_name="config")
def main(config: DictConfig):
    s = OmegaConf.to_yaml(config)
    log.info(s)
    new_file_name = "new_config.yaml"
    s_dl = OmegaConf.to_yaml(config.dl)
    with open(new_file_name, 'w') as f:
        s = s_dl.replace("-", "  -")  # fix cpp yaml interprete
        f.write(s)

    loader = cxgnndl.get_loader(config.dl)

    # pagerank_metric2(config, loader)

    for func in [profile_metric, degree_metric]:
        print(str(func))
        metric = func(config, loader)
        cache_status = get_cache_map(metric, cache_rate=config.dl.cache_rate)
        test_cache(cache_status, loader)


if __name__ == '__main__':
    main()