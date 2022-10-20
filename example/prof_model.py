from omegaconf import DictConfig
import torch
import cxgnncomp
import hydra
from omegaconf import DictConfig, OmegaConf


class Batch():

    def __init__(self, ptr, idx, x, num_node_in_layer, label):
        self.ptr = ptr
        self.idx = idx
        self.x = x
        self.num_node_in_layer = num_node_in_layer
        self.y = label


def prepare_data(dev):
    torch.manual_seed(0)
    dataset_name = "paper100m"
    file_dir = "/home/huangkz/repos/new-diskgnn/DiskGNN/graph_loader/{}_batch.pt".format(
        dataset_name)
    # file_dir = "/home/huangkz/repos/CxGNN-DL/dump.pt"
    batch = torch.load(file_dir)
    feat_len = 128
    device = torch.device(dev)
    x = torch.randn([batch["num_node_in_layer"][-1], feat_len],
                    dtype=torch.float32,
                    device=device)
    ptr = batch["ptr"].to(device)
    idx = batch["idx"].to(device)
    label = torch.randint(0,
                          171, (batch["num_node_in_layer"][0], ),
                          dtype=torch.long,
                          device=device)
    return Batch(ptr, idx, x, batch["num_node_in_layer"], label)


class Runner:

    def __init__(self, config):
        self.device = torch.device(config.dl.device)
        self.model = cxgnncomp.get_model(config)
        self.model = self.model.to(self.device)
        self.optimizer = cxgnncomp.get_optimizer(config.train, self.model)
        self.scheduler = cxgnncomp.get_scheduler(config.train, self.optimizer)
        self.loss_fn = cxgnncomp.get_loss_fn(config.train)

    def forward(self, batch):
        out = self.model(batch)
        loss = self.loss_fn(out, batch.y)
        return loss

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()


@hydra.main(version_base=None,
            config_path="../../CxGNN-DL/configs",
            config_name="config")
def main(config: DictConfig):
    batch = prepare_data(config.dl.device)
    runner = Runner(config)
    import time
    t_fwd = 0
    t_bwd = 0
    batch.x.requires_grad = True
    for i in range(100):
        t0 = time.time()
        runner.model.train()
        loss = runner.forward(batch)
        torch.cuda.synchronize()
        t1 = time.time()
        runner.backward(loss)
        torch.cuda.synchronize()
        t2 = time.time()
        t_fwd += t1 - t0
        t_bwd += t2 - t1
    print("fwd: {:.3f}ms, bwd: {:.3f}ms".format(t_fwd * 1000, t_bwd * 1000))


if __name__ == "__main__":
    main()
