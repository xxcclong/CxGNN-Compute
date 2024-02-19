import torch
import cxgnncomp
import cxgnndl
from tqdm import tqdm
import dgl
from .util import log
import time
from copy import deepcopy
from os import path
import cxgnncomp_backend
import torch.distributed as dist


class MultiModel():
    def __init__(self, model, num_device, dist):
        self.models = []
        self.hidden_channels = model.hidden_channels
        self.dist = dist
        for i in range(num_device):
            self.models.append(deepcopy(model).to(i))

    def __call__(self, batches):
        outs = []
        for it, model in enumerate(self.models):
            torch.cuda.set_device(it)
            outs.append(model(batches[it], skip_first=self.dist in ["tp", "opt"]))
        return outs

    def train(self):
        for item in self.models:
            item.train()
    
    def eval(self):
        for item in self.models:
            item.eval()

class MultiOptimier():
    def __init__(self, models, config):
        self.optimizers = []
        for model in models:
            self.optimizers.append(cxgnncomp.get_optimizer(config.train, model))

    def zero_grad(self):
        for item in self.optimizers:
            item.zero_grad()

    def step(self):
        for item in self.optimizers:
            item.step()

class MultiScheduler():
    def __init__(self, models, config):
        self.schedulers = []
        for model in models:
            self.schedulers.append(cxgnncomp.get_scheduler(config.train, model))

    def step(self):
        for item in self.schedulers:
            item.step()

class MultiLoss():
    def __init__(self, losses):
        self.losses = losses

    def backward(self):
        losses = []
        for it, loss in enumerate(losses):
            torch.cuda.set_device(it)
            loss.backward()

class MultiLossFn():
    def __init__(self, models, config):
        self.lossfn = cxgnncomp.get_loss_fn(config.train)

    def __call__(self, preds, ys):
        losses = []
        for i, item in enumerate(preds):
            losses.append(self.lossfn(item, ys[i]))
        return MultiLoss(losses)


class Trainer:

    def __init__(self, config):
        self.device = torch.device(config.dl.device)
        self.model = cxgnncomp.get_model(config)
        self.num_device = int(config.dl.num_device)
        if self.num_device <= 1: # single GPU
            self.num_device = 1
            self.model = self.model.to(self.device)
            self.optimizer = cxgnncomp.get_optimizer(config.train, self.model)
            self.scheduler = cxgnncomp.get_scheduler(config.train, self.optimizer)
            self.loss_fn = cxgnncomp.get_loss_fn(config.train)
        else: # Multi GPU
            self.dist = str(config.dl.dist).lower()
            self.model = MultiModel(self.model, self.num_device, self.dist)
            self.optimizer = MultiOptimier(self.model.models, config)
            self.scheduler = MultiScheduler(self.optimizer.optimizers, config)
            self.loss_fn = MultiLossFn(self.model.models, config)
            self.total_num_node = int(
                open(path.join(str(config["dl"]["dataset"]["path"]), "processed", "num_nodes.txt")).readline())
            self.in_channel = int(config.dl.dataset.feature_dim)
            self.local_feats = []
            self.local_starts = []
            self.local_ends = []
            print("using", self.dist)
            if self.dist == "ddp":
                for i in range(self.num_device):
                    self.local_feats.append(torch.randn(
                        [(self.total_num_node // self.num_device) + ((self.total_num_node % self.num_device) if i == self.num_device - 1 else 0), self.in_channel], dtype=torch.float32, device=i))
                    self.local_starts.append(self.total_num_node // self.num_device * i)
                    self.local_ends.append(self.total_num_node // self.num_device * (i + 1))
                self.local_ends[-1] = self.total_num_node
            elif self.dist in ["tp", "opt"]:
                self.weights = []
                for i in range(self.num_device):
                    self.local_feats.append(
                        torch.randn(
                            [self.total_num_node, self.in_channel // self.num_device], dtype=torch.float32, device=i)
                    )
                    self.local_starts.append(self.in_channel // self.num_device * i)
                    self.local_ends.append(self.in_channel // self.num_device * (i + 1))
                    if self.dist == "tp":
                        self.weights.append(torch.randn([self.in_channel // self.num_device, self.model.hidden_channels], dtype=torch.float32, device=i, requires_grad=True))
                    else:
                        self.weights.append(torch.randn([self.in_channel, self.model.hidden_channels], dtype=torch.float32, device=i, requires_grad=True))
                self.local_ends[-1] = self.in_channel
            else:
                assert False, f"dist {self.dist} not supported"


        self.evaluator = cxgnncomp.get_evaluator(config.train)
        self.loader = cxgnndl.get_loader(config.dl)
        self.type = config.train.type.lower()
        self.load_type = config.dl.type.lower()
        log.info(f"train type {self.type} load type {self.load_type}")
        self.config = config
        self.t_begin = 0
        self.t_end = 0
        self.status = "train"
        self.eval_begin = config.train.train.eval_begin
        self.skip = config.train.skip
        self.mode = "train"
        self.perform_test = True
        dataset_name = config.dl.dataset.name
        if "mag240" in dataset_name.lower():
            self.perform_test = False

    def to_dgl_block(self, batch):
        blocks = []
        num_layer = batch.num_node_in_layer.shape[0] - 1
        for i in range(len(batch.num_node_in_layer) - 1):
            num_src = batch.num_node_in_layer[num_layer - i]
            num_dst = batch.num_node_in_layer[num_layer - i - 1]
            ptr = batch.ptr[:num_dst + 1]
            idx = batch.idx[:batch.num_edge_in_layer[num_layer - i - 1]]
            blocks.append(
                dgl.create_block(('csc', (ptr, idx, torch.tensor([]))),
                                 int(num_src), int(num_dst)))
        return blocks

    def cxg_dgl_train_epoch(self):
        self.model.train()
        self.t_begin = time.time()
        for batch in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            self.optimizer.zero_grad()
            blocks = self.to_dgl_block(batch)
            batch_inputs = batch.x
            batch_labels = batch.y
            batch_pred = self.model([blocks, batch_inputs])
            loss = self.loss_fn(batch_pred, batch_labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        torch.cuda.synchronize()
        self.t_end = time.time()

    def cxg_dgl_eval_epoch(self, split="val"):
        self.model.eval()
        y_preds, y_trues = [], []
        losses = []
        num_seeds = 0
        if split == "val":
            loader = self.loader.val_loader
        else:
            loader = self.loader.test_loader
        self.t_begin = time.time()
        for batch in tqdm(
                loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            blocks = self.to_dgl_block(batch)
            if self.skip:
                continue
            batch_inputs = batch.x
            batch_labels = batch.y
            batch_pred = self.model([blocks, batch_inputs])
            loss = self.loss_fn(batch_pred, batch_labels)
            losses.append(loss.detach())
            y_preds.append(batch_pred.detach())
            y_trues.append(batch.y.detach())
            num_seeds += batch.y.shape[0]
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        self.metric = self.evaluator(y_preds, y_trues).item()
        self.loss = torch.sum(torch.stack(losses)) / num_seeds
        torch.cuda.synchronize()
        self.t_end = time.time()

    def cxg_train_epoch(self):
        self.model.train()
        self.t_begin = time.time()
        for batch in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            if self.skip:
                continue
            self.optimizer.zero_grad()
            out = self.model(batch)
            torch.cuda.synchronize()
            loss = self.loss_fn(out, batch.y)
            torch.cuda.synchronize()
            loss.backward()
            torch.cuda.synchronize()
            self.optimizer.step()
            self.scheduler.step()
        torch.cuda.synchronize()
        self.t_end = time.time()
    
    def generate_x(self, batches):
        for i, item in enumerate(batches):
            item.x = torch.randn(item.num_node_in_layer[-1], self.in_channel, device=i)
    
    def generate_x_ddp(self, batches):
        for i, batch in enumerate(batches):
            feats = []
            for j in range(self.num_device):
                # j -> i
                torch.cuda.set_device(j)
                sub_to_full = batch.sub_to_fulls[j]
                needed = sub_to_full[torch.logical_and(sub_to_full >= self.local_starts[j], sub_to_full < self.local_ends[j])]
                feats.append(self.local_feats[j][needed - self.local_starts[j]].to(i))
            torch.cuda.set_device(i)
            batch.x = torch.cat(feats, dim=0)

    def generate_x_tp(self, batches):
        for tar_it in range(self.num_device):
            arr_node_feat = []
            for dev_it in range(self.num_device):
                torch.cuda.set_device(dev_it)
                out = torch.index_select(
                    self.local_feats[dev_it],
                    dim=0,
                    index=batches[tar_it].sub_to_fulls[dev_it]
                )
                out = cxgnncomp_backend.sage_mean_forward(
                    out,
                    batches[tar_it].ptrs[dev_it],
                    batches[tar_it].idxs[dev_it],
                    batches[tar_it].num_node_in_layer[-2]
                )
                out = torch.mm(out, self.weights[dev_it])
                # arr_node_feat[tar_it][dev_it] = out
                arr_node_feat.append(out)
            batches[tar_it].x = arr_node_feat[tar_it]
            for dev_it in range(self.num_device):
                if dev_it == tar_it:
                    continue
                batches[tar_it].x += arr_node_feat[dev_it].to(tar_it)
    
    def generate_x_opt(self, batches):
        for tar_it in range(self.num_device):
            arr_node_feat = []
            for dev_it in range(self.num_device):
                torch.cuda.set_device(dev_it)
                out = torch.index_select(
                    self.local_feats[dev_it],
                    dim=0,
                    index=batches[tar_it].sub_to_fulls[dev_it]
                )
                out = cxgnncomp_backend.sage_mean_forward(
                    out,
                    batches[tar_it].ptrs[dev_it],
                    batches[tar_it].idxs[dev_it],
                    batches[tar_it].num_node_in_layer[-2]
                )
                # out = torch.mm(out, self.weights[dev_it])
                arr_node_feat.append(out)
            # batches[tar_it].x = arr_node_feat[tar_it]
            collect_feat = [] 
            for dev_it in range(self.num_device):
                collect_feat.append(arr_node_feat[dev_it].to(tar_it))
                # batches[tar_it].x += arr_node_feat[dev_it].to(tar_it)
            batches[tar_it].x = torch.cat(collect_feat, dim=1)
            batches[tar_it].x = torch.mm(batches[tar_it].x, self.weights[tar_it])

    
    def prepare_batch(self, batches):
        for i, batch in enumerate(batches):
            batch.ptr = batch.ptrs[i]
            batch.idx = batch.idxs[i]
            batch.y = batch.ys[i]



    def cxg_train_epoch_multigpu(self):
        self.model.train()
        self.t_begin = time.time()
        batches = []
        for batch in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            if self.skip:
                assert False
            batches.append(batch)
            if len(batches) != self.num_device:
                continue
            for i in range(self.num_device):
                torch.cuda.synchronize(i)
            # begin execution
            self.optimizer.zero_grad()
            self.prepare_batch(batches)
            if self.dist == "ddp":
                self.generate_x_ddp(batches)
            elif self.dist == "tp":
                self.generate_x_tp(batches)
            elif self.dist == "opt":
                self.generate_x_opt(batches)
            else:
                print("generating random x")
                self.generate_x(batches)
            out = self.model(batches)
            # torch.cuda.synchronize()
            loss = self.loss_fn(out, [b.y for b in batches])
            # torch.cuda.synchronize()
            loss.backward()
            # torch.cuda.synchronize()
            self.optimizer.step()
            self.scheduler.step()
            batches = []
        batches = []
        torch.cuda.synchronize()
        self.t_end = time.time()

    def pyg_train_epoch(self):
        self.model.train()
        self.t_begin = time.time()
        for batch in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            batch = batch.to(self.device)
            if self.skip:
                continue
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.loss_fn(out[:batch.y.shape[0]], batch.y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        torch.cuda.synchronize()
        self.t_end = time.time()

    def cxg_eval_epoch(self, split="val"):
        self.model.eval()
        y_preds, y_trues = [], []
        losses = []
        num_seeds = 0
        if split == "val":
            loader = self.loader.val_loader
        else:
            loader = self.loader.test_loader
        self.t_begin = time.time()
        for batch in tqdm(
                loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            out = self.model(batch)
            loss = self.loss_fn(out, batch.y)
            losses.append(loss.detach())
            y_preds.append(out.detach())
            y_trues.append(batch.y.detach())
            num_seeds += batch.y.shape[0]
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        self.metric = self.evaluator(y_preds, y_trues).item()
        self.loss = torch.sum(torch.stack(losses)) / num_seeds
        torch.cuda.synchronize()
        self.t_end = time.time()

    def load_subtensor(self, nfeat, labels, seeds, input_nodes, device):
        input_nodes = input_nodes.to(device)
        batch_inputs = nfeat[input_nodes]
        batch_labels = labels[seeds].to(device).long()
        return batch_inputs, batch_labels

    def dgl_train_epoch(self):
        self.model.train()
        self.t_begin = time.time()
        with self.loader.train_loader.enable_cpu_affinity():
            for (input_nodes, seeds, blocks) in tqdm(
                    self.loader.train_loader,
                    bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"
            ):
                if self.skip:
                    continue
                self.optimizer.zero_grad()
                blocks = [block.int().to(self.device) for block in blocks]
                batch_inputs, batch_labels = self.load_subtensor(
                    self.loader.feat,
                    self.loader.graph.ndata['labels'],
                    seeds,
                    input_nodes,
                    device=self.device)
                batch_pred = self.model([blocks, batch_inputs])
                loss = self.loss_fn(batch_pred, batch_labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
        torch.cuda.synchronize()
        self.t_end = time.time()

    def dgl_eval_epoch(self, split="val"):
        self.model.eval()
        y_preds, y_trues = [], []
        losses = []
        num_seeds = 0
        if split == "val":
            loader = self.loader.val_loader
        else:
            loader = self.loader.test_loader
        self.t_begin = time.time()
        for (input_nodes, seeds, blocks) in tqdm(
                loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            blocks = [block.int().to(self.device) for block in blocks]
            batch_inputs, batch_labels = self.load_subtensor(
                self.loader.feat,
                self.loader.graph.ndata['labels'],
                seeds,
                input_nodes,
                device=self.device)
            out = self.model([blocks, batch_inputs])
            loss = self.loss_fn(out, batch_labels)
            losses.append(loss.detach())
            y_preds.append(out.detach())
            y_trues.append(batch_labels.detach())
            num_seeds += batch_labels.shape[0]
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        self.metric = self.evaluator(y_preds, y_trues).item()
        self.loss = torch.sum(torch.stack(losses)) / num_seeds
        torch.cuda.synchronize()
        self.t_end = time.time()
        log.info("split {} metric {}".format(split, self.metric))

    def train(self):
        for epoch in range(self.config.train.train.num_epochs):
            self.mode = "train"
            if self.type == "dgl" and "dgl" in self.load_type:
                self.dgl_train_epoch()
            elif self.type == "dgl" and self.load_type == "cxg":
                self.cxg_dgl_train_epoch()
            elif self.type == "cxg":
                if self.num_device > 1:
                    self.cxg_train_epoch_multigpu()
                else:
                    self.cxg_train_epoch()
            elif self.type == "pyg" and self.load_type == "pyg":
                self.pyg_train_epoch()
            log.info(
                f"{self.mode}-epoch {epoch} {self.mode}-time {self.t_end - self.t_begin}"
            )
            if epoch >= self.eval_begin and not self.skip:
                self.mode = "val"
                if self.type == "dgl" and "dgl" in self.load_type:
                    self.dgl_eval_epoch("val")
                elif self.type == "dgl" and self.load_type == "cxg":
                    self.cxg_dgl_eval_epoch("val")
                elif self.type == "cxg":
                    self.cxg_eval_epoch("val")
                log.info(
                    f"{self.mode}-epoch {epoch} {self.mode}-time {self.t_end - self.t_begin} {self.mode}-loss {self.loss} {self.mode}-metric {self.metric}"
                )
                if self.perform_test:
                    self.mode = "test"
                    if self.type == "dgl" and "dgl" in self.load_type:
                        self.dgl_eval_epoch("test")
                    elif self.type == "dgl" and self.load_type == "cxg":
                        self.cxg_dgl_eval_epoch("test")
                    elif self.type == "cxg":
                        self.cxg_eval_epoch("test")
                    log.info(
                        f"{self.mode}-epoch {epoch} {self.mode}-time {self.t_end - self.t_begin} {self.mode}-loss {self.loss} {self.mode}-metric {self.metric}"
                    )
