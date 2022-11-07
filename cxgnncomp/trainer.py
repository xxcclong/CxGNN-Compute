import torch
import cxgnncomp
import cxgnndl
from tqdm import tqdm
import dgl
from .util import log
import time


class Trainer:

    def __init__(self, config):
        self.device = torch.device(config.dl.device)
        self.model = cxgnncomp.get_model(config)
        self.model = self.model.to(self.device)
        self.optimizer = cxgnncomp.get_optimizer(config.train, self.model)
        self.scheduler = cxgnncomp.get_scheduler(config.train, self.optimizer)
        self.loss_fn = cxgnncomp.get_loss_fn(config.train)
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
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.loss_fn(out, batch.y)
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
        for (input_nodes, seeds, blocks) in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
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

    def train(self):
        for epoch in range(self.config.train.train.num_epochs):
            if self.type == "dgl" and self.load_type == "dgl":
                self.dgl_train_epoch()
            elif self.type == "dgl" and self.load_type == "cxg":
                self.cxg_dgl_train_epoch()
            elif self.type == "cxg":
                self.cxg_train_epoch()
            log.info(f"train-epoch {epoch} time {self.t_end - self.t_begin}")
            if epoch >= self.eval_begin:
                if self.type == "dgl" and self.load_type == "dgl":
                    self.dgl_eval_epoch("val")
                    self.dgl_eval_epoch("test")
                elif self.type == "dgl" and self.load_type == "cxg":
                    self.cxg_dgl_eval_epoch("val")
                    self.cxg_dgl_eval_epoch("test")
                elif self.type == "cxg":
                    self.cxg_eval_epoch("val")
                    self.cxg_eval_epoch("test")
                log.info(
                    f"val-epoch {epoch} time {self.t_end - self.t_begin} loss {self.loss} metric {self.metric}"
                )
                log.info(
                    f"test-epoch {epoch} time {self.t_end - self.t_begin} loss {self.loss} metric {self.metric}"
                )
