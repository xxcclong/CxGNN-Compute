import torch
import cxgnncomp
import cxgnndl
from tqdm import tqdm


class Trainer:

    def __init__(self, config):
        self.device = torch.device(config.dl.device)
        self.model = cxgnncomp.get_model(config)
        self.model = self.model.to(self.device)
        self.optimizer = cxgnncomp.get_optimizer(config.train, self.model)
        self.scheduler = cxgnncomp.get_scheduler(config.train, self.optimizer)
        self.loss_fn = cxgnncomp.get_loss_fn(config.train)
        self.loader = cxgnndl.get_loader(config.dl)
        self.type = config.dl.type.lower()
        self.config = config

    def cxg_train_epoch(self):
        self.model.train()
        for batch in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.loss_fn(out, batch.y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def cxg_eval_epoch(self):
        raise NotImplementedError

    def load_subtensor(self, nfeat, labels, seeds, input_nodes, device):
        """
        Extracts features and labels for a subset of nodes
        """
        batch_inputs = nfeat[input_nodes].to(device)
        # batch_inputs = torch.randn([input_nodes.shape[0], feat_len], device=device)
        batch_labels = labels[seeds].to(device)
        return batch_inputs, batch_labels

    def dgl_train_epoch(self):
        self.model.train()
        for (input_nodes, seeds, blocks) in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            self.optimizer.zero_grad()
            blocks = [block.int().to(self.device) for block in blocks]
            batch_inputs, batch_labels = self.load_subtensor(
                self.loader.graph.ndata['features'],
                self.loader.graph.ndata['labels'],
                seeds,
                input_nodes,
                device=self.device)
            batch_pred = self.model([blocks, batch_inputs])
            loss = self.loss_fn(batch_pred, batch_labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        pass

    def train(self):
        if self.type == "dgl":
            for epoch in range(self.config.train.train.num_epochs):
                self.dgl_train_epoch()
                # self.validate_epoch()
        elif self.type == "cxg":
            for epoch in range(self.config.train.train.num_epochs):
                self.cxg_train_epoch()
                # self.cxg_eval_epoch()
