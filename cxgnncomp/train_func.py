import torch


def get_optimizer(config, model):
    lr = float(config.optimizer.lr)
    if config.optimizer.type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif config.optimizer.type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer type {}".format(
            config.optimizer.type))


def get_loss_fn(config):
    if config.model.loss_fn == 'mse':
        return torch.nn.MSELoss()
    elif config.model.loss_fn == 'nll':
        return torch.nn.NLLLoss()
    else:
        raise ValueError("Unknown loss type {}".format(config.model.loss_fn))


def get_scheduler(config, optimizer):
    num_epochs = int(config.train.num_epochs)
    if config.optimizer.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=num_epochs)
    elif config.optimizer.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs)
    # need modification on trainer.py: self.scheduler.step() -> self.scheduler.step(loss)
    elif config.optimizer.scheduler == 'plt':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.8, patience=1000, verbose=True)
    else:
        raise ValueError('Scheduler {} not supported'.format(
            config.optimizer.scheduler))
    return scheduler