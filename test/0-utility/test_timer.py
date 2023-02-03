import torch
import cxgnncomp as cxgc


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = cxgc.TimerOP.apply(x, "l1", 1)
        x = self.linear1(x)
        x = cxgc.TimerOP.apply(x, "l1", 0)
        x = cxgc.TimerOP.apply(x, "l2", 1)
        x = self.linear2(x)
        x = cxgc.TimerOP.apply(x, "l2", 0)
        x = cxgc.TimerOP.apply(x, "l3", 1)
        x = self.linear3(x)
        x = cxgc.TimerOP.apply(x, "l3", 0)
        return x


if __name__ == '__main__':
    model = Model()
    x = torch.randn(2, 2)
    x.requires_grad_(True)
    label = torch.randn(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lossfn = torch.nn.MSELoss()
    cxgc.set_timers()
    x = model(x)
    loss = lossfn(x, label)
    loss.backward()
    optimizer.step()
    cxgc.get_timers().log_all(print)
