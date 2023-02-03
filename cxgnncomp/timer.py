import torch
import time

_GLOBAL_TIMERS = None


class TimerModule(torch.nn.Module):

    def __init__(self, model: torch.nn.Module, timer_name, report_shape=False):
        super().__init__()
        self.model = model
        self.timer_name = timer_name
        self.report_shape = report_shape

    def forward(self, input):
        input = TimerOP.apply(input, self.timer_name, True)
        output = self.model(input)
        output = TimerOP.apply(output, self.timer_name, False)

        if self.report_shape:
            timers = get_timers()
            timers(self.timer_name + '_shape').input_shape = input.shape
            timers(self.timer_name + '_shape').output_shape = output.shape

        return output


class TimerOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, timer_name, is_front):
        timers = get_timers()
        if is_front:
            timers(timer_name + '_fwd').start()
        else:
            timers(timer_name + '_fwd').stop()

        ctx.timer_name = timer_name
        ctx.is_front = is_front
        return input

    @staticmethod
    def backward(ctx, grad_in):
        timers = get_timers()
        timer_name = ctx.timer_name + '_bwd'
        # print(f"executing backward {ctx.timer_name}")
        if ctx.is_front:
            timers(timer_name).stop()
        else:
            timers(timer_name).start()

        return grad_in, None, None


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()
        self.input_shape = None
        self.output_shape = None

    def start(self):
        """Start the timer."""
        assert not self.started_, 'timer' + self.name_ + 'has already been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer' + self.name_ + ' is not started'
        torch.cuda.synchronize()
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        assert not self.started_, 'timer' + self.name_ + 'is still started'
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def log(self, log_fn, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            if name not in self.timers:
                elapsed_time = 0
            else:
                elapsed_time = self.timers[name].elapsed(
                    reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        log_fn(string)

    def log_all(self, log_fn, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in sorted(self.timers.keys()):
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
            if self.timers[name].input_shape is not None:
                string += '| {}: in:{} out:{}'.format(
                    name, self.timers[name].input_shape,
                    self.timers[name].output_shape)
        log_fn(string)


def set_timers():
    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = Timers()


def get_timers():
    assert _GLOBAL_TIMERS is not None
    return _GLOBAL_TIMERS