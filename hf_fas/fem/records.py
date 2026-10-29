"""Small aggregate work counters; never retain vectors or matvec histories."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter


_timing_stack = ContextVar('fem_timing_stack', default=())


def timed(name, counter=None):
    """Aggregate wall time, subtracting nested regions even across owners."""
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            work = counter(*args, **kwargs) if counter else getattr(args[0], 'work', None)
            if work is None:
                return function(*args, **kwargs)
            with work.measure(name):
                return function(*args, **kwargs)
        return call
    return decorate


class WorkCounter:
    def __init__(self):
        self.values = Counter()

    def add(self, name, amount=1):
        self.values[name] += amount

    @contextmanager
    def measure(self, name):
        stack = _timing_stack.get()
        frame = [perf_counter(), 0.]
        token = _timing_stack.set(stack+(frame,))
        try:
            yield
        finally:
            elapsed = perf_counter()-frame[0]
            _timing_stack.reset(token)
            if stack:
                stack[-1][1] += elapsed
            self.add(f'time.{name}.calls')
            self.add(f'time.{name}.inclusive_seconds', elapsed)
            self.add(f'time.{name}.exclusive_seconds', max(0., elapsed-frame[1]))

    def snapshot(self):
        return dict(self.values)

    def difference(self, before):
        return {key:value-before.get(key,0) for key,value in self.values.items()}


def model_work(model):
    return dict(model=model.work.snapshot(), coulomb=model.coulomb.work.snapshot(),
                mass=model.space.mass_inverse.work.snapshot(),
                pair_mass=model.space.pair_mass_inverse.work.snapshot(),
                auxiliary=model.space.auxiliary_inverse.work.snapshot(),
                kinetic=model.space.kinetic_inverse.work.snapshot())


def work_difference(model, before):
    current = model_work(model)
    return {group:{key:value-before[group].get(key,0) for key,value in counts.items()}
            for group,counts in current.items()}
