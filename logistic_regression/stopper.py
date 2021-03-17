from aenum import Enum, extend_enum


def likelihood(classifier, max_iter_no_imp=5, **kwargs):
    if len(classifier.log_likelihood) >= 5:
        return max(classifier.log_likelihood[-max_iter_no_imp:-1]) == classifier.log_likelihood[-1]
    else:
        return False


class StopCondition(Enum):
    def __new__(cls, function, *args):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)
        obj.evaluate = function
        return obj

    LogLikelihood = likelihood,


class Stopper:
    def __init__(self, max_iter: int = 1000, stop_condition=None, **kwargs):
        self.max_iter = max_iter
        self.n_iter = 0
        self.stop_condition = stop_condition
        self.kwargs = kwargs

    def new_training(self):
        self.n_iter = 0

    def stop(self, classifier, **kwargs) -> bool:
        kwargs.update(self.kwargs)
        self.n_iter += 1
        if self.n_iter >= self.max_iter:
            return True
        # TODO: other criteria
        elif self.stop_condition is not None:
            return self.stop_condition.evaluate(classifier, **kwargs)

        return False

