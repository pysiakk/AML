from aenum import Enum, extend_enum


def worse_than_worst(classifier, max_iter_no_imp=5, imp_coef=0.0001, **kwargs):
    if len(classifier.log_likelihood) >= max_iter_no_imp:
        return min(classifier.log_likelihood[-max_iter_no_imp:-1]) >= (1 + imp_coef) * classifier.log_likelihood[-1]
    else:
        return False


class StopCondition(Enum):
    def __new__(cls, function, *args):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)
        obj.evaluate = function
        return obj

    WorseThanWorst = worse_than_worst,


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

