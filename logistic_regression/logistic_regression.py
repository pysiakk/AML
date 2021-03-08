

class Classifier:
    pass


class IWLS(Classifier):
    pass


class GeneralGradientDescent(Classifier):

    def __init__(self, batch_size: int):
        self.batch_size: int = batch_size


class GD(GeneralGradientDescent):

    def __init__(self):
        super().__init__(-1)


class SGD(GeneralGradientDescent):

    def __init__(self):
        super().__init__(1)


class MiniBatchGD(GeneralGradientDescent):

    def __init__(self):
        super().__init__(32)

