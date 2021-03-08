

class Stopper:
    def __init__(self, max_iter: int = 1000):
        self.max_iter = max_iter
        self.n_iter = 0

    def new_training(self):
        self.n_iter = 0

    def stop(self) -> bool:
        self.n_iter += 1
        if self.n_iter >= self.max_iter:
            return True
        # TODO: other criteria

        return False
