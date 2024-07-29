
class EarlyStopping:
    def __init__(self, patience = 5, min_delta=0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta

        self.pre_loss = float('inf')
        self.no_improvement_count = 0

    def should_stop(self, epoch_loss, epoch):
        if epoch == 0 or (epoch_loss < self.pre_loss - self.min_delta):
            self.pre_loss = epoch_loss
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1
            return self.no_improvement_count >= self.patience