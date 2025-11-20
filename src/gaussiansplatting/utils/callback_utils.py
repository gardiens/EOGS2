class early_stopping:
    def __init__(
        self, patience: int = 5, operator: str = "min", metric_name: str = "psnr"
    ):
        self.patience = patience
        if operator == "min":
            self.best_loss = float("inf")
        elif operator == "max":
            self.best_loss = -float("inf")
        self.counter = 0
        self.early_stop = False
        self.operator = operator
        self.metric_name = metric_name

    def __call__(self, metric_dict: dict):
        metric = metric_dict[self.metric_name]
        if metric == 0:
            print("the metric is 0 , we skip the early stopping")
            return
        if self.operator == "min":
            if metric < self.best_loss:
                self.best_loss = metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("we exceeded the patience")
                    print("best loss was ", self.best_loss)

                    self.early_stop = True
        elif self.operator == "max":
            if metric > self.best_loss:
                self.best_loss = metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("we exceeded the patience")
                    print("we exceeded the patience")
                    print("best loss was ", self.best_loss)
                    self.early_stop = True
        else:
            raise ValueError("operator should be either min or max")
        return self.early_stop
