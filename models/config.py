import json

# self.x_dim = 784
# self.h_dims = [512, 256]  # hidden dims in enc and dec
# self.z_dim = 8  # latent dim size
# self.num_categories = 4  # number of categories

# # Training Hyperparameters
# self.learning_rate = 1e-3
# self.batch_size = 64
# self.num_epochs = 100

# # Gumbel-Softmax Temperatures
# self.tau1 = 0.67
# self.tau2 = 1.0

# # Device
# self.device = device


class Config:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            try:
                json.dumps(val)
                setattr(self, key, val)
            except TypeError:
                # value not json serializable
                continue

        super().__init__()

    def __getattribute__(self, key):
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            raise e
