import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import functools
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MLPModule(nn.Module):
    def __init__(self, hidden_layer_sizes, activation, num_features, num_outputs) -> None:
        super().__init__()
        layers_list = list()
        for i in range(len(hidden_layer_sizes)-1):
            layers_list.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1])),
            layers_list.append(activation())
        hidden_layers = nn.Sequential(*layers_list)
        self.layers = nn.Sequential(
            nn.Linear(num_features, hidden_layer_sizes[0]),
            activation(),
            hidden_layers,
            nn.Linear(hidden_layer_sizes[-1], num_outputs)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class MLPClassifierGPU(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu", *,
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000) -> None:
        self.is_model_built = False
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha  # L2 regularization term
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter  # number of epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun

    def _build_model(self):
        self.model = MLPModule(self.hidden_layer_sizes, self._get_activation_function(), self.num_features, self.num_outputs)
        self.is_model_built = True

    def _get_activation_function(self):
        if self.activation == 'relu':
            return functools.partial(nn.ReLU, inplace=True)
        raise ValueError('Invalid activation')

    def predict(self, X):
        if not self.is_model_built:
            raise ValueError('Please run fit before calling perdict')

        y_prob = self.predict_proba(X)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred

    def predict_proba(self, X):
        if not self.is_model_built:
            raise ValueError('Please run fit before calling perdict')

        with torch.no_grad():
            X_tensor = torch.Tensor(X)
            X_tensor = X_tensor.to('cuda')
            y_prob = self.model(X_tensor)
            y_prob = torch.softmax(y_prob, dim=1)
        return y_prob.cpu().numpy()

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns a trained MLP model.
        """
        self.n_samples = X.shape[0]
        self.num_features = X.shape[1]
        self.num_outputs = int(y.max()+1) if y.ndim == 1 else y.shape[1]

        if not self.is_model_built:
            self._build_model()

        if self.batch_size == 'auto':
            batch_size = min(200, self.n_samples)
        else:
            batch_size = self.batch_size

        device = torch.device('cuda')

        self.model = self.model.to(device)

        X_tensor = torch.Tensor(X)
        y_tensor = torch.Tensor(y).to(dtype=torch.long)

        def tensor_memory_size(a):
            return a.element_size() * a.nelement()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        dataset_size = tensor_memory_size(X_tensor) + tensor_memory_size(y_tensor)
        if dataset_size < gpu_memory:
            X_tensor = X_tensor.to(device)
            y_tensor = y_tensor.to(device)

        # dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        # dataloder = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=0)
        optimizer = self._get_optimizer()

        loss_fn = nn.CrossEntropyLoss() if self.num_outputs > 1 else nn.BCELoss()
        n_iterations_no_change = 0

        samples_indexes = torch.arange(0, self.n_samples)
        # print(samples_indexes)

        for _ in range(self.max_iter):
            # for inp, target in dataloder:
            for idx in range(0, self.n_samples, batch_size):
                self.model.zero_grad()
                i = samples_indexes[idx:idx+batch_size]
                inp = X_tensor[i]
                inp = inp.to(device)
                target = y_tensor[i]
                target = target.to(device)

                y_pred = self.model.forward(inp)
                loss = loss_fn(y_pred, target)
                loss.backward()

                optimizer.step()

                loss_value = loss.item()
                if loss_value < self.tol:
                    n_iterations_no_change += 1
                    if n_iterations_no_change == self.n_iter_no_change:
                        return
                else:
                    n_iterations_no_change = 0

            if self.shuffle:
                samples_indexes = torch.randperm(self.n_samples)

    def _get_optimizer(self):
        if self.solver == 'adam':
            return optim.Adam(self.model.parameters(), self.learning_rate_init, betas=(self.beta_1, self.beta_2), eps=self.epsilon, weight_decay=self.alpha)
        raise ValueError('Invalid solver')
