from collections import OrderedDict
from typing import List

import torch


class ConvolutionalBlock(torch.nn.Module):

    def __init__(self, channels: int, filters: List[int], kernel_sizes: List[int]) -> None:
        """ Convolutional block.

        Args:
            features (int): The input channels
            filters (List[int]): List with the number of channels per convolutional layer in each block
            kernel_sizes (List[int]): List with the kernel sizes per convolutional layer in each block
        """
        super(ConvolutionalBlock, self).__init__()

        if len(filters) != len(kernel_sizes):
            raise ValueError(f'The number of filters and kernel sizes must be the same.')
        n_blocks = len(filters)

        modules = OrderedDict()
        for i in range(n_blocks):
            modules[f'Conv1d_{i}'] = torch.nn.Conv1d(in_channels=channels if i == 0 else filters[i - 1],
                                                     out_channels=filters[i],
                                                     kernel_size=(kernel_sizes[i],),
                                                     padding='same',
                                                     bias=False)

            modules[f'BatchNorm1d_{i}'] = torch.nn.BatchNorm1d(num_features=filters[i], eps=0.001, momentum=0.99)

            modules[f'ReLU_{i}'] = torch.nn.ReLU()

        self.model = torch.nn.Sequential(modules)

    def forward(self, x: torch.Tensor):
        """ Forward pass.

        Args:
            x (torch.Tensor): tensor with shape (batch size, channels, input lenght)

        Returns:
            _type_: _description_
        """

        return self.model(x)


class FullyConvolutionalNetwork(torch.nn.Module):

    def __init__(self,
                 num_classes: int,
                 channels: int,
                 filters: List[int],
                 kernel_sizes: List[int],
                 linear_layer_len: int = 128):
        """ Fully convolutional network implementation.

        Paper:
        Z. Wang, W. Yan and T. Oates
        "Time series classification from scratch with deep neural networks: A strong baseline"
        Proc. Int. Joint Conf. Neural Netw., pp. 1578-1585, 2017.

        Args:
            num_classes (int): The number of classes
            channels (int): The input channels
            filters (List[int]): List with the number of channels per convolutional layer in each block
            kernel_sizes (List[int]): List with the kernel sizes per convolutional layer in each block
            linear_layer_len (int): The size of the classification head
        """

        super(FullyConvolutionalNetwork, self).__init__()
        self.model_name = "FullyConvolutionalNetwork"
        self.fcn = ConvolutionalBlock(channels=channels, filters=filters, kernel_sizes=kernel_sizes)

        # classification head from:
        # Strodthoff N, Wagner P, Schaeffter T, Samek W.
        # "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL."
        # doi: 10.1109/JBHI.2020.3022989
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)

        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=2 * filters[-1])
        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=linear_layer_len)

        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.dropout2 = torch.nn.Dropout(p=0.50)

        self.linear1 = torch.nn.Linear(in_features=2 * filters[-1], out_features=linear_layer_len)
        self.linear2 = torch.nn.Linear(in_features=linear_layer_len, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x (torch.Tensor): tensor with shape (batch size, channels, input lenght)

        Returns:
            y (torch.Tensor): logits, tensor with shape (batch size, num_classes)
        """
        x = x.permute(0, 2, 1)
        h = self.fcn(x)
        h = torch.squeeze(torch.concat([self.avg_pool(h), self.max_pool(h)], dim=1))
        h = self.linear1(self.dropout1(self.batch_norm1(h)))
        h = torch.nn.functional.relu(h)
        h = self.linear2(self.dropout2(self.batch_norm2(h)))
        return h


class Lambda(torch.nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class LSTMBlock(torch.nn.Module):

    def __init__(self, timesteps: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool = False):
        """ LSTM block.

        Args:
            timesteps (int): Length of each time series
            hidden_size (int): Number of units of each LSTM layer
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate applied after each LSTM layer
            bidirectional (bool): Use bidirectional LSTM
        """
        super(LSTMBlock, self).__init__()

        modules = OrderedDict()
        for i in range(num_layers):
            modules[f'LSTM_{i}'] = torch.nn.LSTM(input_size=timesteps if i == 0 else hidden_size,
                                                 hidden_size=hidden_size,
                                                 batch_first=True,
                                                 bidirectional=bidirectional)

            modules[f'Lambda_{i}'] = Lambda(f=lambda x: x[0])

            modules[f'Dropout_{i}'] = torch.nn.Dropout(p=dropout)
        self.model = torch.nn.Sequential(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x (torch.Tensor): tensor with shape (batch size, channels, input lenght)

        Returns:
            _type_: _description_
        """

        #Note the dimension shuffling.
        return self.model(x)[:, -1, :]


class LSTMConvolutionalNetwork(torch.nn.Module):

    def __init__(self,
                 num_classes: int,
                 timesteps: int,
                 channels: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 filters: List[int],
                 kernel_sizes: List[int],
                 linear_layer_len: int = 128,
                 bidirectional: bool = False):
        """ LSTM-FCN implementation.

        Paper:
        F. Karim, S. Majumdar, H. Darabi and S. Chen
        "LSTM fully convolutional networks for time series classification."
        IEEE access 6 (2017): 1662-1669.

        Args:
            num_classes (int): The number of classes
            timesteps (int): Length of each time series
            channels (int): The input channels
            hidden_size (int): Number of units of each LSTM layer
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate applied after each LSTM layer
            filters (List[int]): List with the number of channels per convolutional layer in each block
            kernel_sizes (List[int]): List with the kernel sizes per convolutional layer in each block
            linear_layer_len (int): The size of the classification head
            bidirectional (bool): Use bidirectional LSTM
        """

        super(LSTMConvolutionalNetwork, self).__init__()

        self.fcn = ConvolutionalBlock(channels=channels, filters=filters, kernel_sizes=kernel_sizes)

        self.lstm = LSTMBlock(timesteps=timesteps,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=bidirectional)

        # classification head from:
        # Strodthoff N, Wagner P, Schaeffter T, Samek W.
        # "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL."
        # doi: 10.1109/JBHI.2020.3022989
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)

        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.dropout2 = torch.nn.Dropout(p=0.50)

        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=2 * filters[-1])

        self.linear1 = torch.nn.Linear(in_features=2 * filters[-1], out_features=linear_layer_len)

        if bidirectional:
            self.batch_norm2 = torch.nn.BatchNorm1d(num_features=linear_layer_len + 2 * hidden_size)

            self.linear2 = torch.nn.Linear(in_features=linear_layer_len + 2 * hidden_size, out_features=num_classes)
        else:
            self.batch_norm2 = torch.nn.BatchNorm1d(num_features=linear_layer_len + hidden_size)

            self.linear2 = torch.nn.Linear(in_features=linear_layer_len + hidden_size, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x (torch.Tensor): tensor with shape (batch size, channels, input lenght)

        Returns:
            y (torch.Tensor): logits, tensor with shape (batch size, num_classes)
        """

        h = self.fcn(x)
        h = torch.squeeze(torch.concat([self.avg_pool(h), self.max_pool(h)], dim=1))
        h = self.batch_norm1(h)
        h = self.dropout1(h)
        h = self.linear1(h)
        h = torch.concat([h, self.lstm(x)], dim=-1)
        h = torch.nn.functional.relu(h)
        h = self.batch_norm2(h)
        h = self.dropout2(h)
        h = self.linear2(h)
        return h
