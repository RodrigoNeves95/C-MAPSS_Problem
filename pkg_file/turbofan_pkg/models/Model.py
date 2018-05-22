import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torch.optim as optim

from turbofan_pkg import Trainer
from .QRNN import QRNN
from .TCN import TemporalConvNet
from .DRNN import DRNN

SEED = 1337


class RNNModel(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=10,
                 num_layers=1,
                 hidden_size=10,
                 cell_type='LSTM'):
        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_cell = None
        self.cell_type = cell_type
        self.output_size = output_size
        self.kernel_size = kernel_size

        assert self.cell_type in ['LSTM', 'RNN', 'GRU', 'QRNN', 'TCN', 'DRNN'], \
            'Not Implemented, choose on of the following options - ' \
            'LSTM, RNN, GRU'

        if self.cell_type == 'LSTM':
            self.encoder_cell = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        if self.cell_type == 'GRU':
            self.encoder_cell = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        if self.cell_type == 'RNN':
            self.encoder_cell = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        if self.cell_type == 'QRNN':
            self.encoder_cell = QRNN(self.input_size, self.hidden_size, self.num_layers, self.kernel_size)
        if self.cell_type == 'DRNN':
            self.encoder_cell = DRNN(self.input_size, self.hidden_size, self.num_layers)  # Batch_First always True
        if self.cell_type == 'TCN':
            self.encoder_cell = TemporalConvNet(self.input_size, self.hidden_size, self.num_layers, self.kernel_size)

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,
                x,
                hidden=None):
        outputs, hidden_state = self.encoder_cell(x,
                                                  hidden)  # returns output variable - all hidden states for seq_len, hindden state - last hidden state
        outputs = self.output_layer(outputs)

        return outputs

    def predict(self,
                x,
                hidden=None):

        prediction, _ = self.encoder_cell(x, hidden)
        prediction = self.output_layer(prediction)

        return prediction


class RNNTrainer(Trainer):
    def __init__(self,
                 lr,
                 number_steps_train,
                 hidden_size,
                 num_layers,
                 cell_type,
                 batch_size,
                 num_epoch,
                 number_features_input=1,
                 number_features_output=1,
                 kernel_size=None,
                 loss_function='MSE',
                 optimizer='Adam',
                 normalizer='Standardization',
                 use_scheduler=False,
                 validation_split=0.2,
                 **kwargs):
        """
        Recurrent + CNN models Trainer

        Parameters
        ----------
        lr : float

        number_steps_train : int
            Sequence train length

        hidden_size : int

        num_layers : int

        cell_type : str
            Choose model to implement

        batch_size : int

        num_epoch : int

        number_features_input : int
            Number of features in input

        number_features_output : int
            Number of features in output

        kernel_size : int, optional, default : None
            Kernel size for convolutional models

        loss_function : str, default : Adam
            Loss function to use. Currently implemented : MSE, MAE

        optimizer : str, default : MSE
            Optimizer to use. Currently implemented : Adam, SGD, RMSProp, Adadelta, Adagrad

        normalizer : str, default : Standardization
            Normalizer for the data

        use_scheduler : boolean, default : False
            If True use learning rate scheduler

        validation_split : int
            Validation split ratio

        kwargs : **
        """

        super(RNNTrainer, self).__init__(**kwargs)

        torch.manual_seed(SEED)

        # Hyper-parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.number_features_input = number_features_input
        self.number_features_output = number_features_output
        self.number_steps_train = number_steps_train
        self.lr = lr
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.use_scheduler = use_scheduler
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.normalizer = normalizer
        self.validation_split = validation_split

        self.file_name = self.filelogger.file_name

        # Save metadata model
        metadata_key = ['number_steps_train',
                        'cell_type',
                        'hidden_size',
                        'kernel_size',
                        'num_layers',
                        'lr',
                        'batch_size',
                        'num_epoch']

        metadata_value = [self.number_steps_train,
                          self.cell_type,
                          self.hidden_size,
                          self.kernel_size,
                          self.num_layers,
                          self.lr,
                          self.batch_size,
                          self.num_epoch]

        metadata_dict = {}
        for i in range(len(metadata_key)):
            metadata_dict[metadata_key[i]] = metadata_value[i]

        # check if it's to load model or not
        if self.filelogger.load_model is not None:
            self.load(self.filelogger.load_model)
            print('Load model from {}'.format(
                self.logger_path + self.file_name + 'model_checkpoint/' + self.filelogger.load_model))
        else:
            self.model = RNNModel(self.number_features_input,
                                  self.number_features_output,
                                  self.kernel_size,
                                  self.num_layers,
                                  self.hidden_size,
                                  self.cell_type)

            print(metadata_dict)
            self.filelogger.write_metadata(metadata_dict)

        # loss function
        if loss_function == 'MSE':
            self.criterion = nn.MSELoss()

        # optimizer
        if optimizer == 'Adam':
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == 'SGD':
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        elif optimizer == 'RMSProp':
            self.model_optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif optimizer == 'Adadelta':
            self.model_optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        elif optimizer == 'Adagrad':
            self.model_optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr)

        if self.use_scheduler:
            self.scheduler = ReduceLROnPlateau(self.model_optimizer, 'min', patience=2, threshold=1e-5)

        # check CUDA availability
        if self.use_cuda:
            self.model.cuda()

    def init_weights(self,
                     m):
        if type(m) in [nn.LSTM, nn.GRU, nn.RNN]:
            print(m)
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant(param, 0.00)
                elif 'weight' in name:
                    nn.init.xavier_normal(param)
        if type(m) in [nn.Linear, nn.Conv1d]:
            print(m)
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.00)

    def training_step(self):

        self.model_optimizer.zero_grad()
        loss = 0
        X, Y = next(self.datareader.train_generator)
        length = X.shape[0]
        X = Variable(torch.from_numpy(X)).float().cuda()
        Y = Variable(torch.from_numpy(Y)).float()

        results = self.model(X)

        loss = self.criterion(results, Y.unsqueeze(2).cuda())

        loss.backward()
        self.model_optimizer.step()

        return loss.data[0], loss.data[0] * length

    def evaluation_step(self):

        X, Y = next(self.datareader.validation_generator)
        length = X.shape[0]
        X = Variable(torch.from_numpy(X), requires_grad=False, volatile=True).float().cuda()
        Y = Variable(torch.from_numpy(Y), requires_grad=False, volatile=True).float().cuda()

        results = self.model.predict(X)

        valid_loss = self.criterion(results, Y.unsqueeze(2).cuda())

        return valid_loss.data[0], valid_loss.data[0] * length

    def prediction_step(self):

        X, Y = next(self.datareader.test_generator)
        X = Variable(torch.from_numpy(X), requires_grad=False, volatile=True).float().cuda()

        results = self.model.predict(X)

        return results, Y