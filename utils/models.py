"""
Gather models for hurricane prediction.
Check whether we want to have our own version of attention layers.
https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983
"""
import torch
import torch.nn as nn 
import math
import warnings

#===========================================
# Encoders
# V0
class EncoderCNN(nn.Module):
    """
    #TODO: Ask whether to use dropout / Remove from args ?
    Encoder CNN:
    Implements a sequence of convolutions.
    Uses fixed strides, kernel sizes for now.
    
    IN:
        param: in_channels -  fixed to 3*3 
        param: dropout 
    """
    def __init__(self, dropout=0.5):
        super(EncoderCNN, self).__init__()
        self.in_channels = 3 * 3
        #self.adjust_dim = False
       # if len(levellist) > 1 and len(params) >1:
        #self.adjust_dim = True

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64,
                               kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256,
                               kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.init_weights()

    def init_weights(self):
        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d) and idx == 1:
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.bias, mean=0, std=1)
            if isinstance(m, nn.Conv2d) and idx != 1:
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
                nn.init.normal_(m.bias, mean=0, std=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.view(1, -1)
        return x

#V0
class EncoderLinear1(nn.Module):
    """
    IN:
        param : rnn_decoder - bool
        #TODO: Ask whether to use dropout / Remove from args ?
    """
    def __init__(self, rnn_decoder, dropout=0.5):
        super(EncoderLinear1, self).__init__()

        self.fc1 = nn.Linear(in_features=256*4*4, out_features=576)
        self.fc1_bn = nn.BatchNorm1d(576)
        self.fc2 = nn.Linear(in_features=576, out_features=128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc3_bn = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(in_features=64, out_features=2)
        self.init_weights()
        self.rnn_decoder = rnn_decoder

    def init_weights(self):
        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and idx == 1:
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.bias, mean=0, std=1)
            elif isinstance(m, nn.Linear) and idx != 1:
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
                nn.init.normal_(m.bias, mean=0, std=1)

    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        if self.rnn_decoder:
            return x
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.fc4(x)
        return x

#V0
class EncoderCNNLinear1(nn.Module):
    """#TODO: Ask whether to use dropout / Remove from args ?
    IN:
        param : rnn_decoder - bool
    """
    def __init__(self, dropout=0.5, rnn_decoder=False):
        super(EncoderCNNLinear1, self).__init__()

        self.cnn = EncoderCNN()
        self.linear = EncoderLinear1(rnn_decoder)

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x

# ====================
# Decoders
class DecoderGRU(nn.Module):
    """
    #TODO: Write smmall doc.
    """
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(DecoderGRU, self).__init__()
        self.in_channels = 3 * 3
        #self.adjust_dim = False
       # if len(levellist) > 1 and len(params) >1:
        #self.adjust_dim = True
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size)

    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(self.hidden_size)
        outputs, hiddens = self.rnn(inputs.unsqueeze(0), hidden)
        return outputs, hiddens

    def init_hidden(self, hidden_size):
        return torch.zeros((1, 1, hidden_size))


class EncoderDecoder(nn.Module):
    """
    #TODO: Write small doc.
    """
    def __init__(self, 
                input_size, 
                hidden_size, 
                rnn_decoder=True, 
                dropout=0.5):
        super(EncoderDecoder, self).__init__()
        self.in_channels = 3 * 3
        #self.adjust_dim = False
       # if len(levellist) > 1 and len(params) >1:
        #self.adjust_dim = True
        self.hidden_size = hidden_size
        self.encoder = EncoderCNNLinear1(rnn_decoder=True)
        self.decoder = DecoderGRU(input_size, hidden_size)
        self.lastlinear = nn.Linear(hidden_size, 2)
        self.hidden = self.init_hidden(hidden_size)

    def forward(self, inputs, hidden=None):
        x = self.encoder(inputs)
        o, self.hidden = self.decoder(x, self.hidden)
        y = self.lastlinear(o)
        return y

    def init_hidden(self, hidden_size):
        return torch.zeros((1, 1, hidden_size))

#=========================================
# Baseline(s)
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

#==========================
# New versions of our layers
encoder_config = (
    ('conv', 64),
    ('conv', 64),
    ('maxpool', None),
    ('conv', 256),
    ('maxpool', None),
    ('flatten', 256 * 4 * 4),
    ('linear', 256),
    ('fc', 128)
)

decoder_config = (
    ('gru', 128),
    ('gru', 128)
)


class CNNEncoder(nn.Module):
    """
    CNNEncoder class
    param: n_in
    param: n_out 
    param: hidden_configuration 
        Tuple of tuples to set the configuration of the net.
    Ex:
        >>> config = (
            ('conv', 64), 
            ('conv', 64), 
            ('maxpool', None),
            ('conv', 256),
            ('maxpool', None),
            ('flatten', 256*4*4),
            ('linear', 256), 
            ('fc', 10) )
        >>> model = CNNEncoder(n_in=9, n_out=10, hidden_config=config)
    """

    def __init__(self,
                 n_in,
                 n_out,
                 hidden_configuration,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 pool_kernel_size=2,
                 pool_stride=2,
                 pool_padding=0,
                 activation=nn.ReLU()):
        super(CNNEncoder, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.activation = activation
        assert isinstance(hidden_configuration, tuple)
        self.hidden_configuration = hidden_configuration
        self.layers = self.create_cells()

    def create_vision_cell(self, n_in, n_out):
        cell = [nn.Conv2d(in_channels=n_in,
                          out_channels=n_out,
                          kernel_size=self.kernel_size,
                          stride=self.stride,
                          padding=self.padding,
                          groups=self.groups,
                          bias=True),
                nn.BatchNorm2d(n_out),
                self.activation]
        return cell

    def create_linear_cell(self, n_in, n_out):
        cell = [nn.Linear(in_features=n_in, out_features=n_out),
                nn.BatchNorm1d(n_out),
                self.activation]
        return cell

    def create_maxpool_cell(self, n_in, n_out):
        """
        n_in, n_out are not used here.
        """
        cell = [nn.MaxPool2d(kernel_size=self.pool_kernel_size,
                             stride=self.pool_stride,
                             padding=self.pool_padding)]
        return cell

    def create_cells(self):
        cells = {
            'linear': self.create_linear_cell,
            'conv': self.create_vision_cell,
            'maxpool': self.create_maxpool_cell,
            'fc': lambda n_in, n_out: [nn.Linear(n_in, n_out)],
            'flatten': lambda n_in, n_out: [nn.Flatten()]
        }
        layers = []
        n_prev = self.n_in
        for cell_type, hidden_out in self.hidden_configuration:
            layers.extend(cells[cell_type](n_prev, hidden_out))
            if hidden_out is not None:
                n_prev = hidden_out
        assert n_prev == self.n_out  # Make sure the last layer is correct
        #We can remove that actually
        return nn.Sequential(*layers)

    def forward(self, x):
        #TODO: Neew to reshape or something ?
        return self.layers(x)

    def init_weights(self):
        #TODO: Ask whether we need to create some init. methods really?
        return NotImplementedError


class ENCDEC(nn.Module):
    #TODO: Make sure we can apply the CNN before feeding all of it
    #to the RNN --> I'm affraid we may not backprop correctly. Make sure that works
    """
    The output is just the very last hidden for now.
    """

    def __init__(self,
                 encoder,
                 n_in_decoder: int,
                 n_out_decoder: int,
                 hidden_configuration_decoder: tuple,
                 window_size: int):
        super(ENCDEC, self).__init__()
        self.encoder = encoder

        self.n_in_decoder = n_in_decoder
        self.n_out_decoder = n_out_decoder
        self.hidden_config = hidden_configuration_decoder
        self.window_size = window_size
        self.decoder_cells, \
            self.last_linear = self.create_rec_cells()

    def create_rec_cells(self):
        #Does not work with LSTM for now
        cells = {'gru': nn.GRUCell,
                 'lstm': nn.LSTMCell,
                 'rnn': nn.RNNCell}
        rec_layers = []
        n_prev = self.n_in_decoder
        for cell_type, hidden_out in self.hidden_config:
            rec_layers.append(cells[cell_type](n_prev, hidden_out))
            n_prev = hidden_out
        #Create the last linear as well:
        last_linear = nn.Linear(in_features=hidden_out*self.window_size,
                                out_features=self.n_out_decoder)
        return nn.Sequential(*rec_layers), last_linear

    def forward_rec(self, x, hidden):
        out_prev = x
        out_hidden = []
        for rec_cell, hidden_tensor in zip(self.decoder_cells, hidden):
            out_prev = rec_cell(out_prev, hidden_tensor)
            out_hidden.append(out_prev)
        return out_prev, out_hidden

    def forward(self, x_img, x_stat, predict_at=8):
        """
        x_img: bs, timesteps
        x_stat: bs, timesteps, ....
        """
        bs = x_img.size(0)  # batch_size to init the hidden layers
        hidden = self.init_hidden(bs)
        #list(map(
        #        lambda n: torch.zeros(bs, n),
        #        lambdaself.hidden_config))
        outputs = []
        #List of zeros tensors
        for t in range(x_img.size(1)):
            x = self.encoder(torch.select(x_img, axis=1, index=t))
            #FUSION
            x = torch.cat([x, torch.select(x_stat, axis=1, index=t)],
                          axis=1)
            out, hidden = self.forward_rec(x, hidden)
            outputs.append(out)
        outputs = torch.stack(outputs).transpose(1, 0)

        #Final transformation
        outputs = self.last_linear(outputs.flatten(start_dim=1))
        return outputs

    def init_hidden(self, bs):
        hidden_dims = list(map(lambda x: getattr(
            x, 'hidden_size'),  self.decoder_cells))
        hidden = [torch.zeros(bs, dim) for dim in hidden_dims]
        return hidden


class TRANSFORMER(nn.Module):
    """

    """
    def __init__(self, 
                encoder, 
                d_model, 
                n_head, 
                n_transformer_layer):
        super(TRANSFORMER, self).__init__()
        warnings.warn('Using a Transformer model without positional encoding', 
                UserWarning)
        self.encodercnn = encoder
        self.d_model = d_model
        self.n_head = n_head
        self.n_transformer_layer = n_head
        self.transformer_layers = self.create_layers()
        assert hasattr(self, 'encodercnn')
    
    def create_layers(self):
        layers = [self.encodercnn]
        layers.extend([nn.TransformerEncoderLayer(
                                d_model=self.d_model, 
                                nhead=self.n_head)] 
                                * self.n_transformer_layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        #Naive way
        out = []
        for x_ in x.unbind(1): #Loop over the sequence
            out.append(self.encodercnn(x_))
        out = torch.stack(out).transpose(0, 1)
        return out#self.transformer_layers(out)


class LINEARTransform(torch.nn.Module):
    """
    Simple Net used to debug.
    """
    def __init__(self, encoder, window_size, target_intensity = False):
        super(LINEARTransform, self).__init__()
        self.encoder = encoder
        self.linear = torch.nn.Linear(128 * window_size, 128)
        self.target_intensity = target_intensity
        if self.target_intensity:
            self.predictor = torch.nn.Linear(128, 1)
            self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
        else:
            self.predictor = torch.nn.Linear(128, 2)
            self.activation = torch.nn.ReLU()
    
    def forward(self, x_viz, x_stat):
        #Apply the econder to all the elements 
        out_enc = list(map(self.encoder, x_viz.unbind(1)))
        out_enc = torch.stack(out_enc).transpose(0,1) #keep batch first
        #Flatten before passing through the linear layer
        out_enc = out_enc.flatten(start_dim=1)
        out_enc = self.linear(out_enc)
        out_enc = self.activation(out_enc)
        out_enc = self.predictor(out_enc)
        return out_enc


if __name__ == "__main__":
    encoder_config = (
        ('conv', 64),
        ('conv', 64),
        ('maxpool', None),
        ('conv', 256),
        ('maxpool', None),
        ('flatten', 256 * 4 * 4),
        ('linear', 512)
    )
    encoder = CNNEncoder(n_in=9,
                 n_out=512,
                 hidden_configuration=encoder_config)
    x = torch.randn(10,9,25,25)
    print(x.size())
    out = encoder(x)
    print(out.size())

    transfo = TRANSFORMER(encoder, 
                d_model=512, 
                n_head=4, 
                n_transformer_layer=3)
    x = torch.randn(10,8, 9,25,25)
    out = transfo(x)
    print(out.size())
