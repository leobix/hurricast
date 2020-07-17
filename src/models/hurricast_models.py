"""
Gather models for hurricane prediction.
Check whether we want to have our own version of attention layers.
https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983\


One way to implement those models would to have that CNN + Decoder
interaction could be:
1. x : (bs, T, H) 
    x.flatten(end_dim=1) (bs*T, H) --> CNN (bs*T, H_out)--> Reshape
    ENCODER Over 

"""
import torch
import torch.nn as nn 
import math
import warnings
import sys
from factory import MODEL_REGISTRY, RegisterModel
#==========================
# New versions of our layers
@RegisterModel('CNNEncoder')
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


@RegisterModel('ENCDEC')
class ENCDEC(nn.Module):
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
        last_linear = nn.Linear(
            in_features=hidden_out*self.window_size,
            out_features=self.n_out_decoder)

        return nn.Sequential(*rec_layers), last_linear

    def forward_rec(self, x, hidden):
        out_prev = x
        out_hidden = []
        for rec_cell, hidden_tensor in zip(self.decoder_cells, hidden):
            out_prev = rec_cell(out_prev, hidden_tensor)
            out_hidden.append(out_prev)
        return out_prev, out_hidden

    def forward(self, x_viz, x_stat, predict_at=8):
        """
        x_img: bs, timesteps
        x_stat: bs, timesteps, ....
        """
        bs = x_viz.size(0)  # batch_size to init the hidden layers
        hidden = self.init_hidden(bs)
        #list(map(
        #        lambda n: torch.zeros(bs, n),
        #        lambdaself.hidden_config))
        outputs = []
        #List of zeros tensors
        for t in range(x_viz.size(1)):
            x = self.encoder(torch.select(x_viz, axis=1, index=t))
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


'''
TODO: Add a simple wrapper around a recurrent torch recurrent model
# The idea would be to create the wrapper in which all models could work
1. Encoder woudl be either None, One big CNN or 3 CNNs and would map x_vis--> bs, T, H
2. Fusion module that can concatenate image and stat data 
3. The decoder: Could be a Transformer/RNN/or anything that maps bs, T, H --> Class
'''

class ExperimentalENCDEC(nn.Module):
    def __init__(self):
        return None

    def fwd_viz(self, x_viz):
        return x_viz

    def fusion(self, x_stat, x_viz):
        return torch.cat([x_stat, x_viz], -1)

    def decode(self, x):
        return x
    
    def forward(self, x_stat, x_viz):
        x_viz = self.fwd_viz(x_viz)
        fused_x = self.fusion(x_stat, x_viz)
        return self.decode(fused_x)



@RegisterModel('TRANSFORMER')
class TRANSFORMER(nn.Module):
    """
    #TODO: Test
    #TODO: Add positional encoding
    """

    def __init__(self,
                 encoder,
                 n_in_decoder: int,
                 n_out_transformer: int, 
                 n_out_decoder: int,
                 hidden_configuration_decoder: dict,
                 window_size: int):
        super(TRANSFORMER, self).__init__()
        warnings.warn('Using a Transformer model without positional encoding',
                      UserWarning)
        self.encodercnn = encoder
        self.n_in_decoder = n_in_decoder
        self.n_out_decoder = n_out_decoder
        self.window_size = window_size
        self.config = hidden_configuration_decoder
        
        self.n_out_transformer = n_out_transformer


        self.transformer_layers = self.create_transformer()
        self.last_linear = self.create_linear()
        
        #Leonard: That's here for now, we'll use a wrapper later on
        self.predictor = nn.Linear(self.n_out_transformer, 
                                   self.n_out_decoder)

    def create_transformer(self):
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_in_decoder,
            nhead=self.config['nhead'])
        transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=self.config['num_layers'])
        return transformer_encoder

    def create_linear(self):
        return nn.Linear(self.window_size * self.n_in_decoder,
                         self.n_out_transformer)

    def forward(self, x_viz, x_stat):
        #Naive way
        out = []
        for x_ in x_viz.unbind(1):  # Loop over the sequence
            out.append(self.encodercnn(x_))
        out = torch.stack(out).transpose(0, 1)
        out = torch.cat([out, x_stat], axis=-1)
        out = self.transformer_layers(out)
        out = self.last_linear(out.flatten(start_dim=1))
        #TODO: Remove it when using wrapper
        out = self.predictor(out)
        return out


@RegisterModel('LINEARTransform')
class LINEARTransform(torch.nn.Module):
    """
    Simple Net used to debug.
    """
    def __init__(self, encoder, window_size, target_intensity = False, target_intensity_cat = False):
        super(LINEARTransform, self).__init__()
        self.encoder = encoder
        self.linear = torch.nn.Linear(128 * window_size, 128)
        self.target_intensity = target_intensity
        self.target_intensity_cat = target_intensity_cat
        if self.target_intensity:
            self.predictor = torch.nn.Linear(128, 1)
            self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
        elif self.target_intensity_cat:
            self.predictor = torch.nn.Linear(128, 7)
            self.activation = torch.nn.ReLU()
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



class PositionalEncoding(nn.Module):
    "Implement the Positional Encoding function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (bs, T)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



if __name__ == "__main__":

    def test():
        print(MODEL_REGISTRY)
        encoder_config = (
            ('conv', 64),
            ('conv', 64),
            ('maxpool', None),
            ('conv', 256),
            ('maxpool', None),
            ('flatten', 256 * 4 * 4),
            ('linear', 512))
        encoder = CNNEncoder(n_in=9,
                        n_out=512,
                        hidden_configuration=encoder_config)
    
    
        x = torch.randn(10,9,25,25)
        print(x.size())
        out = encoder(x)
        print(out.size())
        

        transfo_config = dict(
            n_in_decoder=128 + 10,
            n_out_decoder=None,
            n_out_transformer=128,
            hidden_configuration_decoder={
                'nhead': 2,
                'num_layers': 4,
                'dropout': 0.1,  # Default
                'dim_feedforward': 2048  # Default
            })
        transfo = TRANSFORMER(**transfo_config)
        
        x = torch.randn(10,8, 9,25,25)
        out = transfo(x)
        print(out.size())
    
    test()
