
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import sys
from factory import MODEL_REGISTRY, RegisterModel
from typing import (Any, Callable, Dict, List,
                    NewType, Tuple, Union, Optional)


'''
TODO: Add a simple wrapper around a recurrent torch recurrent model
# The idea would be to create the wrapper in which all models could work
1. Encoder woudl be either None, One big CNN or 3 CNNs and would map x_vis--> bs, T, H
2. Fusion module that can concatenate image and stat data
3. The decoder: Could be a Transformer/RNN/or anything that maps bs, T, H --> bs, H
4. Linear Predictor: bs, H_out --> bs, Num_classes


Different ways to go through the forward pass:
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
'''
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
        >> > config = (
            ('conv', 64),
            ('conv', 64),
            ('maxpool', None),
            ('conv', 256),
            ('maxpool', None),
            ('flatten', 256*4*4),
            ('linear', 256),
            ('fc', 10))
        >> > model = CNNEncoder(n_in=9, n_out=10, hidden_config=config)
    """

    def __init__(self,
                 n_in: int,
                 n_out: int,
                 hidden_configuration: Tuple[str, int],
                 kernel_size: int=3,
                 stride: int=1,
                 padding: int=0,
                 groups:int=1,
                 pool_kernel_size: int=2,
                 pool_stride: int=2,
                 pool_padding:int=0,
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


@RegisterModel('ExperimentalHurricast')
class ExperimentalHurricast(nn.Module):
    def __init__(self, 
                decoder_name: str='TRANSFORMER',
                encoder_name: str = 'CNNEncoder',
                decoder_config: Dict[str, int],
                encoder_config: Optional[Dict[str, int]]=None, 
                split_cnns: bool = False):

        self.split_cnns = split_cnns
        self.encoder_config = encoder_config
        self.encoder_name = encoder_name

        self.decoder_name = decoder_name
        self.decoder_config = decoder_config
        
        self.encoder, \
            self.recombine_encoders = self.create_encoder(
            encoder_config, split_cnns)
        
        self.decoder = self.create_decoder()

        self.predictor = self.create_predictor()


    def create_encoder(self, encoder_config, split_cnns):
        """
        HARD CODE THE NUMBER OF CNNS THAT WE ACTUALLY WANT
        """
        if encoder_config is None:
            return None, None
        
        encoder = MODEL_REGISTRY[self.encoder_name](
            **encoder_config)
        recombine_encoders = None
        
        if split_cnns:    
            encoder = _get_clones(encoder, 3)
            recombine_encoders = nn.Linear(
                3 * encoder_config['n_out'],
                encoder_config['n_out'])

        return encoder, recombine_encoders
    
    def create_decoder():
        decoder = MODEL_REGISTRY[self.decoder_name](
            **self.decoder_config)


    #def single_encode(self, x_viz):
    #    """
    #    x_viz: bs, T, num_channels, dim, dim
    #    out: x_viz, T, dim
    #    """
    #    return None
    def _nosplit_encode(self, x_viz):
        """out_cnns: (bs, 1, H_cnn)
        (add a dimension for easy concat)
        """
        return self.encoder(x_viz).unsqueeze(1)


    def _split_encode(self, x_viz):
        """
        x_viz: (bs, num_channels=9, h1, h2)
        out_cnns: (bs, 1, H_cnn)
            (add a dimension for easy concat)
        """
        #out_cnns = torch.zeros(x_viz.size(0), 3,)
        out_cnns = []
        #Split into 3 components across dim 1.
        x_viz = torch.split(x_viz, 3, dim=1)
        #Tuple of tensors with sizes (bs, T, num_channels/3)
        for i, (img, cnn) in enumerate(
            zip(x_viz, self.encoder)):
            #self.encoder: Module List=iterable
            out_cnns.append(cnn(img)) #bs, out_cnn
        #concatenate and Linear Layer
        out_cnns = self.recombine_encoders(
            torch.cat(out_cnns, -1)).unsqueeze(1)
        return out_cnns

    
    def encode(self, x_viz):    
        """
        out_encoder: bs, T, out_cnn
        """
        encode_fn = self._split_encode if self.split_cnns\
                else self._nosplit_encode

        out_encoder = list(map(encode_fn, x_viz.unbind(1))) #Li
        out_encoder = torch.cat(out_encoder, dim=1)
        return out_encoder


    def fusion(self, x_stat, x_viz):
        return torch.cat([x_stat, x_viz], -1)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x_stat, x_viz):
        x_viz = self.encode(x_viz)
        fused_x = self.fusion(x_stat, x_viz)
        
        return self.decode(fused_x)


#TODO: Write DOC

class ExpTRANSFORMER(nn.Module):
    """
    The Transformer keeps the same hidden dimension:
        - input of the net: dim = n_in 
        - output of the net: dim = n_in
    
    If pool_method is "unroll" then a linear layer maps the unrolled
        sequence into one.

    window_size: Necessary if using the unroll pooling method.
    """
    def __init__(self,
                 n_in: int=512,
                 n_head: int=4, 
                 dim_feedforward: int=2048,
                 num_layers: int=6,
                 dropout: float=0.1,
                 window_size: Optional[int]=None, 
                 n_out_unroll: Optional[int]=None,
                 max_len_pe: int=10,
                 pool_method: str='default', 
                 activation: str='tanh'):
        super(ExpTRANSFORMER, self).__init__()
                
        self.n_in = n_in #Input of the Transformer
        self.num_layers = num_layers
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        #Optional
        self.n_out_unroll = n_out_unroll  # In case we use the unroll pooling?
        self.window_size = window_size


        self.transformer_layers = self.create_transformer()
        
        self.pool_method = pool_method
        self.pool_fn = self.create_pooler()
        
        self.activation = activation
        self.activation_fn = self.create_activation_fn()

        self.max_len_pe = max_len_pe
        self.pe = self.create_pe()


    def create_transformer(self):
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_in,
            nhead=self.n_head, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)

        transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=self.num_layers, 
            )
        return transformer_encoder

    def _pool_unroll(self, X):
        #bs, T, H1, H2, H3 --> bs, T*H1*H2*H3
        out = X.flatten(start_dim=1)
        return self.pool_linear(out)

    @staticmethod
    def _pool_mean(X):
        return X.mean(1) 
   
    def create_pooler(self):
        assert self.pool_method in ('default', 'mean', 'unroll'), "\
        Wrong pooling method specified"
        if self.pool_method == 'unroll':
            
            assert self.n_out_unroll is not None, "\
            please specifiy an output dimension if using the unroll\
            pooling method."

            assert self.window_size is not None, "\
            please specifiy a window size if using the unroll\
            pooling method."
            
            self.pool_linear = nn.Linear(
                self.window_size * self.n_in,
                self.n_out_unroll)
            return self._pool_unroll
        
        else:
            return self._pool_mean
    
    def create_activation_fn(self):
        assert self.activation in (
            'relu', 'tanh')
        if self.activation == 'relu':
            return F.relu
        else: #tanh
            return torch.tanh

    def create_pe(self):
        return PositionalEncoding(
            self.n_in, self.dropout,
            max_len=self.max_len_pe)

    def forward(self, x_fuz):
        """
        x_fuz: bs, T, H
        """
        out = self.pe(x_fuz) 
        #out = x_fuz
        #invert bs, T before transformer
        out = self.transformer_layers(out.transpose(0, 1))
        out = out.transpose(0,1) #invert Again
        
        out = self.pool_fn(out)
        
        out = self.activation_fn(out)
        
        return out


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


@RegisterModel('ExperimentalLINEARTransform')
class LINEARTransform(torch.nn.Module):
    """
    Simple Net used to debug.
    Not using x_stat in the forward pass
    """

    def __init__(self, encoder, window_size, target_intensity=False, target_intensity_cat=False):
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
        out_enc = torch.stack(out_enc).transpose(0, 1)  # keep batch first
        #Flatten before passing through the linear layer
        out_enc = out_enc.flatten(start_dim=1)
        out_enc = self.linear(out_enc)
        out_enc = self.activation(out_enc)
        out_enc = self.predictor(out_enc)
        return out_enc


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))

#===================================
def test_transformer():
    input = torch.randn(32,10, 512)
    T = ExpTRANSFORMER(
        pool_method='mean',
        n_out_unroll=100,
        window_size=10,
        activation='relu')
    out = T(input)
    print(out, out.size())
    print('Passed Test')

def test_cnnencoder():
    input = torch.randn(32, 9, 25, 25)
    encoder_config = dict(
        n_in=3 * 3,
        n_out=128,
        hidden_configuration=(
            ('conv', 64),
            ('conv', 64),
            ('maxpool', None),
            ('conv', 256),
            ('maxpool', None),
            ('flatten', 256 * 4 * 4),
            ('linear', 256),
            ('fc', 128)
        ))
    Cnn = CNNEncoder(**encoder_config)
    out = Cnn(input)
    print(out.size())

    new_encoder_config = dict(
        n_in=3,
        n_out=128,
        hidden_configuration=(
            ('conv', 64),
            ('conv', 64),
            ('maxpool', None),
            ('conv', 256),
            ('maxpool', None),
            ('flatten', 256 * 4 * 4),
            ('linear', 256),
            ('fc', 128))
            )

    small_encoder = CNNEncoder(**new_encoder_config)
    cloned_cnns = _get_clones(small_encoder, 3)
    split_images = torch.split(input, 3, dim=1)
    
    for  i, (cnn, img) in enumerate(zip(cloned_cnns, split_images)):
        out_ = cnn(img)
        print(i, out_.size())
    
    input_seq = torch.randn(32,10,9,25,25)
    out_enc = list(map(Cnn, input_seq.unbind(1))) #List of 32, 128
    try:
        print(torch.cat(out_enc).size(), '1')
    except:
        pass
    try:
        print(torch.cat(out_enc).size(), '2')
    except:
        pass
    
    
    


if __name__ == "__main__":
    #test_transformer()
    test_cnnencoder()
