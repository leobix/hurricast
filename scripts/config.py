#========================================================
# Create some configs that we'll be using frequently
#Using dataclass and typing for safety: make sure we have the right number of arguments
# and the right type
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Tuple

@dataclass
class EncoderConfig:
    n_in: int
    n_out: int
    hidden_configuration: Tuple[str, int]

@dataclass
class DecoderConfig:
    n_in_decoder: int
    hidden_configuration_decoder: Tuple[str, int]
    n_out_decoder: Any = None  # Need to defined later 
                                #(depends on the task)
    window_size:  Any = None

@dataclass
class TransformerConfig:
    n_in_decoder: int
    hidden_configuration_decoder: Dict[str, int]
    n_out_transformer: int
    n_out_decoder: Any=None #Need to defined later
                            #(depends on the task)
    window_size: Any=None

@dataclass
class LINEARTransformConfig:
    target_intensity: Any = None
    target_intensity_cat: Any=None
    window_size: Any = None
    n_out_decoder: Any=None #Need to defined later
                            #(depends on the task)

#Enc.
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

#Dec.
encdec_config = dict(
    #out_cnn + number of stat
    n_in_decoder=128 + 10, 
    n_out_decoder=None, 
    hidden_configuration_decoder=(
        ('gru', 128),
        ('gru', 128)
    ))

#Dec.
transformer_config = dict(
    #out_cnn + number of stat
    n_in_decoder=128 + 10,
    n_out_decoder=None,
    n_out_transformer=128,
    hidden_configuration_decoder={
        'nhead': 2,
        'num_layers': 4,
        'dropout': 0.1,  # Default
        'dim_feedforward': 2048  # Default
    })

#Dec
lstm_config = dict(n_in=128+10,
                   hidden_dim=256,
                   rnn_num_layers=2,
                   N_OUT=128,
                   rnn_type='gru',
                   dropout=0.1,
                   activation_fn='tanh',
                   bidir=True)
                   
lineartransform_config = dict()

