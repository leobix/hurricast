import torch 
import sys
sys.path.append("..")
import src
from src.models import experimental_models as expm

full_encoder_config = dict(
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

split_encoder_config = dict(
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
        ('fc', 128)
    ))

def test_split_encoder(in_model):
    
    #No Stat
    transfo_config = dict(
        n_in=128+10,
        n_head=2,
        dim_feedforward=2048,
        num_layers=64,
        dropout=0.1,
        window_size=None,
        n_out_unroll=None,
        max_len_pe=10,
        pool_method='default',
        activation='tanh')

    Wrap = expm.ExperimentalHurricast(
        n_pred=2,
        decoder_config=transfo_config,
        encoder_config=split_encoder_config,
        decoder_name='ExpTRANSFORMER',
        encoder_name='CNNEncoder',
        split_cnns=True,
        no_stat=False)
    print('Split Encoder, Stat')
    out = Wrap(**in_model)
    print('Out Size', out.size())
    

    transfo_config = dict(
        n_in=128,
        n_head=2,
        dim_feedforward=2048,
        num_layers=64,
        dropout=0.1,
        window_size=None,
        n_out_unroll=None,
        max_len_pe=10,
        pool_method='default',
        activation='tanh')
    
    Wrap = expm.ExperimentalHurricast(
        n_pred=2,
        decoder_config=transfo_config,
        encoder_config=split_encoder_config,
        decoder_name='ExpTRANSFORMER',
        encoder_name='CNNEncoder',
        split_cnns=True,
        no_stat=True)
    print('Split Encoder, No Stat')
    out = Wrap(**in_model)
    print('Out Size', out.size())


def test_full_encoder(in_model):
    transfo_config = dict(
        n_in=128+10,
        n_head=2,
        dim_feedforward=2048,
        num_layers=64,
        dropout=0.1,
        window_size=None,
        n_out_unroll=None,
        max_len_pe=10,
        pool_method='default',
        activation='tanh')

    Wrap = expm.ExperimentalHurricast(
        n_pred=2,
        decoder_config=transfo_config,
        encoder_config=full_encoder_config,
        decoder_name='ExpTRANSFORMER',
        encoder_name='CNNEncoder',
        split_cnns=False,
        no_stat=False)
    print('Full Encoder, Stat')
    out = Wrap(**in_model)
    print('Out Size', out.size())

    transfo_config = dict(
        n_in=128,
        n_head=2,
        dim_feedforward=2048,
        num_layers=64,
        dropout=0.1,
        window_size=None,
        n_out_unroll=None,
        max_len_pe=10,
        pool_method='default',
        activation='tanh')

    Wrap = expm.ExperimentalHurricast(
        n_pred=2,
        decoder_config=transfo_config,
        encoder_config=full_encoder_config,
        decoder_name='ExpTRANSFORMER',
        encoder_name='CNNEncoder',
        split_cnns=False,
        no_stat=True)
    print('Full Encoder, No Stat')
    out = Wrap(**in_model)
    print('Out Size', out.size())


def test_no_encoder(in_model):
    
    transfo_config = dict(
        n_in=10,
        n_head=2,
        dim_feedforward=2048,
        num_layers=64,
        dropout=0.1,
        window_size=None,
        n_out_unroll=None,
        max_len_pe=10,
        pool_method='default',
        activation='tanh')

    Wrap = expm.ExperimentalHurricast(
        n_pred=2,
        decoder_config=transfo_config,
        encoder_config=None,
        decoder_name='ExpTRANSFORMER',
        encoder_name=None,
        split_cnns=True,
        no_stat=False)
    
    print('No Encoder, Stat')
    out = Wrap(**in_model)
    print('Out Size', out.size())
    try: 
        Wrap = expm.ExperimentalHurricast(
        n_pred=2,
        decoder_config=transfo_config,
        encoder_config=None,
        decoder_name='ExpTRANSFORMER',
        encoder_name=None,
        split_cnns=True,
        no_stat=True)

        print('No Encoder, Stat')
        out = Wrap(**in_model)
        print('Out Size', out.size())
    except Exception as e:
        print('Caught', e )

