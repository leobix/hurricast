from utils import models
import argparse
import json
import torch
import numpy as np
import sys
import torch 
from torch.utils.tensorboard import SummaryWriter
sys.path.append('../')

#TODO: Put all the shortcuts in __init__ 
#===========================================
# Create some configs that we'll be using frequently
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

encdec_config = (
    ('gru', 128),
    ('gru', 128)
)

lineartransform_config = (
    None,
)

transformer_config = {'d_model': 128,
                      'nhead': 4,
                      'num_layers': 4
                      }
#======================================================
# Store dict of configurations that we'll be using 
encoder_args = dict(n_in=3*3,
                    n_out=128,
                    hidden_configuration=encoder_config)

encdec_args = dict(n_in_decoder=128+6,
                   n_out_decoder=128,
                   hidden_configuration_decoder=encdec_config,
                   window_size=8)

lineartransform_args = dict(n_in_decoder=128,
                            n_out_decoder=128,
                            hidden_configuration_decoder=lineartransform_config,
                            window_size=8)

transformer_args = dict(n_in_decoder=128+6,
                        n_out_decoder=128+6,
                        hidden_configuration_decoder=transformer_config,
                        window_size=None)
#========================================================
# Shortcuts for our models
models_names = {'LINEARTransform': models.LINEARTransform,
          'ENCDEC': models.ENCDEC,
          'TRANSFORMER': models.TRANSFORMER,
          'CNNEncoder': models.CNNEncoder}


configs = {'LINEARTransform': lineartransform_config,
          'ENCDEC':encdec_config,
          'TRANSFORMER': transformer_config,
          'CNNEncoder': None}


stored_args = {'LINEARTransform': lineartransform_args,
               'ENCDEC': encdec_args,
               'TRANSFORMER': transformer_config,
               'CNNEncoder': encoder_args}

def add_model_parser(parser):

    #parser.add_argument('--n_per_hidden_layer_conf',
    #                   help=' Hidden layer sizes. Example of use: \
    #                        python *.py ... --n_per_hidden_layer 512 512 512',
    #                   nargs='+',
    #                   type=int,
    #                   default=[512])

    parser.add_argument('--n_epochs',
                        help='Number of epochs to train',
                        type=int,
                        default=10)

    parser.add_argument('--batch_size',
                        help='Number of data per batch',
                        type=int,
                        default=256)

    parser.add_argument('--lr',
                        help="learning rate",
                        type=float,
                        default=0.001)

    parser.add_argument('--dropout',
                        type=float,
                        help='Dropout proba. applied to the recurrent\
                            layers of out Encoder Decoder',
                        default=0.)
    
    parser.add_argument('--l2_reg',
                       type=float,
                       help='L2 Penalization for the training',
                       default=0.)

    parser.add_argument('--gpu_nb', 
                        type=int,
                        help='GPU index. Use -1 for cpu', 
                        default=-1)

    parser.add_argument('--target_intensity',
                        action="store_true",
                        help='Predict intensity (windspeed) instead of displacement if enabled')

    parser.add_argument('--encdec',
                        action="store_true",
                        help='Decide if ENCDEC')
            
    parser.add_argument('--save',
                        action="store_true",
                        help='Decide if you want to save the model')

    parser.add_argument('--sgd',
                        action="store_true",
                        help='Decide if you want to use SGD over Adam')
    
    parser.add_argument('--encoder_name', 
                            type=str, 
                            default='CNNEncoder')

    parser.add_argument('--decoder_name', 
                            type=str,
                            default='LINEARTransform')
    return parser


def add_data_parser(parser):
    """
    Add path to folders, 
    and arguments to preprocess the files.
    """

    parser.add_argument('--vision_name',
                        type=str,
                        help='Directory to save the results',
                        default='vision_data.npy')

    parser.add_argument('--y_name',
                        type=str,
                        help='Directory to save the results',
                        default='y.npy')

    parser.add_argument('--output_dir',
                            type=str,
                            help='Directory to save the results',
                            default='./results/results_0')
    
    parser.add_argument('--data_dir', 
                        type=str, 
                        help='Path to the director file',
                        default='./data')

    parser.add_argument('--predict_at', 
                        type=int, 
                        default=8, 
                        help='Number of timesteps in the future of the element\
                            we want to predict.')
    
    parser.add_argument('--window_size', 
                        type=int,
                        default=8, 
                        help='Number of data points to consider as part of a sentence.'
                        )
    parser.add_argument('--train_test_split',
                        type=float, 
                        default=0.8)

    return parser


def create_seeds(torch_seed=0, np_seed=2020):
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)


def create_setup():
    """
    
    """
    parser = argparse.ArgumentParser()
    parser = add_model_parser(parser)
    parser = add_data_parser(parser)

    args = parser.parse_args()
    
    #args = parser.parse_args()
    import os.path as osp
    import os

    if osp.exists(args.output_dir):
        import datetime
        now = datetime.datetime.now()
        now_ = '{}_{}_{}_{}'.format(now.month,
                                    now.day,
                                    now.hour,
                                    now.minute)

        args.output_dir = './results/results'+now_
    else:
        pass

    print('Creating {}'.format(args.output_dir))
    os.mkdir(args.output_dir)

    with open(osp.join(args.output_dir, 'args.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)

    return args


def create_board(args, model, configs:list):
    writer = SummaryWriter(args.output_dir)  # Add tensorboard
    #writer.add_hparams(hparam_dict=vars(args),
    #                   metric_dict={'accuracy': torch.tensor(0)})
    config_ = ""
    for config in configs:
        config_ += "{}\n".format(config)
    writer.add_text('Args', json.dumps(vars(args), indent=2))
    writer.add_text('Model', json.dumps(model.__str__(), indent=4))
    writer.add_text('Configs', config_)
    return writer
    

def create_model(encoder_name: str,
                 decoder_name: str,
                 wrapper_args:dict, 
                 args):
    
    stored_args[decoder_name]['window_size'] = args.window_size

    encoder = models_names[encoder_name](**stored_args[encoder_name])
    decoder = models_names[decoder_name](encoder, **stored_args[decoder_name])
    model = models.WRAPPER(model=decoder, **wrapper_args)
    return model


def create_loss_fn(mode='intensity'):
    """
    Wrappers that uses same signature for all loss_functions.
    Can easily add new losses function here.
    #TODO: Théo--> See if we can make an entropy based loss.
    
    """
    assert mode in ['displacement',
                    'intensity']  # , 'sum']

    base_loss_fn = torch.nn.MSELoss()

    def displacement_loss(model_outputs,
                          target_displacement,
                          target_intensity):
        return base_loss_fn(model_outputs,
                            target_displacement)

    def intensity_loss(model_outputs,
                       target_displacement,
                       target_intensity):
        return base_loss_fn(model_outputs,
                            target_intensity)

    losses_fn = dict(displacement=displacement_loss,
                     intensity=intensity_loss)
    return losses_fn[mode]


def create_optimizer(model, sgd=False, lr=1e-4):
    if sgd:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=lr)
    return optimizer
    

