from utils import models
import argparse
import json
import torch
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter
sys.path.append('../')


models = {
        'TRANSFORMER': models.TRANSFORMER, 
        'CNNEncoder': models.CNNEncoder, 
        'ENCDEC': models.ENCDEC, 
        'LINEARTransform': models.LINEARTransform
        }

#========================================================
# Create some configs that we'll be using frequently
encoder_config = (
    ('conv', 64),
    ('conv', 64),
    ('maxpool', None),
    ('conv', 256),
    ('maxpool', None),
    ('flatten', 256 * 4 * 4),
    ('linear', 576),
    ('linear', 256),
    ('fc', 128)
)

decoder_config = decoder_config = (
    ('gru', 128),
    ('gru', 64)
)

#======================================================

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

    parser.add_argument('--target_intensity',
                        action="store_true",
                        help='Predict intensity (windspeed) instead of displacement if enabled')
    parser.add_argument('--target_intensity_cat',
                        action="store_true",
                        help='Predict intensity category (windspeed) instead of displacement if enabled')
    parser.add_argument('--encdec',
                        action="store_true",
                        help='Decide if ENCDEC')

    parser.add_argument('--save',
                        action="store_true",
                        help='Decide if you want to save the model')

    parser.add_argument('--normalize',
                        action="store_true",
                        help='Decide if vision and stats weights are normalized or standardized')

    parser.add_argument('--normalize_intensity',
                        action="store_true",
                        help='Decide if intensity target is normalized or standardized')
    parser.add_argument('--test_nostat',
                        action="store_true",
                        help='Decide if using stat data for GRU')

    parser.add_argument('--sgd',
                        action="store_true",
                        help='Decide if you want to use SGD over Adam')

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

    parser.add_argument('--sub_window_size',
                        type=int,
                        default=8,
                        help='Number of data points to consider as part of a sentence.'
                        )

    parser.add_argument('--sub_area',
                        type=int,
                        default=0,
                        help='how much sub area to delete from 25*25 grid.'
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
        now_ = '{}_{}_{}_{}_{}'.format(now.month,
                                    now.day,
                                    now.hour,
                                    now.minute,
                                    now.second)

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
    

#def create_models(model_name, 
#                config_name, 
#                **model_kwargs):
#    #TODO: Use when know the different models that we want.
#    models[model_name)


    

