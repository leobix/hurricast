import argparse
import json
import torch
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter
sys.path.append('../')
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

    parser.add_argument('--gpu_nb', 
                        type=int,
                        help='GPU index. Use -1 for cpu', 
                        default=-1)

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
                            default='../results/results_0')
    
    parser.add_argument('--data_dir', 
                        type=str, 
                        help='Path to the director file',
                        default='../data')

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

    #save_parser = parser.add_mutually_exclusive_group()
    parser.add_argument('--load_tensors', 
        action='store_true', 
        help="Whether to load the tensors from disk.\
            In that case, the tensors must be named under the convention \
            train_tensor_KEY.npy. If False, the tensors will be created." )
    
    parser.add_argument(
        '--save_tensors', 
        action='store_true', 
        help="Whether to save the newly create directly.\
            In that case, the tensors will be saved under\
            data_dir/train_tensor_KEY.npy (numpy format)")
            
    return parser


def add_reformat_parser(parser):
    parser.add_argument(
        '--mode', type=str, default='intensity_cat', 
        help='The mode/task that we want to work on.\
            one of intensity, displacement, etc...')
    
    parser.add_argument(
        '--get_training_stats', action='store_false')

    parser.add_argument(
        '--numpy_seed', type=int, default=2020)
    
    parser.add_argument(
        '--torch_seed', type=int, default=0)

    parser.add_argument(
        '--no_stat', action='store_true', default=False)

    parser.add_argument(
        '--encoder_config', type=str, default='full_encoder_config', 
        help="One of the registered configs")
    
    parser.add_argument(
        '--decoder_config', type=str, default='transformer_config', 
        help="One of the registered configs")


    parser.add_argument(
        '--transformer', action='store_true',
        help="Whether you want to use the Transformer model.\
                        If False a recurrent model will be used.")

    encoder_parser = parser.add_mutually_exclusive_group()
    encoder_parser.add_argument(
        '--full_encoder', action='store_true')
    encoder_parser.add_argument(
        '--split_encoder', action='store_true')
    encoder_parser.add_argument(
        '--no_encoder', action='store_true')
    
    parser.add_argument(
        '--sampler_weights',
        help=' Weights for the sampler. \
            Example of use: --sampler_weights 0.1 0.3 0.5 ...',
        nargs='+',
        type=int,
        default=[])

    
    
    #parser.add_argument(
    #    '--encoder', type=str, default=None, 
    #    help="One of the Models registered in the model factory")


    #parser.add_argument(
    #
    #)

    #parser.add_argument(
    #    '--decoder', type=str, default='ExpLSTM',
    #    help="One of the Models registered in the model factory.\
    #                        Common models: ExpLSTM, TRANSFORMER")
    #parser.add_argument(
    #   '--split_encoder', )
    
    return parser


def create_seeds(torch_seed=0, np_seed=2020):
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)


def create_device(gpu_nb):
    #device = torch.device(
        #f'cuda:{gpu_nb}' if torch.cuda.is_available() and gpu_nb != -1 else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(' Prepare the training using ', device)
    return device


def create_setup():
    """
    Create cl parser and safe creation of the 
    output directory.
    """
    parser = argparse.ArgumentParser()
    parser = add_model_parser(parser)
    parser = add_data_parser(parser)
    parser = add_reformat_parser(parser)

    args = parser.parse_args()
    
    
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

        args.output_dir = '../results/results'+now_
    else:
        pass

    print('Creating {}'.format(args.output_dir))
    os.mkdir(args.output_dir)

    with open(osp.join(args.output_dir, 'args.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    
    create_seeds(
        torch_seed=args.torch_seed, 
        np_seed=args.numpy_seed)
    
    writer = create_board(args)

    device = create_device(gpu_nb=args.gpu_nb)

    args.device = device
    args.writer = writer

    return args


def create_board(args):
    """
    Create board, add model config as some text.
    """
    writer = SummaryWriter(args.output_dir)  # Add tensorboard
    writer.add_text('Args', json.dumps(vars(args), indent=2))
    return writer
    




    

