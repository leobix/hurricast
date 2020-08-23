import sys
sys.path.append("../")
from src import prepro, metrics, run, setup
import src.models.factory as model_factory
import config
import torch

modes = {#Modes and associated tasks
    'intensity': 'regression',
    'displacement': 'regression',
    'intensity_cat': 'classification',
    'baseline_intensity_cat': 'classification',
    'baseline_displacement': 'regression'
}

def main(args):
    #TODO: Add the saving/data analysis functions
    task = modes[args.mode]
    print('MODE AND TASK: {} | {}'.format(args.mode, task))
    
    train_loader, val_loader, _ = prepro.create_loaders(
        mode=args.mode,
        data_dir=args.data_dir,
        vision_name=args.vision_name,
        y_name=args.y_name,
        batch_size=args.batch_size,
        train_test_split=args.train_test_split,
        predict_at=args.predict_at,
        window_size=args.window_size, 
        do_save_tensors=args.save_tensors,
        do_load_tensors=args.load_tensors)

    return None

    #=======================================






if __name__ == "__main__":
    args = setup.create_setup()
    main(args)
