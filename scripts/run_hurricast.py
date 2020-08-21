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
    
    train_loader, val_loader = prepro.create_loaders(
        mode=args.mode,
        data_dir=args.data_dir,
        vision_name=args.vision_name,
        y_name=args.y_name,
        batch_size=args.batch_size,
        train_test_split=args.train_test_split,
        predict_at=args.predict_at,
        window_size=args.window_size, 
        do_save_tensors=False,
        load_tensors=True)

    #========================
    encoder_conf = config.create_config(args.encoder_config)
    decoder_conf = config.create_config(args.decoder_config)
    
    #=======================
    model = model_factory.create_model(
        mode=args.mode, 
        encoder_config=encoder_conf,
        decoder_config=decoder_conf, 
        args=args)
   
    #====================
    train_loss_fn, eval_loss_fn, \
        metrics_fn = metrics.create_metrics_fn(task)

    print("Using model", model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr)

    if args.sgd:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9)
    #=====================================
    best_model, \
        optimizer, \
        training_stats = run.train(
            model, optimizer,
            num_epochs=args.n_epochs,
            train_loss_fn=train_loss_fn,
            test_loss_fn=eval_loss_fn,
            metrics_fn=metrics_fn,
            train_iterator=train_loader,
            val_iterator=val_loader,
            test_iterator=val_loader,
            mode=args.mode,
            task=task,
            get_training_stats=args.get_training_stats,
            clip=None,
            scheduler=None,
            l2_reg=args.l2_reg,
            save=args.save,
            args=args,
            output_dir=args.output_dir,
            writer=args.writer)
    #TODO: Add the saving/data analysis functions
    return None

    #=======================================






if __name__ == "__main__":
    args = setup.create_setup()
    main(args)
