import sys
sys.path.append("../")
from src import prepro, models, metrics, run, setup
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
        window_size=args.window_size)

    #========================
    if args.encdec: decoder_config = config.encdec_config
    elif args.transformer: decoder_config = config.transformer_config
    else: decoder_config = config.lineartransform_config
    model = models.factory.get_model(
        mode=args.mode, 
        encoder_config=config.encoder_config, 
        decoder_config=decoder_config, 
        args=args)
    #TODO:Clean up the model creation part.
    #N_IN = 3 * 3
    #N_OUT = 128
    #N_IN_ENCODER = 128 + 10
    #N_OUT_DECODER = 7 if args.target_intensity_cat else (
    #    2 - args.target_intensity)  # 7 classes of storms if categorical
    #N_OUT_TRANSFORMER = 128
    #Encoder
    #encoder_config = config.encoder_config
    #encoder = models.CNNEncoder(
    #                           n_in=N_IN,
    #                           n_out=N_OUT,
    #                           hidden_configuration=encoder_config)
    #Decoder
    #if args.encdec: 
    #    decoder_config = config.decoder_config
    #    model = models.ENCDEC(n_in_decoder=N_IN_ENCODER,
    #                         n_out_decoder=N_OUT_DECODER,
    #                         encoder=encoder,
    #                         hidden_configuration_decoder=decoder_config,
    #                         window_size=args.window_size)
    #elif args.transformer:
    #    decoder_config = config.transformer_config
    #    model = models.TRANSFORMER(encoder,
    #                               n_in_decoder=N_IN_ENCODER,
    #                               n_out_decoder=N_OUT_DECODER,
    #                              n_out_transformer=N_OUT_TRANSFORMER,
    #                               hidden_configuration_decoder=decoder_config,
    #                               window_size=args.window_size)
    #else:
    #    model = models.LINEARTransform(encoder, args.window_size, target_intensity=args.target_intensity,
    #                                   target_intensity_cat=args.target_intensity_cat)
    #    decoder_config = None
    
    #writer = setup.create_board(
    #    args, model, configs=[encoder_config, decoder_config])

    #model = model.to(args.device)

    #configs = [encoder_config, decoder_config]
    #config_ = ""
    #for config__ in configs:
    #    config_ += "{}\n".format(config__)
    #args.writer.add_text('Configs', config_)
    #===================================
    train_loss_fn, eval_loss_fn, metrics_fn = metrics.create_metrics_fn(task)

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
