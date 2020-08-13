

NO_REGISTRY_ERR = "Model {} not in MODEL_REGISTRY! Available models are {}"

#global MODEL_REGISTRY
MODEL_REGISTRY = {}


def RegisterModel(name):
    """Registers a model."""

    def decorator(f):
        MODEL_REGISTRY[name] = f
        return f
    return decorator



#======================
#FOMER FUNCTION
def get_model(mode, encoder_config, decoder_config, args):
    #Needs to upload window size and _OUT_DECODER
    #Get some
    assert (int(args.encdec) + int(args.transformer) < 2), "\
    Only one of encdec or transformer can be specified"
    #print(MODEL_REGISTRY)
    _encoder = MODEL_REGISTRY['CNNEncoder']
    if args.encdec:
        _model = MODEL_REGISTRY['ENCDEC']
    elif args.transformer:
        _model = MODEL_REGISTRY['TRANSFORMER']
    else:
        _model = MODEL_REGISTRY['LINEARTransform']

    N_OUT_DECODER = 7 if mode == 'intensity_cat' else (
        2 - (mode == 'intensity'))  # 7 classes of storms if categorical

    #Encoder
    encoder_config = encoder_config if isinstance(encoder_config, dict)\
        else vars(encoder_config)

    encoder = _encoder(**encoder_config)

    #Decoder: Update the config
    decoder_config = decoder_config if isinstance(decoder_config, dict)\
        else vars(encoder_config)

    if not args.encdec and not args.transformer:
        decoder_config['target_intensity'] = args.target_intensity,
        decoder_config['target_intensity_cat'] = args.target_intensity_cat

    else:
        decoder_config['encoder'] = encoder
        decoder_config['window_size'] = args.window_size
        decoder_config['n_out_decoder'] = N_OUT_DECODER

    model = _model(**decoder_config)

    model = model.to(args.device)

    if args.writer is not None:
        configs = [encoder_config, decoder_config]
        config_ = ""
        for config__ in configs:
            config_ += "{}\n".format(config__)
        args.writer.add_text('Configs', config_)
    return model


def create_model(mode, 
                encoder_config, 
                decoder_config, 
                args):
    #TODO: Change the import of the MODEL_REGISTRY TO MAKE IT CLEANER
    
    from .experimental_models import MODEL_REGISTRY
    
    assert (int(args.encdec) + int(args.transformer) < 2), "\
    Only one of encdec or transformer can be specified"
    
    N_OUT_DECODER = 7 if mode == 'intensity_cat' else (
        2 - (mode == 'intensity'))  # 7 classes of storms if categorical
    #Encoder
    encoder_config = encoder_config if isinstance(encoder_config, dict)\
        else vars(encoder_config)
    #Decoder: Update the config
    decoder_config = decoder_config if isinstance(decoder_config, dict)\
        else vars(encoder_config)
    
    _decoder = "ExpTRANSFORMER" if args.transformer else "ExpLSTM"
    _encoder = None if args.no_encoder else "CNNEncoder"
    print(MODEL_REGISTRY.keys())
    model = MODEL_REGISTRY[
        'ExperimentalHurricast'](
        n_pred=N_OUT_DECODER,
        decoder_config=decoder_config,
        encoder_config=encoder_config,
        decoder_name=_decoder,
        encoder_name=_encoder,
        split_cnns=args.split_encoder,
        no_stat=args.no_stat)

    #model = _model(**decoder_config)

    model = model.to(args.device)

    if args.writer is not None:
        configs = [encoder_config, decoder_config]
        config_ = ""
        for config__ in configs:
            config_ += "{}\n".format(config__)
        args.writer.add_text('Configs', config_)
    return model

#MODEL_REGISTRY = {**MODEL_REGISTRY, .hurricast_models.MODEL_REGISTRY}
