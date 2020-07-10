#========================================================
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

decoder_config = decoder_config = (
    ('gru', 128),
    ('gru', 128)
)

transformer_config = {
    'nhead': 2,
    'num_layers': 4,
    'dropout': 0.1,  # Default
    'dim_feedforward': 2048  # Default
}
