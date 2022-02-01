# Hurricast - Hurricane Forecasting with Deep Learning

This repository holds the code corresponding to the paper:
**Hurricane Forecasting: A Novel Multimodal Machine Learning Framework**
https://arxiv.org/abs/2011.06125
by Léonard Boussioux, Cynthia Zeng, Théo Guénais, Dimitris Bertsimas.

Hurricast is a novel machine learning (ML) framework for tropical cyclone intensity and track forecasting, combining multiple ML techniques and utilizing diverse data sources. Our multimodal framework efficiently combines spatial-temporal data with statistical data by extracting features with deep-learning encoder-decoder architectures and predicting with gradient-boosted trees.

We evaluate our models in the North Atlantic and Eastern Pacific basins on 2016-2019 for 24-hour lead time track and intensity forecasts and show they achieve comparable mean average error and skill to current operational forecast models while computing in seconds.

Furthermore, the inclusion of Hurricast into an operational forecast consensus model could improve over the National Hurricane Center's official forecast, thus highlighting the complementary properties with existing approaches. In summary, our work demonstrates that utilizing machine learning techniques to combine different data sources can lead to new opportunities in tropical cyclone forecasting.

## Hurricast Methodology

The overall multimodal pipeline follows a 3-step mechanism. 
- During Step 1, we extract embeddings from the reanalysis maps using encoder-decoder architectures to obtain a one-dimensional representation. 
- During Step 2, we concatenate the statistical data with the features extracted from the reanalysis maps. 
- During Step 3, we train one XGBoost model for each of the prediction tasks: intensity in 24 h, latitude displacement in 24 h, and longitude displacement in 24 h.

![pipeline.pdf](https://github.com/leobix/hurricast/files/7980070/pipeline.pdf)

## Structure 
- run.py
- utils/
  - utils_vision_data.py
  - data_processing.py	
  - models.py	
  - plot.py	

## Update 
**How to use ?**
- scripts: The python file to run a model and model configs.
  run_hurricast.py and config.py 
- The command line parser is in src/setup.py

The entire code base is wrapped up in src. 
- prepro.py :
  - 1. $\rightarrow$ Class to process the data
  - 2. Add a collate function that allows to batch the data using dictionary. Together with a dataloader\
  the command ```next(iter(loader))``` will output a dictionary. 
  ```py
  Example: mode = intensity; loader = DataLoader(foofoo); in_model, in_loss = next(iter(loader))
  >>> print(in_loss.keys(), in_model.keys())
  "trg_y", "x_viz", "x_stat" 
  ```
  - 3. TODO: Add Reweighted sampling

All the files work upon choosing a "mode". Depending on the 
mode, the variables will be different, as well as the layer of our model.
For instance, for each mode, here is the "target variable" we aim at predicting.
```py
accepted_modes = {#Modes and associated targets
    'intensity': 'tgt_intensity',
    'displacement': 'tgt_displacement',
    'intensity_cat': 'tgt_intensity_cat',
    'baseline_intensity_cat': 'tgt_intensity_cat_baseline',
    'baseline_displacement': 'tgt_displacement_baseline'
    }
```
Each mode is also associated with a task, i.e classification/regression (or potentially new tasks in the future).
```py
modes = {#Modes and associated tasks
    'intensity': 'regression',
    'displacement': 'regression',
    'intensity_cat': 'classification',
    'baseline_intensity_cat': 'classification',
    'baseline_displacement': 'regression'
}
```

The **model creation** is wrapped up in the src/models/factory.py file. 
The different models are in the src/models/hurricast_models.py, and each model should be "registered" before being used (with a simple decorator.)
Example:
```py
@RegisterModel('TRANSFORMER')
class TRANSFORMER(nn.Module):
  ...
```

The baselines and older versions are in a separate file (baselines.py).
The file experimental_models.py is still experimental, but class inheritance can probably allow us to do something cool :sunglass:

____
About the "experimental model"
- **Image Encoder**: Any module that transforms $(BS \times  num\_channels\times H_1\times H_2) \rightarrow (BS, H_{out})$ is accepted as an encoder.
- **Decoder**: Any module that models a sequence $(BS \times Sequence\_length \times H_{in_decoder}) \rightarrow (BS \times H_{out\_decoder})$ is an accepted decoder.
- **Hurricast Model**: Wraps the two models above. 
  - The Encoder can be ommited by initializing the model with a ```None``` encoder (or batching a ```None``` image).
  - Similarly, we can decide not to use one the tabular data by specifying ```no_stats=True```
  - The fusion of the tabular and vision data can eventually be changed.
  - A last activation and linear layer will transform $(BS \times H_{out_decoder}) \rightarrow Activation \rightarrow (BS \times 1)$ 

  


# How to use from the command line 
```bash
python run_hurricast.py\
    --mode=intensity\ #Or intensity_cat/displacement
    --full_encoder\ #Or no_encoder/split_encoder
    --encoder_config=full_encoder_config\ #Name of the config in config.py
    --transformer\ #Whether we use the transformer or a recurrent model
    --decoder_config=transformer_config\ #Name of the config in config.py
    --y_name ...\
    --vision_name ...\
    --predict_at ...\
    --window_size..\
    --train_test_split
    #Can add --no_stat if we want the vision data only
```

## Data:
- The data is very large (>30Gb): we are looking for a solution to host it.
