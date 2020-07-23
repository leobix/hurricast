# Hurricast - Development branch :hammer:

This is the repository for Hurricast Project.  

## Data:
- https://drive.google.com/file/d/1viv4Li31JF6LVCt45i0lq8c9qZGhPlVP/view?usp=sharing
- Move it to data_era.
- Be sure it is named: geopotential_u_component_of_wind_v_component_of_wind

## Structure 
- **(NEW)** run.py
- utils/
  - utils_vision_data.py
  - data_processing.py	
  - **(NEW)** models.py	
  - **(NEW)** plot.py	

## Update 
**How to use ?**
- scripts: The python file to run a model and model configs.
  run_hurricast.py and config.py 
- The command line parser is still the same (for now). Details in src/setup.py

The entire code base is wrapped up in src. 
- prepro.py :
  - 1. $\rightarrow$ Class to process the data: unchanged, except from 
using named dictionaries as the output. 
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


