# ContextCountBooster
Sequence context count modeling with weights using xgboost



# Input data

Input data for encode command should be in TSV file format with no column headers or row indices. 
Context counts should be a TSV file with two columns: sequence context and context count. 
Context weights should be a TSV file with two columns: sequence context and context weight (opportunities). 


# Usage

All commands and their descriptions can be found by running: 

```
uv run contextcountbooster --help
```

Usage and all parameters of specific commands can be found by running: 

```
uv run contextcountbooster command --help
```


## Data encoding

Run context encoding for 3-mer contexts and A2C mutation counts with 4-bit encoding.

Training data (training data is split to train and validation set): 
```
uv run contextcountbooster encode --output_dir ../output/3mers/A2C/4bit/ --encoding 4 --ref_base A --pseudocount 1 --train_val_split --val_frac 0.1 ../input_files/train_input_counts.txt ../input_files/train_input_weights.txt
```

Test data: 
```
uv run contextcountbooster encode --output_dir ../output/3mers/A2C/4bit/ --encoding 4 --ref_base A --pseudocount 1 ../input_files/test_input_counts.txt ../input_files/test_input_weights.txt
```

Encoding function outputs a combined TSV file with context count, weight, frequency and the encoding.


## Model training

CCB trains a boosting model using `xgb` package. 

Model training with hyperparameter tuning can be run as: 

```
uv run contextcountbooster train --encoding 4 --outdir ../output/3mers/A2C/4bit/mod/ ../output/3mers/A2C/4bit/train90pct_refA_3mers_4bitOHE.tsv ../output/3mers/A2C/4bit/val10pct_refA_3mers_4bitOHE.tsv
```

Training command outputs the optimal hyperparameter values, training loss and validation loss to `bst_best_param.csv`; trained model to `bst_best.json` (used for loading and making predictions) and `bst_best.raw.txt` (used for model downstream analysis); plots feature gain and weight in `feature_gain.png` and `feature_weight.png`, respectively, and outputs the best result from each tested hyperparameter setting with training and validation loss to `training_res.csv`. 



## Predictions

The trained model can be used to make predictions as: 

```
uv run contextcountbooster predict --outdir ../output/3mers/A2C/4bit/pred/ ../output/3mers/A2C/4bit/test_refA_3mers_4bitOHE.tsv ../output/3mers/A2C/4bit/mod/bst_best.json
```

The predict command outputs the test loss to `test_nll.csv` and test set predictions (along with observed contxt, weight and frequency data) to `test_pred.csv`.