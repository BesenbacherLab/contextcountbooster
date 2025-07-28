![example workflow](https://github.com/carmenoroperv/contextcountbooster/actions/workflows/ruff.yml/badge.svg)



# ContextCountBooster
Sequence context rate modeling with xgboost


# Input data

Input data sets for the encode command should be in TSV file format with no column headers or row indices. 
Context counts should be a TSV file with two columns: sequence context and context count. 
Context weights should be a TSV file with two columns: sequence context and context weight (opportunities). 


# Usage

All commands and their descriptions can be founds by running: 

```
uv run contextcountbooster --help
```

Usage and all parameters of specific commands can be found by running: 

```
uv run contextcountbooster command --help
```


## Data encoding

Run encoding to one-hot encode k-mer sequences and return a data set with the explanatory and target variables for modeling

```
uv run contextcountbooster encode --output_dir ../output/ --output_prefix train_ --encoding 4 ../input/train_counts.txt ../input/train_weights.txt
```

Encoding command outputs a combined TSV file with columns representing the k-mer, count, weight, rate (count/weight) and the sequence encoding.


## Model training

CCB trains a boosting model using `xgboost` Python package. 

Hyperparameter tuning via grid search cross-validation and subsequent model training can be run as: 

```
uv run contextcountbooster train --encoding 4 --output_dir ../output/ --train_data ../output/train_3mers_4bitOHE.tsv
```

The default grids and hyperparameter values can be specified with command line arguments specified in CCB documentation.


Training command outputs the optimal hyperparameter values and mean validation loss to `bst_best_param.csv` and the trained model to `bst_best.json` (used for loading and making predictions). Additionally, figures of feature gains and weights are generated, and cross-validation results with mean validation loss and mean Nagelkerke R2 values are output to `training_res.csv`. Lastly, the null model, represented by the mean rate of the training set, is output to `mu_freq.csv`.


Cross-validation and model training is feasible to run as a one computational task up to 9-mers. Beyond 9-mers, it is recommended to split cross-validation to independent jobs and subsequently aggregate CV results and train the full model. 

To run cross-validation in a parallel manner, testing one combination of hyperparameters can be run as: 

```
uv run contextcountbooster train --encoding 4 --output_dir ../output/CV/ --dist_CV --l2_lambda 100 --eta 0.1 --max_depth 6 --alpha 0.01 --train_data ../output/train_3mers_4bitOHE.tsv 
```
Here, five-fold cross-validation is run with L2 regularization value of 100, learning rate of 0.1, maximum tree depth of 6 and pseudoweight fraction of 0.01. The training command with the dist_CV flag outputs the CV results to a .CSV file named based on the paramter values (L2_100__ETA_0.1__MD_6__ALPHA_0.01.csv).

After running cross-validation in a distributed manned, the full model can be trained as: 

```
uv run contextcountbooster train --aggregate_and_train_only --encoding 4 --output_dir ../output/ --train_data ../output/train_3mers_4bitOHE.tsv
```
The distributed CV results will be automatically collected from the subfolder specified by the --output_dir and suffixed with /CV/. 
The train command outputs the same files as described above. The distributed model training can easily be applied by running the [CCB Snakemake workflow]((https://github.com/BesenbacherLab/ccb_pipeline/tree/master))


## Predictions

The trained model can be used to make predictions as: 

```
uv run contextcountbooster predict --output_dir ../output/ ../input/test_3mers_4bitOHE.tsv ../output/bst_best.json ../output/mu_freq.csv
```
Here, the `test_3mers_4bit.tsv` is the test data encoded with CCB's encode command, `bst_best.json` is the fitted model, and `mu_freq.csv` is the null model output by the train command. 

The predict command outputs the test loss to `test_nll.csv` and test set predictions (along with observed contxt, weight and frequency data) to `test_pred.csv`.