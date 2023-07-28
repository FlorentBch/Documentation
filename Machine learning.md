# Helthcare prediction

## Installation

```bash
git clone https://github.com/AnderGarro/AutoMl_Demo
cd ./AutoMl_Demo/Docker/
docker-compose up
```


But : à partir de données existantes, predire le risque d'accident vasculaire cerebrale.
La source des données vien de : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset.

Attribute Information
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient


```python
# Import de librairies
import pandas as pd

#Une librairie pour segmenter les données
from sklearn.model_selection import train_test_split

# Prise en charge des modeles d'entrainement et models prédictifs
from autogluon.tabular import TabularDataset, TabularPredictor
```

## Chargement des données


```python
df = pd.read_csv('/data/healthcare-dataset-stroke-data.csv')
print(df.head())
print(df.count())
```

          id  gender   age  hypertension  heart_disease ever_married  \
    0   9046    Male  67.0             0              1          Yes   
    1  51676  Female  61.0             0              0          Yes   
    2  31112    Male  80.0             0              1          Yes   
    3  60182  Female  49.0             0              0          Yes   
    4   1665  Female  79.0             1              0          Yes   
    
           work_type Residence_type  avg_glucose_level   bmi   smoking_status  \
    0        Private          Urban             228.69  36.6  formerly smoked   
    1  Self-employed          Rural             202.21   NaN     never smoked   
    2        Private          Rural             105.92  32.5     never smoked   
    3        Private          Urban             171.23  34.4           smokes   
    4  Self-employed          Rural             174.12  24.0     never smoked   
    
       stroke  
    0       1  
    1       1  
    2       1  
    3       1  
    4       1  
    id                   5110
    gender               5110
    age                  5110
    hypertension         5110
    heart_disease        5110
    ever_married         5110
    work_type            5110
    Residence_type       5110
    avg_glucose_level    5110
    bmi                  4909
    smoking_status       5110
    stroke               5110
    dtype: int64


## Split dataset

Separation du jeu de données en 2 parties :

- jeu de données d'entrainement
- jeu de données test


```python
df_train, df_test = train_test_split(df, test_size=0.33, random_state=1)
print("df_train shape : ", df_train.shape)
print("df_test shape : ", df_test.shape)
```

    df_train shape :  (3423, 12)
    df_test shape :  (1687, 12)


## Modification du jeu test

Il faut supprimer la colonne "stroke" du jeu de données test


```python
test_data = df_test.drop(['stroke'], axis=1)
print(test_data)
```

             id  gender   age  hypertension  heart_disease ever_married  \
    4673  49833  Female  42.0             0              0          Yes   
    3232  20375  Female  78.0             0              0          Yes   
    3694  39834    Male  28.0             0              0           No   
    1070  42550  Female  81.0             0              0          Yes   
    4163  19907  Female  52.0             0              0          Yes   
    ...     ...     ...   ...           ...            ...          ...   
    386   63732    Male  70.0             1              0          Yes   
    3961   4655    Male  49.0             0              0          Yes   
    1608   9011    Male  59.0             0              0          Yes   
    1459  11726  Female  49.0             0              0          Yes   
    4058  35772    Male  17.0             0              0           No   
    
              work_type Residence_type  avg_glucose_level   bmi   smoking_status  
    4673       Govt_job          Rural             112.98  37.2  formerly smoked  
    3232        Private          Urban              78.29  30.1  formerly smoked  
    3694        Private          Urban              73.27  25.4           smokes  
    1070  Self-employed          Rural             246.34  21.1     never smoked  
    4163        Private          Rural              97.05  28.0          Unknown  
    ...             ...            ...                ...   ...              ...  
    386   Self-employed          Urban             251.60  27.1     never smoked  
    3961        Private          Urban              79.51  37.8     never smoked  
    1608        Private          Urban              93.58  25.1           smokes  
    1459       Govt_job          Rural              83.84  19.3  formerly smoked  
    4058        Private          Urban              71.58  25.6          Unknown  
    
    [1687 rows x 11 columns]


## Prediction

Il faut construire une prediction pour entrainer à classifier si un individu avec ces carracteristiques est à risque de faire un accident vasculaire cerebrale. 

Pour cela, on utilise la methode **TabularPredictor** en spécifiant que la sortie doit etre une colonne 'stroke'.
La methode va choisir automatiquement l'algorithme le mieux adapté pour entrainer le jeu de donnée.

Les arguments (optionnels):

- 'verbosity=2' afficheront toutes les étapes que la prediction prend pour arriver au meilleur modèle
- 'presets= best quality' garantiront que le meilleur modèle est sélectionné parmi ceux formés.

Il existe d'autres arguments supplémentaires mentionnés dans la documentation officielle qui peuvent être utilisés pour affiner le modèle.


```python
predictor = TabularPredictor(label="stroke").fit(train_data = df_train, verbosity=2, presets="best_quality")
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20230621_093607/"
    Presets specified: ['best_quality']
    Beginning AutoGluon training ...
    AutoGluon will save models to "AutogluonModels/ag-20230621_093607/"
    AutoGluon Version:  0.3.1
    Train Data Rows:    3423
    Train Data Columns: 11
    Preprocessing data ...
    AutoGluon infers your prediction problem is: 'binary' (because only two unique label-values observed).
    	2 unique label values:  [0, 1]
    	If 'binary' is not the correct problem_type, please manually specify the problem_type argument in fit() (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
    Selected class <--> label mapping:  class 1 = 1, class 0 = 0
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    23134.9 MB
    	Train Data (Original)  Memory Usage: 1.25 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 5 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('float', [])  : 3 | ['age', 'avg_glucose_level', 'bmi']
    		('int', [])    : 3 | ['id', 'hypertension', 'heart_disease']
    		('object', []) : 5 | ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])  : 2 | ['work_type', 'smoking_status']
    		('float', [])     : 3 | ['age', 'avg_glucose_level', 'bmi']
    		('int', [])       : 1 | ['id']
    		('int', ['bool']) : 5 | ['gender', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type']
    	0.1s = Fit runtime
    	11 features in original data used to generate 11 features in processed data.
    	Train Data (Processed) Memory Usage: 0.13 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.18s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'accuracy'
    	To change this, specify the eval_metric argument of fit()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 13 L1 models ...
    Fitting model: KNeighborsUnif_BAG_L1 ...
    	0.9533	 = Validation score   (accuracy)
    	0.01s	 = Training   runtime
    	0.11s	 = Validation runtime
    Fitting model: KNeighborsDist_BAG_L1 ...
    	0.9518	 = Validation score   (accuracy)
    	0.01s	 = Training   runtime
    	0.1s	 = Validation runtime
    Fitting model: LightGBMXT_BAG_L1 ...
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    	0.9582	 = Validation score   (accuracy)
    	4.42s	 = Training   runtime
    	0.09s	 = Validation runtime
    Fitting model: LightGBM_BAG_L1 ...
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    	0.9573	 = Validation score   (accuracy)
    	5.58s	 = Training   runtime
    	0.09s	 = Validation runtime
    Fitting model: RandomForestGini_BAG_L1 ...
    	0.9559	 = Validation score   (accuracy)
    	1.51s	 = Training   runtime
    	0.21s	 = Validation runtime
    Fitting model: RandomForestEntr_BAG_L1 ...
    	0.9565	 = Validation score   (accuracy)
    	1.49s	 = Training   runtime
    	0.2s	 = Validation runtime
    Fitting model: CatBoost_BAG_L1 ...
    	0.9565	 = Validation score   (accuracy)
    	6.87s	 = Training   runtime
    	0.06s	 = Validation runtime
    Fitting model: ExtraTreesGini_BAG_L1 ...
    	0.955	 = Validation score   (accuracy)
    	1.49s	 = Training   runtime
    	0.2s	 = Validation runtime
    Fitting model: ExtraTreesEntr_BAG_L1 ...
    	0.9544	 = Validation score   (accuracy)
    	1.48s	 = Training   runtime
    	0.2s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L1 ...
    	Warning: Exception caused NeuralNetFastAI_BAG_L1 to fail during training... Skipping this model.
    		future feature annotations is not defined (dispatch.py, line 4)
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.6/dist-packages/autogluon/tabular/trainer/abstract_trainer.py", line 962, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, **model_fit_kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/tabular/trainer/abstract_trainer.py", line 934, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, **model_fit_kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/abstract/abstract_model.py", line 522, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 153, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 189, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 388, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 59, in after_all_folds_scheduled
        self._fit_fold_model(*job)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 64, in _fit_fold_model
        fold_model = self._fit(model_base, time_start_fold, time_limit_fold, fold_ctx, kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 111, in _fit
        fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, **kwargs_fold)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/abstract/abstract_model.py", line 522, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 163, in _fit
        try_import_fastai()
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/utils/try_import.py", line 107, in try_import_fastai
        import autogluon.tabular.models.fastainn.imports_helper
      File "/usr/local/lib/python3.6/dist-packages/autogluon/tabular/models/fastainn/imports_helper.py", line 1, in <module>
        from fastai.tabular.all import *
      File "/usr/local/lib/python3.6/dist-packages/fastai/tabular/all.py", line 1, in <module>
        from ..basics import *
      File "/usr/local/lib/python3.6/dist-packages/fastai/basics.py", line 1, in <module>
        from .data.all import *
      File "/usr/local/lib/python3.6/dist-packages/fastai/data/all.py", line 1, in <module>
        from ..torch_basics import *
      File "/usr/local/lib/python3.6/dist-packages/fastai/torch_basics.py", line 9, in <module>
        from .imports import *
      File "/usr/local/lib/python3.6/dist-packages/fastai/imports.py", line 30, in <module>
        from fastcore.all import *
      File "/usr/local/lib/python3.6/dist-packages/fastcore/all.py", line 3, in <module>
        from .dispatch import *
      File "/usr/local/lib/python3.6/dist-packages/fastcore/dispatch.py", line 4
        from __future__ import annotations
                                         ^
    SyntaxError: future feature annotations is not defined
    Fitting model: XGBoost_BAG_L1 ...
    	0.9582	 = Validation score   (accuracy)
    	2.66s	 = Training   runtime
    	0.1s	 = Validation runtime
    Fitting model: NeuralNetMXNet_BAG_L1 ...
    	0.9562	 = Validation score   (accuracy)
    	88.65s	 = Training   runtime
    	0.52s	 = Validation runtime
    Fitting model: LightGBMLarge_BAG_L1 ...
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    	0.9576	 = Validation score   (accuracy)
    	13.14s	 = Training   runtime
    	0.11s	 = Validation runtime
    Fitting model: WeightedEnsemble_L2 ...
    	0.9582	 = Validation score   (accuracy)
    	2.57s	 = Training   runtime
    	0.01s	 = Validation runtime
    Fitting 11 L2 models ...
    Fitting model: LightGBMXT_BAG_L2 ...
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    	0.9573	 = Validation score   (accuracy)
    	4.4s	 = Training   runtime
    	0.09s	 = Validation runtime
    Fitting model: LightGBM_BAG_L2 ...
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    	0.9606	 = Validation score   (accuracy)
    	5.13s	 = Training   runtime
    	0.09s	 = Validation runtime
    Fitting model: RandomForestGini_BAG_L2 ...
    	0.9582	 = Validation score   (accuracy)
    	1.6s	 = Training   runtime
    	0.22s	 = Validation runtime
    Fitting model: RandomForestEntr_BAG_L2 ...
    	0.9576	 = Validation score   (accuracy)
    	1.5s	 = Training   runtime
    	0.22s	 = Validation runtime
    Fitting model: CatBoost_BAG_L2 ...
    	0.9597	 = Validation score   (accuracy)
    	7.88s	 = Training   runtime
    	0.06s	 = Validation runtime
    Fitting model: ExtraTreesGini_BAG_L2 ...
    	0.9556	 = Validation score   (accuracy)
    	1.5s	 = Training   runtime
    	0.22s	 = Validation runtime
    Fitting model: ExtraTreesEntr_BAG_L2 ...
    	0.9576	 = Validation score   (accuracy)
    	1.49s	 = Training   runtime
    	0.21s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L2 ...
    	Warning: Exception caused NeuralNetFastAI_BAG_L2 to fail during training... Skipping this model.
    		future feature annotations is not defined (dispatch.py, line 4)
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.6/dist-packages/autogluon/tabular/trainer/abstract_trainer.py", line 962, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, **model_fit_kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/tabular/trainer/abstract_trainer.py", line 934, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, **model_fit_kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/abstract/abstract_model.py", line 522, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 153, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 189, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 388, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 59, in after_all_folds_scheduled
        self._fit_fold_model(*job)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 64, in _fit_fold_model
        fold_model = self._fit(model_base, time_start_fold, time_limit_fold, fold_ctx, kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 111, in _fit
        fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, **kwargs_fold)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/models/abstract/abstract_model.py", line 522, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.6/dist-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 163, in _fit
        try_import_fastai()
      File "/usr/local/lib/python3.6/dist-packages/autogluon/core/utils/try_import.py", line 107, in try_import_fastai
        import autogluon.tabular.models.fastainn.imports_helper
      File "/usr/local/lib/python3.6/dist-packages/autogluon/tabular/models/fastainn/imports_helper.py", line 1, in <module>
        from fastai.tabular.all import *
      File "/usr/local/lib/python3.6/dist-packages/fastai/tabular/all.py", line 1, in <module>
        from ..basics import *
      File "/usr/local/lib/python3.6/dist-packages/fastai/basics.py", line 1, in <module>
        from .data.all import *
      File "/usr/local/lib/python3.6/dist-packages/fastai/data/all.py", line 1, in <module>
        from ..torch_basics import *
      File "/usr/local/lib/python3.6/dist-packages/fastai/torch_basics.py", line 9, in <module>
        from .imports import *
      File "/usr/local/lib/python3.6/dist-packages/fastai/imports.py", line 30, in <module>
        from fastcore.all import *
      File "/usr/local/lib/python3.6/dist-packages/fastcore/all.py", line 3, in <module>
        from .dispatch import *
      File "/usr/local/lib/python3.6/dist-packages/fastcore/dispatch.py", line 4
        from __future__ import annotations
                                         ^
    SyntaxError: future feature annotations is not defined
    Fitting model: XGBoost_BAG_L2 ...
    	0.9597	 = Validation score   (accuracy)
    	20.16s	 = Training   runtime
    	0.1s	 = Validation runtime
    Fitting model: NeuralNetMXNet_BAG_L2 ...
    	0.9588	 = Validation score   (accuracy)
    	328.95s	 = Training   runtime
    	1.67s	 = Validation runtime
    Fitting model: LightGBMLarge_BAG_L2 ...
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:239: UserWarning: 'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
    	0.9582	 = Validation score   (accuracy)
    	12.59s	 = Training   runtime
    	0.09s	 = Validation runtime
    Fitting model: WeightedEnsemble_L3 ...
    	0.9606	 = Validation score   (accuracy)
    	2.08s	 = Training   runtime
    	0.01s	 = Validation runtime
    AutoGluon training complete, total runtime = 528.36s ...
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20230621_093607/")


L'analyse peut etre assez longue.

Au cours du défilement des logs, on peut constater que **AutoGluon** est capable de choisir une classification binaire automatiquement sans qu'il y ai eu le besoin de lui renseigner.
Le resultat est donc classé en label unique "0" ou "1" dans la colonne de sortie.

On note egalement qu'il choisi de selectionner le model le mieux adapté sur la précision ("accuracy").
Une fois l'entrainemeent terminé, il est posssible de voir un résumé de tous les modeles et leur précision avec la commande **predictor.fit_summary()**


```python
predictor.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                          model  score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0           LightGBM_BAG_L2   0.960561       2.063482  132.439821                0.089859           5.127946            2       True         15
    1       WeightedEnsemble_L3   0.960561       2.074748  134.519620                0.011266           2.079799            3       True         24
    2           CatBoost_BAG_L2   0.959684       2.038080  135.190438                0.064458           7.878563            2       True         18
    3            XGBoost_BAG_L2   0.959684       2.073196  147.476847                0.099573          20.164972            2       True         21
    4     NeuralNetMXNet_BAG_L2   0.958808       3.639860  456.264859                1.666238         328.952984            2       True         22
    5         LightGBMXT_BAG_L1   0.958224       0.090365    4.423393                0.090365           4.423393            1       True          3
    6            XGBoost_BAG_L1   0.958224       0.095039    2.660855                0.095039           2.660855            1       True         10
    7       WeightedEnsemble_L2   0.958224       0.102480    6.991363                0.012115           2.567971            2       True         13
    8      LightGBMLarge_BAG_L2   0.958224       2.066210  139.900362                0.092587          12.588487            2       True         23
    9   RandomForestGini_BAG_L2   0.958224       2.191179  128.912157                0.217556           1.600282            2       True         16
    10     LightGBMLarge_BAG_L1   0.957639       0.105921   13.135603                0.105921          13.135603            1       True         12
    11    ExtraTreesEntr_BAG_L2   0.957639       2.179946  128.802242                0.206324           1.490367            2       True         20
    12  RandomForestEntr_BAG_L2   0.957639       2.189527  128.810289                0.215904           1.498414            2       True         17
    13          LightGBM_BAG_L1   0.957347       0.087040    5.580223                0.087040           5.580223            1       True          4
    14        LightGBMXT_BAG_L2   0.957347       2.063100  131.712806                0.089477           4.400931            2       True         14
    15          CatBoost_BAG_L1   0.956471       0.055576    6.871174                0.055576           6.871174            1       True          7
    16  RandomForestEntr_BAG_L1   0.956471       0.199324    1.490844                0.199324           1.490844            1       True          6
    17    NeuralNetMXNet_BAG_L1   0.956179       0.515805   88.646938                0.515805          88.646938            1       True         11
    18  RandomForestGini_BAG_L1   0.955887       0.206950    1.510387                0.206950           1.510387            1       True          5
    19    ExtraTreesGini_BAG_L2   0.955595       2.193562  128.810273                0.219939           1.498398            2       True         19
    20    ExtraTreesGini_BAG_L1   0.955010       0.201937    1.488284                0.201937           1.488284            1       True          8
    21    ExtraTreesEntr_BAG_L1   0.954426       0.203993    1.483316                0.203993           1.483316            1       True          9
    22    KNeighborsUnif_BAG_L1   0.953257       0.106717    0.012008                0.106717           0.012008            1       True          1
    23    KNeighborsDist_BAG_L1   0.951797       0.104957    0.008851                0.104957           0.008851            1       True          2
    Number of models trained: 24
    Types of models trained:
    {'StackerEnsembleModel_RF', 'StackerEnsembleModel_LGB', 'StackerEnsembleModel_XT', 'WeightedEnsembleModel', 'StackerEnsembleModel_KNN', 'StackerEnsembleModel_TabularNeuralNet', 'StackerEnsembleModel_CatBoost', 'StackerEnsembleModel_XGBoost'}
    Bagging used: True  (with 10 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])  : 2 | ['work_type', 'smoking_status']
    ('float', [])     : 3 | ['age', 'avg_glucose_level', 'bmi']
    ('int', [])       : 1 | ['id']
    ('int', ['bool']) : 5 | ['gender', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type']
    *** End of fit() summary ***


    /usr/local/lib/python3.6/dist-packages/autogluon/core/utils/plots.py:138: UserWarning: AutoGluon summary plots cannot be created because bokeh is not installed. To see plots, please do: "pip install bokeh==2.0.1"
      warnings.warn('AutoGluon summary plots cannot be created because bokeh is not installed. To see plots, please do: "pip install bokeh==2.0.1"')





    {'model_types': {'KNeighborsUnif_BAG_L1': 'StackerEnsembleModel_KNN',
      'KNeighborsDist_BAG_L1': 'StackerEnsembleModel_KNN',
      'LightGBMXT_BAG_L1': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'RandomForestGini_BAG_L1': 'StackerEnsembleModel_RF',
      'RandomForestEntr_BAG_L1': 'StackerEnsembleModel_RF',
      'CatBoost_BAG_L1': 'StackerEnsembleModel_CatBoost',
      'ExtraTreesGini_BAG_L1': 'StackerEnsembleModel_XT',
      'ExtraTreesEntr_BAG_L1': 'StackerEnsembleModel_XT',
      'XGBoost_BAG_L1': 'StackerEnsembleModel_XGBoost',
      'NeuralNetMXNet_BAG_L1': 'StackerEnsembleModel_TabularNeuralNet',
      'LightGBMLarge_BAG_L1': 'StackerEnsembleModel_LGB',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBMXT_BAG_L2': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'RandomForestGini_BAG_L2': 'StackerEnsembleModel_RF',
      'RandomForestEntr_BAG_L2': 'StackerEnsembleModel_RF',
      'CatBoost_BAG_L2': 'StackerEnsembleModel_CatBoost',
      'ExtraTreesGini_BAG_L2': 'StackerEnsembleModel_XT',
      'ExtraTreesEntr_BAG_L2': 'StackerEnsembleModel_XT',
      'XGBoost_BAG_L2': 'StackerEnsembleModel_XGBoost',
      'NeuralNetMXNet_BAG_L2': 'StackerEnsembleModel_TabularNeuralNet',
      'LightGBMLarge_BAG_L2': 'StackerEnsembleModel_LGB',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'KNeighborsUnif_BAG_L1': 0.95325737657026,
      'KNeighborsDist_BAG_L1': 0.9517966695880806,
      'LightGBMXT_BAG_L1': 0.9582237803096699,
      'LightGBM_BAG_L1': 0.9573473561203623,
      'RandomForestGini_BAG_L1': 0.9558866491381829,
      'RandomForestEntr_BAG_L1': 0.9564709319310546,
      'CatBoost_BAG_L1': 0.9564709319310546,
      'ExtraTreesGini_BAG_L1': 0.9550102249488752,
      'ExtraTreesEntr_BAG_L1': 0.9544259421560035,
      'XGBoost_BAG_L1': 0.9582237803096699,
      'NeuralNetMXNet_BAG_L1': 0.9561787905346187,
      'LightGBMLarge_BAG_L1': 0.9576394975167981,
      'WeightedEnsemble_L2': 0.9582237803096699,
      'LightGBMXT_BAG_L2': 0.9573473561203623,
      'LightGBM_BAG_L2': 0.9605609114811569,
      'RandomForestGini_BAG_L2': 0.9582237803096699,
      'RandomForestEntr_BAG_L2': 0.9576394975167981,
      'CatBoost_BAG_L2': 0.9596844872918493,
      'ExtraTreesGini_BAG_L2': 0.955594507741747,
      'ExtraTreesEntr_BAG_L2': 0.9576394975167981,
      'XGBoost_BAG_L2': 0.9596844872918493,
      'NeuralNetMXNet_BAG_L2': 0.9588080631025416,
      'LightGBMLarge_BAG_L2': 0.9582237803096699,
      'WeightedEnsemble_L3': 0.9605609114811569},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'KNeighborsUnif_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/KNeighborsUnif_BAG_L1/',
      'KNeighborsDist_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/KNeighborsDist_BAG_L1/',
      'LightGBMXT_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/LightGBMXT_BAG_L1/',
      'LightGBM_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/LightGBM_BAG_L1/',
      'RandomForestGini_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/RandomForestGini_BAG_L1/',
      'RandomForestEntr_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/RandomForestEntr_BAG_L1/',
      'CatBoost_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/CatBoost_BAG_L1/',
      'ExtraTreesGini_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/ExtraTreesGini_BAG_L1/',
      'ExtraTreesEntr_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/ExtraTreesEntr_BAG_L1/',
      'XGBoost_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/XGBoost_BAG_L1/',
      'NeuralNetMXNet_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/NeuralNetMXNet_BAG_L1/',
      'LightGBMLarge_BAG_L1': 'AutogluonModels/ag-20230621_093607/models/LightGBMLarge_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230621_093607/models/WeightedEnsemble_L2/',
      'LightGBMXT_BAG_L2': 'AutogluonModels/ag-20230621_093607/models/LightGBMXT_BAG_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230621_093607/models/LightGBM_BAG_L2/',
      'RandomForestGini_BAG_L2': 'AutogluonModels/ag-20230621_093607/models/RandomForestGini_BAG_L2/',
      'RandomForestEntr_BAG_L2': 'AutogluonModels/ag-20230621_093607/models/RandomForestEntr_BAG_L2/',
      'CatBoost_BAG_L2': 'AutogluonModels/ag-20230621_093607/models/CatBoost_BAG_L2/',
      'ExtraTreesGini_BAG_L2': 'AutogluonModels/ag-20230621_093607/models/ExtraTreesGini_BAG_L2/',
      'ExtraTreesEntr_BAG_L2': 'AutogluonModels/ag-20230621_093607/models/ExtraTreesEntr_BAG_L2/',
      'XGBoost_BAG_L2': 'AutogluonModels/ag-20230621_093607/models/XGBoost_BAG_L2/',
      'NeuralNetMXNet_BAG_L2': 'AutogluonModels/ag-20230621_093607/models/NeuralNetMXNet_BAG_L2/',
      'LightGBMLarge_BAG_L2': 'AutogluonModels/ag-20230621_093607/models/LightGBMLarge_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230621_093607/models/WeightedEnsemble_L3/'},
     'model_fit_times': {'KNeighborsUnif_BAG_L1': 0.012008190155029297,
      'KNeighborsDist_BAG_L1': 0.008850574493408203,
      'LightGBMXT_BAG_L1': 4.4233925342559814,
      'LightGBM_BAG_L1': 5.580223083496094,
      'RandomForestGini_BAG_L1': 1.5103874206542969,
      'RandomForestEntr_BAG_L1': 1.4908440113067627,
      'CatBoost_BAG_L1': 6.871173620223999,
      'ExtraTreesGini_BAG_L1': 1.4882838726043701,
      'ExtraTreesEntr_BAG_L1': 1.4833157062530518,
      'XGBoost_BAG_L1': 2.6608550548553467,
      'NeuralNetMXNet_BAG_L1': 88.64693760871887,
      'LightGBMLarge_BAG_L1': 13.135603189468384,
      'WeightedEnsemble_L2': 2.5679705142974854,
      'LightGBMXT_BAG_L2': 4.400930643081665,
      'LightGBM_BAG_L2': 5.1279456615448,
      'RandomForestGini_BAG_L2': 1.6002821922302246,
      'RandomForestEntr_BAG_L2': 1.4984140396118164,
      'CatBoost_BAG_L2': 7.878563404083252,
      'ExtraTreesGini_BAG_L2': 1.4983980655670166,
      'ExtraTreesEntr_BAG_L2': 1.4903669357299805,
      'XGBoost_BAG_L2': 20.16497230529785,
      'NeuralNetMXNet_BAG_L2': 328.95298409461975,
      'LightGBMLarge_BAG_L2': 12.588486671447754,
      'WeightedEnsemble_L3': 2.079799175262451},
     'model_pred_times': {'KNeighborsUnif_BAG_L1': 0.10671687126159668,
      'KNeighborsDist_BAG_L1': 0.10495686531066895,
      'LightGBMXT_BAG_L1': 0.09036517143249512,
      'LightGBM_BAG_L1': 0.08703970909118652,
      'RandomForestGini_BAG_L1': 0.20694971084594727,
      'RandomForestEntr_BAG_L1': 0.1993236541748047,
      'CatBoost_BAG_L1': 0.05557608604431152,
      'ExtraTreesGini_BAG_L1': 0.20193696022033691,
      'ExtraTreesEntr_BAG_L1': 0.2039930820465088,
      'XGBoost_BAG_L1': 0.09503865242004395,
      'NeuralNetMXNet_BAG_L1': 0.5158050060272217,
      'LightGBMLarge_BAG_L1': 0.10592079162597656,
      'WeightedEnsemble_L2': 0.012114763259887695,
      'LightGBMXT_BAG_L2': 0.0894770622253418,
      'LightGBM_BAG_L2': 0.0898592472076416,
      'RandomForestGini_BAG_L2': 0.21755647659301758,
      'RandomForestEntr_BAG_L2': 0.21590423583984375,
      'CatBoost_BAG_L2': 0.06445789337158203,
      'ExtraTreesGini_BAG_L2': 0.2199392318725586,
      'ExtraTreesEntr_BAG_L2': 0.20632386207580566,
      'XGBoost_BAG_L2': 0.09957337379455566,
      'NeuralNetMXNet_BAG_L2': 1.6662378311157227,
      'LightGBMLarge_BAG_L2': 0.09258699417114258,
      'WeightedEnsemble_L3': 0.011265993118286133},
     'num_bag_folds': 10,
     'max_stack_level': 3,
     'num_classes': 2,
     'model_hyperparams': {'KNeighborsUnif_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'KNeighborsDist_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'LightGBMXT_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'RandomForestGini_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'RandomForestEntr_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'CatBoost_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'ExtraTreesGini_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'ExtraTreesEntr_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'XGBoost_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetMXNet_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBMLarge_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBMXT_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'RandomForestGini_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'RandomForestEntr_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'CatBoost_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'ExtraTreesGini_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'ExtraTreesEntr_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'XGBoost_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetMXNet_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBMLarge_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                       model  score_val  pred_time_val    fit_time  \
     0           LightGBM_BAG_L2   0.960561       2.063482  132.439821   
     1       WeightedEnsemble_L3   0.960561       2.074748  134.519620   
     2           CatBoost_BAG_L2   0.959684       2.038080  135.190438   
     3            XGBoost_BAG_L2   0.959684       2.073196  147.476847   
     4     NeuralNetMXNet_BAG_L2   0.958808       3.639860  456.264859   
     5         LightGBMXT_BAG_L1   0.958224       0.090365    4.423393   
     6            XGBoost_BAG_L1   0.958224       0.095039    2.660855   
     7       WeightedEnsemble_L2   0.958224       0.102480    6.991363   
     8      LightGBMLarge_BAG_L2   0.958224       2.066210  139.900362   
     9   RandomForestGini_BAG_L2   0.958224       2.191179  128.912157   
     10     LightGBMLarge_BAG_L1   0.957639       0.105921   13.135603   
     11    ExtraTreesEntr_BAG_L2   0.957639       2.179946  128.802242   
     12  RandomForestEntr_BAG_L2   0.957639       2.189527  128.810289   
     13          LightGBM_BAG_L1   0.957347       0.087040    5.580223   
     14        LightGBMXT_BAG_L2   0.957347       2.063100  131.712806   
     15          CatBoost_BAG_L1   0.956471       0.055576    6.871174   
     16  RandomForestEntr_BAG_L1   0.956471       0.199324    1.490844   
     17    NeuralNetMXNet_BAG_L1   0.956179       0.515805   88.646938   
     18  RandomForestGini_BAG_L1   0.955887       0.206950    1.510387   
     19    ExtraTreesGini_BAG_L2   0.955595       2.193562  128.810273   
     20    ExtraTreesGini_BAG_L1   0.955010       0.201937    1.488284   
     21    ExtraTreesEntr_BAG_L1   0.954426       0.203993    1.483316   
     22    KNeighborsUnif_BAG_L1   0.953257       0.106717    0.012008   
     23    KNeighborsDist_BAG_L1   0.951797       0.104957    0.008851   
     
         pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                 0.089859           5.127946            2       True   
     1                 0.011266           2.079799            3       True   
     2                 0.064458           7.878563            2       True   
     3                 0.099573          20.164972            2       True   
     4                 1.666238         328.952984            2       True   
     5                 0.090365           4.423393            1       True   
     6                 0.095039           2.660855            1       True   
     7                 0.012115           2.567971            2       True   
     8                 0.092587          12.588487            2       True   
     9                 0.217556           1.600282            2       True   
     10                0.105921          13.135603            1       True   
     11                0.206324           1.490367            2       True   
     12                0.215904           1.498414            2       True   
     13                0.087040           5.580223            1       True   
     14                0.089477           4.400931            2       True   
     15                0.055576           6.871174            1       True   
     16                0.199324           1.490844            1       True   
     17                0.515805          88.646938            1       True   
     18                0.206950           1.510387            1       True   
     19                0.219939           1.498398            2       True   
     20                0.201937           1.488284            1       True   
     21                0.203993           1.483316            1       True   
     22                0.106717           0.012008            1       True   
     23                0.104957           0.008851            1       True   
     
         fit_order  
     0          15  
     1          24  
     2          18  
     3          21  
     4          22  
     5           3  
     6          10  
     7          13  
     8          23  
     9          16  
     10         12  
     11         20  
     12         17  
     13          4  
     14         14  
     15          7  
     16          6  
     17         11  
     18          5  
     19         19  
     20          8  
     21          9  
     22          1  
     23          2  }



**Remarque :** attention, les modeles ne sont pas dans l'ordre de leur création! 
pour cela il faut utiliser la methode : **predictor.leaderboard(df_train, silent=True)**


```python
predictor.leaderboard(df_train, silent=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score_test</th>
      <th>score_val</th>
      <th>pred_time_test</th>
      <th>pred_time_val</th>
      <th>fit_time</th>
      <th>pred_time_test_marginal</th>
      <th>pred_time_val_marginal</th>
      <th>fit_time_marginal</th>
      <th>stack_level</th>
      <th>can_infer</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KNeighborsDist_BAG_L1</td>
      <td>1.000000</td>
      <td>0.951797</td>
      <td>0.122752</td>
      <td>0.104957</td>
      <td>0.008851</td>
      <td>0.122752</td>
      <td>0.104957</td>
      <td>0.008851</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestEntr_BAG_L1</td>
      <td>1.000000</td>
      <td>0.956471</td>
      <td>0.354431</td>
      <td>0.199324</td>
      <td>1.490844</td>
      <td>0.354431</td>
      <td>0.199324</td>
      <td>1.490844</td>
      <td>1</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RandomForestGini_BAG_L1</td>
      <td>1.000000</td>
      <td>0.955887</td>
      <td>0.370783</td>
      <td>0.206950</td>
      <td>1.510387</td>
      <td>0.370783</td>
      <td>0.206950</td>
      <td>1.510387</td>
      <td>1</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ExtraTreesGini_BAG_L1</td>
      <td>1.000000</td>
      <td>0.955010</td>
      <td>0.463382</td>
      <td>0.201937</td>
      <td>1.488284</td>
      <td>0.463382</td>
      <td>0.201937</td>
      <td>1.488284</td>
      <td>1</td>
      <td>True</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ExtraTreesEntr_BAG_L1</td>
      <td>1.000000</td>
      <td>0.954426</td>
      <td>0.467937</td>
      <td>0.203993</td>
      <td>1.483316</td>
      <td>0.467937</td>
      <td>0.203993</td>
      <td>1.483316</td>
      <td>1</td>
      <td>True</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ExtraTreesGini_BAG_L2</td>
      <td>0.997955</td>
      <td>0.955595</td>
      <td>16.316356</td>
      <td>2.193562</td>
      <td>128.810273</td>
      <td>0.428131</td>
      <td>0.219939</td>
      <td>1.498398</td>
      <td>2</td>
      <td>True</td>
      <td>19</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ExtraTreesEntr_BAG_L2</td>
      <td>0.996202</td>
      <td>0.957639</td>
      <td>16.317497</td>
      <td>2.179946</td>
      <td>128.802242</td>
      <td>0.429272</td>
      <td>0.206324</td>
      <td>1.490367</td>
      <td>2</td>
      <td>True</td>
      <td>20</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NeuralNetMXNet_BAG_L2</td>
      <td>0.976629</td>
      <td>0.958808</td>
      <td>29.310920</td>
      <td>3.639860</td>
      <td>456.264859</td>
      <td>13.422695</td>
      <td>1.666238</td>
      <td>328.952984</td>
      <td>2</td>
      <td>True</td>
      <td>22</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RandomForestGini_BAG_L2</td>
      <td>0.971078</td>
      <td>0.958224</td>
      <td>16.231102</td>
      <td>2.191179</td>
      <td>128.912157</td>
      <td>0.342877</td>
      <td>0.217556</td>
      <td>1.600282</td>
      <td>2</td>
      <td>True</td>
      <td>16</td>
    </tr>
    <tr>
      <th>9</th>
      <td>RandomForestEntr_BAG_L2</td>
      <td>0.971078</td>
      <td>0.957639</td>
      <td>16.313704</td>
      <td>2.189527</td>
      <td>128.810289</td>
      <td>0.425479</td>
      <td>0.215904</td>
      <td>1.498414</td>
      <td>2</td>
      <td>True</td>
      <td>17</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LightGBMLarge_BAG_L1</td>
      <td>0.966404</td>
      <td>0.957639</td>
      <td>0.307789</td>
      <td>0.105921</td>
      <td>13.135603</td>
      <td>0.307789</td>
      <td>0.105921</td>
      <td>13.135603</td>
      <td>1</td>
      <td>True</td>
      <td>12</td>
    </tr>
    <tr>
      <th>11</th>
      <td>XGBoost_BAG_L2</td>
      <td>0.966404</td>
      <td>0.959684</td>
      <td>16.405802</td>
      <td>2.073196</td>
      <td>147.476847</td>
      <td>0.517577</td>
      <td>0.099573</td>
      <td>20.164972</td>
      <td>2</td>
      <td>True</td>
      <td>21</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LightGBM_BAG_L2</td>
      <td>0.965819</td>
      <td>0.960561</td>
      <td>16.101750</td>
      <td>2.063482</td>
      <td>132.439821</td>
      <td>0.213525</td>
      <td>0.089859</td>
      <td>5.127946</td>
      <td>2</td>
      <td>True</td>
      <td>15</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WeightedEnsemble_L3</td>
      <td>0.965819</td>
      <td>0.960561</td>
      <td>16.109449</td>
      <td>2.074748</td>
      <td>134.519620</td>
      <td>0.007699</td>
      <td>0.011266</td>
      <td>2.079799</td>
      <td>3</td>
      <td>True</td>
      <td>24</td>
    </tr>
    <tr>
      <th>14</th>
      <td>XGBoost_BAG_L1</td>
      <td>0.964943</td>
      <td>0.958224</td>
      <td>0.473526</td>
      <td>0.095039</td>
      <td>2.660855</td>
      <td>0.473526</td>
      <td>0.095039</td>
      <td>2.660855</td>
      <td>1</td>
      <td>True</td>
      <td>10</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LightGBMLarge_BAG_L2</td>
      <td>0.964067</td>
      <td>0.958224</td>
      <td>16.248666</td>
      <td>2.066210</td>
      <td>139.900362</td>
      <td>0.360441</td>
      <td>0.092587</td>
      <td>12.588487</td>
      <td>2</td>
      <td>True</td>
      <td>23</td>
    </tr>
    <tr>
      <th>16</th>
      <td>NeuralNetMXNet_BAG_L1</td>
      <td>0.963774</td>
      <td>0.956179</td>
      <td>12.639247</td>
      <td>0.515805</td>
      <td>88.646938</td>
      <td>12.639247</td>
      <td>0.515805</td>
      <td>88.646938</td>
      <td>1</td>
      <td>True</td>
      <td>11</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CatBoost_BAG_L2</td>
      <td>0.963774</td>
      <td>0.959684</td>
      <td>15.996216</td>
      <td>2.038080</td>
      <td>135.190438</td>
      <td>0.107991</td>
      <td>0.064458</td>
      <td>7.878563</td>
      <td>2</td>
      <td>True</td>
      <td>18</td>
    </tr>
    <tr>
      <th>18</th>
      <td>LightGBM_BAG_L1</td>
      <td>0.959977</td>
      <td>0.957347</td>
      <td>0.192819</td>
      <td>0.087040</td>
      <td>5.580223</td>
      <td>0.192819</td>
      <td>0.087040</td>
      <td>5.580223</td>
      <td>1</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19</th>
      <td>CatBoost_BAG_L1</td>
      <td>0.959684</td>
      <td>0.956471</td>
      <td>0.115887</td>
      <td>0.055576</td>
      <td>6.871174</td>
      <td>0.115887</td>
      <td>0.055576</td>
      <td>6.871174</td>
      <td>1</td>
      <td>True</td>
      <td>7</td>
    </tr>
    <tr>
      <th>20</th>
      <td>LightGBMXT_BAG_L1</td>
      <td>0.957639</td>
      <td>0.958224</td>
      <td>0.247945</td>
      <td>0.090365</td>
      <td>4.423393</td>
      <td>0.247945</td>
      <td>0.090365</td>
      <td>4.423393</td>
      <td>1</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>WeightedEnsemble_L2</td>
      <td>0.957639</td>
      <td>0.958224</td>
      <td>0.257359</td>
      <td>0.102480</td>
      <td>6.991363</td>
      <td>0.009414</td>
      <td>0.012115</td>
      <td>2.567971</td>
      <td>2</td>
      <td>True</td>
      <td>13</td>
    </tr>
    <tr>
      <th>22</th>
      <td>KNeighborsUnif_BAG_L1</td>
      <td>0.954134</td>
      <td>0.953257</td>
      <td>0.131725</td>
      <td>0.106717</td>
      <td>0.012008</td>
      <td>0.131725</td>
      <td>0.106717</td>
      <td>0.012008</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LightGBMXT_BAG_L2</td>
      <td>0.953550</td>
      <td>0.957347</td>
      <td>16.081167</td>
      <td>2.063100</td>
      <td>131.712806</td>
      <td>0.192942</td>
      <td>0.089477</td>
      <td>4.400931</td>
      <td>2</td>
      <td>True</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>



Il est egalement possible d'afficher les carracteristiques qui contribut le plus à notre score par ordre d'importance avec la methode **predictor.feature_importance(data=df_train)**


```python
predictor.feature_importance(data=df_train)
```

    Computing feature importance via permutation shuffling for 11 features using 1000 rows with 3 shuffle sets...
    	248.34s	= Expected runtime (82.78s per shuffle set)
    	151.03s	= Actual runtime (Completed 3 of 3 shuffle sets)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
      <th>stddev</th>
      <th>p_value</th>
      <th>n</th>
      <th>p99_high</th>
      <th>p99_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bmi</th>
      <td>0.011667</td>
      <td>0.002082</td>
      <td>0.005223</td>
      <td>3</td>
      <td>0.023595</td>
      <td>-0.000262</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.011667</td>
      <td>0.005132</td>
      <td>0.029427</td>
      <td>3</td>
      <td>0.041071</td>
      <td>-0.017738</td>
    </tr>
    <tr>
      <th>avg_glucose_level</th>
      <td>0.005667</td>
      <td>0.001528</td>
      <td>0.011688</td>
      <td>3</td>
      <td>0.014420</td>
      <td>-0.003086</td>
    </tr>
    <tr>
      <th>id</th>
      <td>0.005667</td>
      <td>0.003055</td>
      <td>0.042375</td>
      <td>3</td>
      <td>0.023172</td>
      <td>-0.011839</td>
    </tr>
    <tr>
      <th>ever_married</th>
      <td>0.003667</td>
      <td>0.002082</td>
      <td>0.046368</td>
      <td>3</td>
      <td>0.015595</td>
      <td>-0.008262</td>
    </tr>
    <tr>
      <th>heart_disease</th>
      <td>0.002667</td>
      <td>0.000577</td>
      <td>0.007634</td>
      <td>3</td>
      <td>0.005975</td>
      <td>-0.000642</td>
    </tr>
    <tr>
      <th>hypertension</th>
      <td>0.002667</td>
      <td>0.001155</td>
      <td>0.028595</td>
      <td>3</td>
      <td>0.009283</td>
      <td>-0.003950</td>
    </tr>
    <tr>
      <th>work_type</th>
      <td>0.002000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>3</td>
      <td>0.002000</td>
      <td>0.002000</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>0.001667</td>
      <td>0.000577</td>
      <td>0.018875</td>
      <td>3</td>
      <td>0.004975</td>
      <td>-0.001642</td>
    </tr>
    <tr>
      <th>Residence_type</th>
      <td>0.001333</td>
      <td>0.000577</td>
      <td>0.028595</td>
      <td>3</td>
      <td>0.004642</td>
      <td>-0.001975</td>
    </tr>
    <tr>
      <th>smoking_status</th>
      <td>0.000667</td>
      <td>0.000577</td>
      <td>0.091752</td>
      <td>3</td>
      <td>0.003975</td>
      <td>-0.002642</td>
    </tr>
  </tbody>
</table>
</div>



## test du model predictif

Une fois l'entrainement terminer et le modele pret à l'utilisation, on peut l'utiliser sur un jeux de données pour prédire le risque d'accident vasculaire cerebrale.

Pour ce faire il suffit de fournir un jeux de données à notre predictor et stocker le resultat dans un dataframe.


```python
# Test de mon modele avec un jeu de données

y_pred = predictor.predict(test_data)
df_pred = pd.DataFrame(y_pred, columns=['stroke'])
df_pred
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4673</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3232</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3694</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1070</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4163</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>386</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3961</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1608</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4058</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1687 rows × 1 columns</p>
</div>



Pour comprendre comment est effectué l'evaluation de la precision de cette prediction, la methode **predictor.evaluate(df_test)** est utilisée.


```python
predictor.evaluate(df_test)
```

    Evaluation: accuracy on test data: 0.943687018375815
    Evaluations on test data:
    {
        "accuracy": 0.943687018375815,
        "balanced_accuracy": 0.5036770333263758,
        "mcc": 0.025709735794486963,
        "roc_auc": 0.8405830376400194,
        "f1": 0.02061855670103093,
        "precision": 0.14285714285714285,
        "recall": 0.011111111111111112
    }





    {'accuracy': 0.943687018375815,
     'balanced_accuracy': 0.5036770333263758,
     'mcc': 0.025709735794486963,
     'roc_auc': 0.8405830376400194,
     'f1': 0.02061855670103093,
     'precision': 0.14285714285714285,
     'recall': 0.011111111111111112}



## Resultat

Le prétraitement des données et l'ingénierie des fonctionnalités ont été réalisés par AutoGluon.
Le modèle formé inclut également la validation croisée.

Ainsi, nous avons obtenu le classificateur formé à une précision de 95 % avec seulement deux lignes de code (pour que le classificateur s'entraîne et prédise).

C'est impressionnant !

Avec un modèle ML traditionnel, nous passerions beaucoup de temps à terminer l'ensemble du processus, y compris l'analyse exploratoire des données, le nettoyage des données ainsi que le codage pour configurer plusieurs modèles.

AutoGluon nous a rendu cela assez simple en automatisant toutes les etapes.


```python

```
