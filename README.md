# RNN-VAE


## Run the code 
- python >= 3.6
- tensorflow >= 2.4

We recommend using [Anaconda](https://www.anaconda.com/). 

Open Anaconda Prompt to execute the following lines

### Environment 
To create an environment with the requirements for the detector to work, execute the following line.
```
conda create --name <env> --file requirements.txt
```
The library to run the custom metrics specified in the paper can only be installed with pip. If you want to evaluate your results with these metrics, install the [_prts_](https://github.com/CompML/PRTS) library.

```
pip install prts 
```

After you create the environment, activate it.
```
conda activate <env>
```

### Data
The data format must be a table where the first column corresponds to the timestamp. They must be saved in a .csv file.

### Settings
All the values and hyperparameters necessary to create a DC-VAE model must be specified in a .txt file. An example of this can be seen in the _settings_ folder.

### Training
To train a model you must execute the *train.py* script and indicate the data train and settings files.

```
python train.py data_train.csv settings\model_settings.txt
```

### Alpha definition
If the training set contains labels, the alpha values can be adjusted in order to maximize the value of F1 in detections. Execute the *alpha_definition.py* script and indicate the train data and labels files and the settings file.

```
python alpha_definition.py data_train.csv labels_train.csv settings\model_settings.txt
```

### Testing
To test the trained model execute the *test.py* script and indicate the test data and settings files.

```
python test.py data_test.csv settings\model_settings.txt
```


## Cite
```

```

## Acknowledgment
This work has been partially supported by the ANII-FMV project with reference FMV-1-2019-1-155850 Anomaly Detection with Continual and Streaming Machine Learning on Big Data Telecommunications Networks, by Telefonica, and by the Austrian FFG ICT-of-the-Future project _DynAISEC – Adaptive AI/ML for Dynamic Cybersecurity Systems_. Gaston García González a was supported by the ANII scholarship POS-FMV-2020-1-1009239, and by CSIC, under program Movilidad e Intercambios Academicos 2022.

