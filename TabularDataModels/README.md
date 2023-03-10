### How to run

`python train.py -config config.yml`

The following is an example of the `config.yml` file where all the arguments related to training and testing are set:

```
    patience: 50 # For early stopping. How many epochs to wait.
    epochs: 100 # number of epochs
    b_sz: 256 # batch size
    is_valid : 1
    report_test: 0
    train_path: <path_to_train>
    valid_path: <path_to_valid>
    test_path: <path_to_train>
    n_classes: 2 # number of classes
    ckpt_path: <load the model from a specific checkpoint> (optional)
    num_features: <csv_file_with_numerical_features>
    cat_features: <csv_file_with_categorical_features>
    # label_clm: default.payment.next.month
    label_clm: y
    model_storage_path: trained_weights
```

The user has to provide a list of numerical and categorical columns (categorical columns are otpional). 


To prepare the data in a specific format, run `PrepareData.ipynb`.

### Categorical Columns

![alt text](images/table_example.png "Logo Title Text 1")

The format is a pandas dataframe with the first row indicating the categorical column names and the second row should contain the **number of unique values** for each categorical feature.