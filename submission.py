"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """

    # Selecting variables for modelling
    keepcols = [
    "ci20m101",
    "ci20m003",
    "cv20l003",
    "cv20l002",
    "ci20m154",
    "ci20m100",
    "ci20m150",
    "ci20m146",
    "cw20m000",
    "cw20m572",
    "cv19k067",
    "cv19k073",
    "cw20m001",
    "cw20m095",
    "cw20m088",
    "cv19k003",
    "ca20g001",
    "ca18f008",
    "ca18f078",
    "ch19l219",
    "ch19l228",
    "ch19l077",
    "cv18j003",
    "cv18j166",
    "cw20m012",
    "cw20m011",
    "ch18k077",
    "ch20m219",
    "ch20m003",
    "ch20m228",
    "ch19l003",
    "ch18k003",
    "cs20m294",
    "cs18k449",
    "ch18k219",
    "ch18k227",
    "cs20m028",
    "cs20m029",
    "cs20m023",
    "cs20m026",
    "cs20m018",
    "cs20m038",
    "cs20m008",
    "cs20m523",
    "cs20m041",
    "cs20m021",
    "cs20m016",
    "cs20m449",
    "ca20g008",
    "ca20g078",
    "ca20g009",
    "cs18k296",
    "cs18k028",
    "cs18k005",
    "cs18k029",
    "cs18k038",
    "cs18k023",
    "cs18k018",
    "cs18k039",
    "cs18k026",
    "ch18k097",
    "cs19l449",
    "cs19l041",
    "cs19l038",
    "cs19l029",
    "cs19l028",
    "cs19l039",
    "cs19l040",
    "cs19l445",
    "cs19l018",
    "cs19l298",
    "partner_2018",
    "partner_2019",
    "partner_2020",
    "nomem_encr"
]

    # Keeping data with variables selected
    df = df[keepcols]


    # Create an instance of IterativeImputer
    imputer = IterativeImputer(random_state=0, verbose=2)

    # Fit the imputer on the data
    imputer.fit(df)

    # Transform the data with imputation
    imputed_values = imputer.transform(df)

    # Convert the imputed array back to DataFrame
    df_imputed = pd.DataFrame(imputed_values, columns=df.columns, index=df.index)

    df_imputed = df_imputed[['ci20m101', 'cv20l003', 'cv20l002', 'ci20m154', 'cw20m000', 'cw20m572',
       'cv19k067', 'cv19k073', 'cw20m095', 'cv19k003', 'ca18f008', 'ch19l219',
       'ch19l228', 'cv18j003', 'cw20m012', 'ch20m219', 'ch20m003', 'ch20m228',
       'ch19l003', 'ch18k003', 'cs18k449', 'ch18k219', 'cs20m449', 'ca20g008',
       'cs18k296', 'cs18k005', 'cs19l449', 'cs19l298', 'nomem_encr']]
    return df_imputed


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict

