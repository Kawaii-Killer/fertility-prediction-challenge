"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    ## This script contains a bare minimum working example
    #random.seed(1) # not useful here because logistic regression deterministic
    
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  
    
    # Logistic regression model
    model = LogisticRegression()

    # Fit the model
    model.fit(model_df[['ci20m101', 'cv20l003', 'cv20l002', 'ci20m154', 'cw20m000', 'cw20m572',
       'cv19k067', 'cv19k073', 'cw20m095', 'cv19k003', 'ca18f008', 'ch19l219',
       'ch19l228', 'cv18j003', 'cw20m012', 'ch20m219', 'ch20m003', 'ch20m228',
       'ch19l003', 'ch18k003', 'cs18k449', 'ch18k219', 'cs20m449', 'ca20g008',
       'cs18k296', 'cs18k005', 'cs19l449', 'cs19l298']], model_df['new_child'])

    # Save the model
    joblib.dump(model, "model.joblib")
