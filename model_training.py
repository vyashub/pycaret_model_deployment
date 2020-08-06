# Importing data using pandas
import pandas as pd
loan_df = pd.read_csv('bankloan.csv')

loan_df.dropna(inplace=True)

# Importing module and initializing setup
from pycaret.classification import *
log_reg = setup(data = loan_df, target = 'Loan_Status',ignore_features=['Loan_ID'], silent=True)

# create a model
lr = create_model('lr')


# finalize a model
lr = finalize_model(lr)

save_model(lr, 'lr_deployed_model')