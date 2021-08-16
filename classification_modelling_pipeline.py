import sys
from pycaret.classification import compare_models, create_model, finalize_model, get_config, predict_model, save_config, save_model, setup, tune_model
import all_function
import pandas as pd
import seaborn as sns
from main import *
from all_function import *
# take features to be taken from the user
# take accuracy/f1/others as input from user

# getting the data and the target column
# data = (sys.argv[1])
# target = (sys.argv[2])
# parameter_to_be_optimized = (sys.argv[3])

# reading the input data
# input_data = all_function.read_data(data)

def setting_up_automl(data, final_columns, parameter_to_be_optimized, target):
    # setting up pycaret
    input_data = data[final_columns]
    ordinal_column = all_function.ordinal_columns(input_data, target)
    data_setup = setup(data = input_data, session_id=42, fix_imbalance = True, target = target, fold = 2, ordinal_features = ordinal_column, use_gpu = False, silent = True, remove_multicollinearity = True)
    status = st.empty()
    status.write('Setting up the data  .......')
    download_configs(save_config('data_configs.pkl'))
    status.write('Data Setup Done')

    #choosing the best model
    status.write('Building, comparing and selecting the best model......')
    best_model = compare_models(sort = parameter_to_be_optimized, turbo = True)
    best_model_results = pull()
    st.write(best_model_results)
    model = create_model(best_model)

    #tuning the model for getting the best F1-score
    status.write('Tuning the best model......')
    tuned_model = tune_model(model, optimize = parameter_to_be_optimized, choose_better = True)

    #finalising the model and saving it
    final_model = finalize_model(tuned_model)
    download_model(save_model(final_model, 'final_model'))
    status.write('Model Building Done')
    return final_model, data_setup, best_model
    
def save_feature_importance(final_model):
    #saving the feature importances as a png file
    X_train = get_config(variable="X_train")
    y_train = get_config(variable="y_train")
    X_test = get_config(variable="X_test")
    y_test = get_config(variable="y_test")
    drop_one_column_feature_importance(X_train, y_train, final_model)
    return X_train, y_train, X_test, y_test


def data_set(X_train, y_train, X_test, y_test):
    train_data = pd.concat([X_train, y_train], axis = 1)
    test_data = pd.concat([X_test,y_test], axis = 1)
    data = pd.concat([train_data, test_data], axis = 0)
    return data

def retrain_model(data, final_columns, parameter_to_be_optimized, target, best_model):
    # setting up pycaret
    input_data = data[final_columns]
    data_setup = setup(data = input_data, target = target, session_id=42 ,preprocess = False, html = False, silent = True)
    status = st.empty()
    status.write('Setting up the data  .......')
    status.write('Data Setup Done')

    #choosing the best model
    status.write('Building the best model......')
    model = create_model(best_model)

    #tuning the model for getting the best F1-score
    status.write('Tuning the best model......')
    tuned_model = tune_model(model, optimize = parameter_to_be_optimized, choose_better = True)

    #finalising the model and saving it
    final_model = finalize_model(tuned_model)
    download_model(save_model(final_model, 'final_model'))
    status.write('Model Building Done')
    return final_model, data_setup