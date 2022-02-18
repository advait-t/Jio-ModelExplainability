import sys
from typing import final
from pycaret.classification import compare_models, create_model, finalize_model, get_config, predict_model, save_config, save_model, setup, tune_model
import all_function
import pandas as pd
import seaborn as sns
from main import *
from all_function import *

def setting_up_automl(data, final_columns, parameter_to_be_optimized, target):
    input_data = data[final_columns]
    models = ['qda','lda','ada']
    ordinal_column = all_function.ordinal_columns(input_data, target)
    data_setup = setup(data = input_data, session_id=42,fix_imbalance = True, target = target, fold = 2, ordinal_features = ordinal_column, use_gpu = False, silent = True, remove_multicollinearity = True)
    
    with st.spinner('Setting up Data'):
        temp_dir = tempfile.TemporaryDirectory()
        save_config((str(temp_dir.name)+'/configs.pkl'))
        configs_path = (str(temp_dir.name)+'/configs.pkl')
        download_configs(save_config(configs_path))

    #choosing the best model
    with st.spinner('Building, comparing and selecting the best model......'):
        best_model = compare_models(sort = parameter_to_be_optimized, turbo = True,  exclude = models)
        best_model_results = pull()
        model_name = best_model_results['Model'][0]
        st.dataframe(best_model_results)
        st.caption(f'Compared all the models and selecting the best model for "{parameter_to_be_optimized}"')
        st.caption(f'The best model chosen for "{parameter_to_be_optimized}" is "{model_name}"')
        model = create_model(best_model)

    #tuning the model for getting the best F1-score
    with st.spinner('Tuning the best model......'):
        tuned_model = tune_model(model, optimize = parameter_to_be_optimized, choose_better = True)

    #finalising the model and saving it
    with st.spinner('Finalising Model'):
        final_model = finalize_model(tuned_model)
        download_model(save_model(final_model, (str(temp_dir.name)+'/final_model')))

    # Printing the model metrics
    eval_model = evaluate_model(final_model)
    evaluated_model = pull()
    st.write(f'{parameter_to_be_optimized} value after optimization :', (evaluated_model[parameter_to_be_optimized][0])*100)
    return final_model, data_setup, best_model, configs_path, best_model_results, model_name
        
          
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
    with st.spinner('Setting up Data'):
        input_data = data[final_columns]
        data_setup = setup(data = input_data, target = target, session_id=42 ,preprocess = False, html = False, silent = True)

    #choosing the best model
    with st.spinner('Building the best model'):
        model = create_model(best_model)

    #tuning the model for getting the best F1-score
    with st.spinner('Tuning the best model'):
        tuned_model = tune_model(model, optimize = parameter_to_be_optimized, choose_better = True)

    #finalising the model and saving it
    final_model = finalize_model(tuned_model)
    temp_dir1 = tempfile.TemporaryDirectory()
    output_file_path = (str(temp_dir1.name)+'/final_model')
    download_model(save_model(final_model, output_file_path))
    return final_model, data_setup