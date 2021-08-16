from classification_modelling_pipeline import *
from explainability_pipeline import *
import streamlit as st
from all_function import *

st.cache()

st.title('Automation of Model Building and Explainability of Models')

data = file_upload()
columns = list(data.columns)
target = st.selectbox('Select your target variable', columns)

container1 = st.beta_container()
all1 = st.checkbox("Select all")
if all1:
    final_columns1 = container1.multiselect("Select columns for model building:", columns, columns)
else:
    final_columns1 =  container1.multiselect("Select columns for model building:", columns)

parameter_to_be_optimized = st.selectbox('Select a parameter you want to optimize', ['Accuracy','AUC','Recall' ,'Prec.', 'F1', 'Kappa', 'MCC'])

button = st.selectbox('Do you want to build model?',['No','Yes'])
if button == 'Yes':
    final_model1, data_setup1, best_model = setting_up_automl(data, final_columns1, parameter_to_be_optimized, target)
    X_train, y_train, X_test, y_test = save_feature_importance(final_model1)
    retrain_data = data_set(X_train, y_train, X_test, y_test)
    target_column = y_train.name
    train_set_columns = list(retrain_data.columns)
    SE, predictions = save_shap_file(final_model1, X_test)
    saving_top_features(SE, predictions, target_column)
    rebuild = st.selectbox('Do you want to remove any features and rebuild the model?', ['Yes', 'No'])
    if rebuild == 'Yes':
        container2 = st.beta_container()
        all2 = st.checkbox("Select all columns")
        if all2:
            final_columns2 = container2.multiselect("Select columns for final model building:", train_set_columns, train_set_columns)
        else:
            final_columns2 =  container2.multiselect("Select columns for final model building:", train_set_columns)
        retrain = st.selectbox('Rebuild Model?', ['No','Yes'])
        if retrain == 'Yes':
            final_model2, data_setup2 = retrain_model(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
            X_train_new, y_train_new, X_test_new, y_test_new = save_feature_importance(final_model2)
            SE, predictions = save_shap_file(final_model2, X_test_new)
            saving_top_features(SE, predictions, target_column)
