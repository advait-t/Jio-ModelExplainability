from pickle import NONE

from streamlit.state.session_state import SessionState
from classification_modelling_pipeline import *
from regression_pipeline import *
from explainability_pipeline import *
import streamlit as st
from all_function import *

# Function for EDA
# def eda():
#     st.subheader('Exploratory Data Analysis')
#     file = st.file_uploader('Upload Dataset', type = ['csv', 'txt'])
#     if file is not None:
#         df = pd.read_csv(file)
#         all_columns = df.columns.to_list()
#         st.dataframe(df.head())

#         if st.checkbox("Show Shape"): 
#             st.write(df.shape)

#         if st.checkbox("Show Columns"):
#             st.write(all_columns)

#         if st.checkbox("Summary"):
#             st.write(df.describe())

#         if st.checkbox("Show Selected Columns"):
#             selected_columns = st.multiselect("Select Columns",all_columns)
#             new_df = df[selected_columns]
#             st.dataframe(new_df)

#         if st.checkbox("Show Value Counts"):
#             selected_columns1 = st.multiselect("Select Column",all_columns)
#             #target = st.selectbox('Target Variable', all_columns)
#             new_df1 = df[selected_columns1]
#             for i in selected_columns1:
#                 st.write(new_df1[i].value_counts())

#         if st.checkbox("Correlation Plot(Matplotlib)"):
#             plt.matshow(df.corr())
#             st.pyplot()

#         if st.checkbox("Correlation Plot(Seaborn)"):
#             fig = st.write(sns.heatmap(df.corr(),annot=True))
#             st.pyplot(fig)

#         if st.checkbox("Pie Plot"):
#             all_columns = df.columns.to_list()
#             column_to_plot = st.selectbox("Select 1 Column",all_columns)
#             pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
#             st.write(pie_plot)
#             st.pyplot() 

def eda():


# Function to make plots
def plots():
    st.subheader("Data Visualization")
    data = st.file_uploader("Upload Dataset", type=["csv", "txt"])

    if data is not None:
        df = pd.read_csv(data)
        st.dataframe(df.head())


        if st.checkbox("Show Value Counts"):
            st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
            st.pyplot()
    
        # Customizable Plot

        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
        selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

            # Plot By Streamlit
            if type_of_plot == 'area':
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)

            elif type_of_plot == 'bar':
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)

            elif type_of_plot == 'line':
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)

            # Custom Plot 
            elif type_of_plot:
                cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot() 

def classifiction_model_function(data, final_columns1, parameter_to_be_optimized, target):
    final_model1, data_setup1, best_model, configs_path, best_model_results, model_name = setting_up_automl(data, final_columns1, parameter_to_be_optimized, target)
    X_train, y_train, X_test, y_test = save_feature_importance(final_model1)
    retrain_data = data_set(X_train, y_train, X_test, y_test)
    target_column = y_train.name
    train_set_columns = list(retrain_data.columns)
    SE, predictions = save_shap_file(final_model1, X_test)
    # saving_top_features(SE, predictions, target_column)
    return train_set_columns, best_model, data_setup1, retrain_data, target_column, best_model_results, model_name, SE, predictions

def regression_model_function(data, final_columns1, parameter_to_be_optimized, target):
    final_model1, data_setup1, best_model, configs_path, best_model_results, model_name = setting_up_automl_regression(data, final_columns1, parameter_to_be_optimized, target)
    X_train, y_train, X_test, y_test = save_feature_importance_regression(final_model1)
    retrain_data = data_set_regression(X_train, y_train, X_test, y_test)
    target_column = y_train.name
    train_set_columns = list(retrain_data.columns)
    SE, predictions = save_shap_file(final_model1, X_test)
    # saving_top_features(SE, predictions, target_column)
    return train_set_columns, best_model, data_setup1, retrain_data, target_column, best_model_results, model_name, SE, predictions


def select_data():
    file = st.file_uploader('Upload Dataset', type = ['csv', 'txt'])
    if file is not None:
        data = pd.read_csv(file)
        return data

def columns_for_model_building(data):
    columns = list(data.columns)

    target = st.selectbox('Select your target variable', columns)
    if target is not None:
        container1 = st.container()
        all1 = st.checkbox("Select all")
        if all1:
            final_columns1 = container1.multiselect("Select columns for model building:", columns, columns)
        else:
            final_columns1 =  container1.multiselect("Select columns for model building:", columns)

    parameter_to_be_optimized = st.selectbox('Select a parameter you want to optimize', ['Accuracy','AUC','Recall' ,'Prec.', 'F1', 'Kappa', 'MCC'])
    
    return data, final_columns1, parameter_to_be_optimized, target

def columns_for_model_building_regression(data):
    columns = list(data.columns)

    target = st.selectbox('Select your target variable', columns)
    if target is not None:
        container1 = st.container()
        all1 = st.checkbox("Select all")
        if all1:
            final_columns1 = container1.multiselect("Select columns for model building:", columns, columns)
        else:
            final_columns1 =  container1.multiselect("Select columns for model building:", columns)

    parameter_to_be_optimized = st.selectbox('Select a parameter you want to optimize', ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'])
    return data, final_columns1, parameter_to_be_optimized, target

def retrain_model_function(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model):
    final_model2, data_setup2 = retrain_model(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
    X_train_new, y_train_new, X_test_new, y_test_new = save_feature_importance(final_model2)
    SE, predictions = save_shap_file(final_model2, X_test_new)
    saving_top_features(SE, predictions, target_column)
    return final_model2

def retrain_model_function_regression(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model):
    final_model2, data_setup2 = retrain_model_regression(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
    X_train_new, y_train_new, X_test_new, y_test_new = save_feature_importance_regression(final_model2)
    SE, predictions = save_shap_file(final_model2, X_test_new)
    saving_top_features(SE, predictions, target_column)
    return final_model2


def main():
    st.set_page_config(layout="wide")
    st.title('Automated EDA, Model Building and Explainability')
    st.subheader('Automation of Model Building and Explainability of Models')
    side_bar = ['EDA','Visualisation','Classification Model Building', 'Regression Model Building']
    choice = st.sidebar.selectbox('Select your task',side_bar)
    if choice == 'EDA':
        eda()
    elif choice == 'Visualisation':
        plots()
    elif choice == 'Classification Model Building':
        file = select_data()
        if file is not None:
            st.dataframe(file)
            st.caption('Uploaded Dataset')
            data, final_columns1, parameter_to_be_optimized, target = columns_for_model_building(file)
            button_state = st.checkbox('Build Model')
            list_of_variables = ['best_model', 'target_column','shapash','predictions','best_model_results','parameter_to_be_optimised', 'model_name', 'train_set_columns', 'retrain_data']
            
            if button_state:
                if any(x in st.session_state for x in list_of_variables):
                    st.write('Model is already built')
                    parameter_to_be_optimized = st.session_state.parameter_to_be_optimized
                    best_model = st.session_state.best_model
                    model_name = st.session_state.model_name
                    best_model_results = st.session_state.best_model_results
                    train_set_columns = st.session_state.train_set_columns
                    retrain_data = st.session_state.retrain_data
                    target_column = st.session_state.target_column
                    shapash = st.session_state.shapash
                    predictions = st.session_state.predictions
                else:
                    train_set_columns, best_model, data_setup1, retrain_data, target_column, best_model_results, model_name, shapash, predictions = classifiction_model_function(data, final_columns1, parameter_to_be_optimized, target)
                
                saving_top_features(shapash, predictions, target_column)
                st.dataframe(best_model_results)
                st.caption(f'Model Selected for "{parameter_to_be_optimized}" is {model_name}')
                
                graphs = ['auc','threshold','pr','confusion_matrix','error','class_report','feature','feature_all']
                selected_graph = st.selectbox('Graphs', graphs)
                plot_model(best_model, plot = selected_graph, display_format='streamlit')
                st.session_state.best_model = best_model
                st.session_state.best_model_results = best_model_results
                st.session_state.parameter_to_be_optimized = parameter_to_be_optimized
                st.session_state.model_name = model_name
                st.session_state.train_set_columns = train_set_columns
                st.session_state.retrain_data = retrain_data
                st.session_state.target_column = target_column
                st.session_state.shapash = shapash
                st.session_state.predictions = predictions

                rebuild = st.selectbox('Do you want to remove any features and rebuild the model?', ['No', 'Yes'])
                if rebuild == 'Yes':
                    container2 = st.container()
                    all2 = st.checkbox("Select all columns")
                    if all2:
                        final_columns2 = container2.multiselect("Select columns for final model building:", train_set_columns, train_set_columns)
                        
                        if st.checkbox('Rebuild Model'):
                            final_model2 = retrain_model_function(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
                            # graphs = ['auc','threshold','pr','confusion_matrix','error','class_report','feature','feature_all']
                            # selected_graph1 = st.selectbox('Graphs', graphs)
                            # plot_model(final_model2, plot = selected_graph1, display_format='streamlit')
                    else:
                        final_columns2 =  container2.multiselect("Select columns for final model building:", train_set_columns)
                        
                        if st.checkbox('Rebuild Model'):
                            final_model2 = retrain_model_function(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
                            # graphs = ['auc','threshold','pr','confusion_matrix','error','class_report','feature','feature_all']
                            # selected_graph2 = st.selectbox('Graphs', graphs)
                            # plot_model(final_model2, plot = selected_graph2, display_format='streamlit')
                            # rebuild_model(train_set_columns, best_model, data_setup1, retrain_data, target_column, parameter_to_be_optimized)
                if rebuild == 'No':
                    st.write('Thank You, Model Building Done.')
    else:
        file = select_data()
        if file is not None:
            st.dataframe(file)
            st.caption('Uploaded Dataset')
            data_regression, final_columns1_regression, parameter_to_be_optimized_regression, target_regression = columns_for_model_building_regression(file)
            button_state_regression = st.checkbox('Build Model')
            list_of_variables_regression = ['best_model', 'target_column','shapash','predictions','best_model_results','parameter_to_be_optimised', 'model_name', 'train_set_columns', 'retrain_data']
            
            if button_state_regression:
                if any(x in st.session_state for x in list_of_variables_regression):
                    st.write('Model is already built')
                    parameter_to_be_optimized_regression = st.session_state.parameter_to_be_optimized_regression
                    best_model_regression = st.session_state.best_model_regression
                    model_name_regression = st.session_state.model_name_regression
                    best_model_results_regression = st.session_state.best_model_results_regression
                    train_set_columns_regression = st.session_state.train_set_columns_regression
                    retrain_data_regression = st.session_state.retrain_data_regression
                    target_column_regression = st.session_state.target_column_regression
                    shapash_regression = st.session_state.shapash_regression
                    predictions_regression = st.session_state.predictions_regression
                else:
                    train_set_columns_regression, best_model_regression, data_setup1_regression, retrain_data_regression, target_column_regression, best_model_results_regression, model_name_regression, shapash_regression, predictions_regression = regression_model_function(data_regression, final_columns1_regression, parameter_to_be_optimized_regression, target_regression)
                
                saving_top_features(shapash_regression, predictions_regression, target_column_regression)
                st.dataframe(best_model_results_regression)
                st.caption(f'Model Selected for "{parameter_to_be_optimized_regression}" is {model_name_regression}')
                
                graphs = ['residuals','error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'parameter']
                selected_graph = st.selectbox('Graphs', graphs)
                plot_model(best_model_regression, plot = selected_graph, display_format='streamlit')

                st.session_state.best_model_regression = best_model_regression
                st.session_state.best_model_results_regression = best_model_results_regression
                st.session_state.parameter_to_be_optimized_regression = parameter_to_be_optimized_regression
                st.session_state.model_name_regression = model_name_regression
                st.session_state.train_set_columns_regression = train_set_columns_regression
                st.session_state.retrain_data_regression = retrain_data_regression
                st.session_state.target_column_regression = target_column_regression
                st.session_state.shapash_regression = shapash_regression
                st.session_state.predictions_regression = predictions_regression

                rebuild = st.selectbox('Do you want to remove any features and rebuild the model?', ['No', 'Yes'])
                if rebuild == 'Yes':
                    container2 = st.container()
                    all2 = st.checkbox("Select all columns")
                    if all2:
                        final_columns2 = container2.multiselect("Select columns for final model building:", train_set_columns_regression, train_set_columns_regression)
                        
                        if st.checkbox('Rebuild Model'):
                            final_model2_regression = retrain_model_function_regression(retrain_data_regression, final_columns2, parameter_to_be_optimized_regression, target_column_regression, best_model_regression)
                            graphs = ['residuals','error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'parameter']
                            selected_graph1 = st.selectbox('Graphs', graphs)
                            plot_model(final_model2_regression, plot = selected_graph1, display_format='streamlit')
                    else:
                        final_columns2_regression =  container2.multiselect("Select columns for final model building:", train_set_columns_regression)
                        
                        if st.checkbox('Rebuild Model'):
                            final_model2_regression = retrain_model_function(retrain_data_regression, final_columns2_regression, parameter_to_be_optimized_regression, target_column_regression, best_model_regression)
                            graphs = ['residuals','error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'parameter']
                            selected_graph2 = st.selectbox('Graphs', graphs)
                            plot_model(final_model2_regression, plot = selected_graph2, display_format='streamlit')
                            # rebuild_model_regression(train_set_columns_regression, best_model_regression, data_setup1_regression, retrain_data_regression, target_column_regression, parameter_to_be_optimized_regression)
                if rebuild == 'No':
                    st.write('Thank You, Model Building Done.')

if __name__ == '__main__':
	main()