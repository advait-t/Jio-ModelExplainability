from pickle import NONE
from classification_modelling_pipeline import *
from explainability_pipeline import *
import streamlit as st
# import SessionState
from all_function import *

def eda():
    st.subheader('Exploratory Data Analysis')
    file = st.file_uploader('Upload Dataset', type = ['csv', 'txt'])
    if file is not None:
        df = pd.read_csv(file)
        all_columns = df.columns.to_list()
        st.dataframe(df.head())

        if st.checkbox("Show Shape"):
            st.write(df.shape)

        if st.checkbox("Show Columns"):
            st.write(all_columns)

        if st.checkbox("Summary"):
            st.write(df.describe())

        if st.checkbox("Show Selected Columns"):
            selected_columns = st.multiselect("Select Columns",all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        if st.checkbox("Show Value Counts"):
            selected_columns1 = st.multiselect("Select Column",all_columns)
            #target = st.selectbox('Target Variable', all_columns)
            new_df1 = df[selected_columns1]
            for i in selected_columns1:
                st.write(new_df1[i].value_counts())

        if st.checkbox("Correlation Plot(Matplotlib)"):
            plt.matshow(df.corr())
            st.pyplot()

        if st.checkbox("Correlation Plot(Seaborn)"):
            fig = st.write(sns.heatmap(df.corr(),annot=True))
            st.pyplot(fig)

        if st.checkbox("Pie Plot"):
            all_columns = df.columns.to_list()
            column_to_plot = st.selectbox("Select 1 Column",all_columns)
            pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_plot)
            st.pyplot() 

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
# @st.cache(suppress_st_warning=True)
def classifiction_model_function(data, final_columns1, parameter_to_be_optimized, target):
    final_model1, data_setup1, best_model = setting_up_automl(data, final_columns1, parameter_to_be_optimized, target)
    X_train, y_train, X_test, y_test = save_feature_importance(final_model1)
    retrain_data = data_set(X_train, y_train, X_test, y_test)
    target_column = y_train.name
    train_set_columns = list(retrain_data.columns)
    SE, predictions = save_shap_file(final_model1, X_test)
    saving_top_features(SE, predictions, target_column)
    return train_set_columns, best_model, data_setup1, retrain_data, target_column

def select_data():
    st.subheader('Automation of Model Building and Explainability of Models')
    file = st.file_uploader('Upload Dataset', type = ['csv', 'txt'])
    if file is not None:
        data = pd.read_csv(file)
        return data

# @st.cache(suppress_st_warning=True)
# @st.experimental_memo
def model_building(data):
#     st.subheader('Automation of Model Building and Explainability of Models')
#     file = st.file_uploader('Upload Dataset', type = ['csv', 'txt'])
#     if file is not None:
#     data = pd.read_csv(file)
    columns = list(data.columns)


    target = st.selectbox('Select your target variable', columns)
    if target is not None:
        container1 = st.beta_container()
        all1 = st.checkbox("Select all")
        if all1:
            final_columns1 = container1.multiselect("Select columns for model building:", columns, columns)
        else:
            final_columns1 =  container1.multiselect("Select columns for model building:", columns)


    parameter_to_be_optimized = st.selectbox('Select a parameter you want to optimize', ['Accuracy','AUC','Recall' ,'Prec.', 'F1', 'Kappa', 'MCC'])

    # if st.button('Build Model'):
    #     train_set_columns, best_model, data_setup1, retrain_data, target_column = dummy_fun(data, final_columns1, parameter_to_be_optimized, target)
        
        # rebuild = st.selectbox('Do you want to remove any features and rebuild the model?', ['Yes', 'No'])

        # return train_set_columns, best_model, data_setup1, retrain_data, target_column, parameter_to_be_optimized
    return data, final_columns1, parameter_to_be_optimized, target
        # if rebuild == 'Yes':
        #     container2 = st.beta_container()
        #     all2 = st.checkbox("Select all columns")
        #     if all2:
        #         st.cache()
        #         final_columns2 = container2.multiselect("Select columns for final model building:", train_set_columns, train_set_columns)
        #         if st.button('Rebuild Model'):
        #             final_model2, data_setup2 = retrain_model(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
        #             X_train_new, y_train_new, X_test_new, y_test_new = save_feature_importance(final_model2)
        #             SE, predictions = save_shap_file(final_model2, X_test_new)
        #             saving_top_features(SE, predictions, target_column) 
        #     else:
        #         final_columns2 =  container2.multiselect("Select columns for final model building:", train_set_columns)
                
        #         if st.button('Rebuild Model'):
        #             final_model2, data_setup2 = retrain_model(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
        #             X_train_new, y_train_new, X_test_new, y_test_new = save_feature_importance(final_model2)
        #             SE, predictions = save_shap_file(final_model2, X_test_new)
        #             saving_top_features(SE, predictions, target_column) 


# def rebuild_model(train_set_columns, best_model, data_setup1, retrain_data, target_column, parameter_to_be_optimized):
#     rebuild = st.selectbox('Do you want to remove any features and rebuild the model?', ['Yes', 'No'])
#     if rebuild == 'Yes':
#         container2 = st.beta_container()
#         all2 = st.checkbox("Select all columns")
#         if all2:
#             st.cache()
#             final_columns2 = container2.multiselect("Select columns for final model building:", train_set_columns, train_set_columns)
#             if st.button('Rebuild Model'):
#                 final_model2, data_setup2 = retrain_model(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
#                 X_train_new, y_train_new, X_test_new, y_test_new = save_feature_importance(final_model2)
#                 SE, predictions = save_shap_file(final_model2, X_test_new)
#                 saving_top_features(SE, predictions, target_column) 
#         else:
#             final_columns2 =  container2.multiselect("Select columns for final model building:", train_set_columns)
            
#             if st.button('Rebuild Model'):
#                 final_model2, data_setup2 = retrain_model(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
#                 X_train_new, y_train_new, X_test_new, y_test_new = save_feature_importance(final_model2)
#                 SE, predictions = save_shap_file(final_model2, X_test_new)
#                 saving_top_features(SE, predictions, target_column) 


def main():
    # st.cache()
    st.title('Automated EDA, Model Building and Explainability')
    side_bar = ['EDA','Visualisation','Model Building']
    choice = st.sidebar.selectbox('Select your task',side_bar)
    if choice == 'EDA':
        eda()
    elif choice == 'Visualisation':
        plots()
    else:
        file = select_data()
        if file is not None:
            data, final_columns1, parameter_to_be_optimized, target = model_building(file)
            build = st.checkbox('Build Model')

            if build:
                # st.cache()
                train_set_columns, best_model, data_setup1, retrain_data, target_column = classifiction_model_function(data, final_columns1, parameter_to_be_optimized, target)
                graphs = ['auc','threshold','pr','confusion_matrix','error','class_report','feature','feature_all']
                selected_graph = st.selectbox('Graphs', graphs)
                # temp_dir = tempfile.TemporaryDirectory()
                # cv2.imwrite((str(temp_dir.name)+'/test.png'), np.array(image))
                # f = np.fromfile((str(temp_dir.name)+'/test.png'))
                # plt.figure(figsize = (18,9))
                plott = plot_model(best_model, plot = selected_graph, display_format='streamlit')
                # st.image(plott)
                

            if build:
                rebuild = st.selectbox('Do you want to remove any features and rebuild the model?', ['Yes', 'No'])
                if rebuild == 'Yes':
                    container2 = st.beta_container()
                    all2 = st.checkbox("Select all columns")
                    if all2:
                        final_columns2 = container2.multiselect("Select columns for final model building:", train_set_columns, train_set_columns)
                        if st.checkbox('Rebuild Model'):
                            final_model2, data_setup2 = retrain_model(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
                            X_train_new, y_train_new, X_test_new, y_test_new = save_feature_importance(final_model2)
                            SE, predictions = save_shap_file(final_model2, X_test_new)
                            saving_top_features(SE, predictions, target_column)
                    else:
                        final_columns2 =  container2.multiselect("Select columns for final model building:", train_set_columns)
                        
                        if st.checkbox('Rebuild Model'):
                            final_model2, data_setup2 = retrain_model(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
                            X_train_new, y_train_new, X_test_new, y_test_new = save_feature_importance(final_model2)
                            SE, predictions = save_shap_file(final_model2, X_test_new)
                            saving_top_features(SE, predictions, target_column) 
                            # rebuild_model(train_set_columns, best_model, data_setup1, retrain_data, target_column, parameter_to_be_optimized)
                if rebuild == 'No':
                    st.write('Thank You, Model Building Done.')


if __name__ == '__main__':
	main()