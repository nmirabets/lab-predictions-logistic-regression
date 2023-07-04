import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def format_column_names(df: pd.DataFrame, column_renames: Dict[str, str] = {} ) -> pd.DataFrame:
    '''

    This function takes a DataFrame and 
    (1) formats column names to lower case and removes white spaces and
    (2) renames columns according to the input dictionary.
    
    Inputs:
        df: input DataFrame
        column_renames: Dictionary with column renaming
    
    Outputs:
        formatted DataFrame

    '''
    df_formatted = df.copy()
    df_formatted.columns = [col.lower().replace(' ','_') for col in df.columns] # remove white spaces & lower cased
    df_formatted.rename(columns = column_renames, inplace = True) # rename columns according to dictionary
    
    return df_formatted


def clean_column_by_replacing_string(df: pd.DataFrame, column:str, replacements: list) -> pd.DataFrame:
    '''

    This function takes a Dataframe and replaces the strings 
    in the input replacements to the specified column.
    
    Inputs:
        df: input DataFrame
        column: column to apply transformations
        replacements: list of lists with replacements 
            [[old_value1, new_value1],[old_value2, new_value2],...]
        
    Output:
        pandas DataFrame with the clean column

    '''
    clean_df = df.copy()
    
    for item in replacements:
        clean_df[column] = clean_df[column].str.replace(item[0],item[1]) # replace items in column
        
    return clean_df


def reassign_column_data_type(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    '''

    This function takes a DataFrame and reassigns data types as specified in the columns parameter.
    
    Input: 
        df: pandas DataFrame
        columns: Dictionary with column and data type assignment
    
    Output:
        Dataframe with data type reassign columns

    '''
    reassigned_df = df.copy()

    for key, value in columns.items():
        reassigned_df[key] = reassigned_df[key].astype(value)

    return reassigned_df


def remove_duplicate_and_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    '''

    This function removes duplicate rows and rows with all the columns empty.
    
    Input:
        df: input DataFrame
    
    Output:
        df: output DataFrame

    '''
    clean_df = df.copy()
    
    clean_df.drop_duplicates(inplace = True) # drop all duplicate rows
    clean_df.dropna(inplace = True) # drop all empty rows
    
    return clean_df


def plot_numeric_columns(df, plot_type='histogram'):
    '''

    This function generates plots for all numerical columns of the input DataFrame.

    Inputs:
        df: input DataFrame
        plot_type: type of plot, 'histogram' or 'boxplot'

    Output:
        A plot for each column

    '''
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    num_plots = len(numerical_columns)
    num_rows = int(np.ceil(num_plots / 2))
    num_cols = 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4)

    for i, column in enumerate(numerical_columns):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]

        if plot_type == 'histogram':
            ax.hist(df[column], bins=20, edgecolor='black')
            ax.set_ylabel('Frequency')
        elif plot_type == 'boxplot':
            ax.boxplot(df[column], vert = False)

        ax.set_xlabel(column)
        ax.set_title(f'{plot_type.capitalize()} of {column}')

    # Remove empty subplots
    if num_plots < num_rows * num_cols:
        if num_rows > 1:
            for i in range(num_plots, num_rows * num_cols):
                fig.delaxes(axes[i // num_cols, i % num_cols])
        else:
            for i in range(num_plots, num_rows * num_cols):
                fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def analyze_skewness(df: pd.DataFrame, interval: tuple=(-2, 2)):
    '''

    This function generates a table to analyze the skewness of the numerical columns of a DataFrame.

    Inputs:
        df: input DataFrame
        interval: (min,max) skewness interval

    Output:
        DataFrame with 3 columns:
        (1) column_name
        (2) skew value
        (3) Boolean for skew value out of interval

    '''
    numerical_cols = df.select_dtypes(include='number').columns
    skew_values = []
    for col in numerical_cols:
        skew = df[col].skew()
        is_outside_interval = not (interval[0] <= skew <= interval[1])
        skew_values.append((col, skew, is_outside_interval))

    result_df = pd.DataFrame(skew_values, columns=['Column', 'Skew', 'Outside Interval'])
    return result_df


def select_features_for_linear_models_based_on_correlation(df: pd.DataFrame, y: str, threshold: float=0.75) -> list:
    '''

    This function takes a DataFrame and returns a list of the columns that
    have a correlation index with y column greater than the threshold.

    Input
        df: pd.DataFrame
        y: column to be predicted

    Output
        list: list of strings

    '''
    correlation_matrix = df.corr(numeric_only=True)#.reset_index()

    list_of_selected_columns = list(correlation_matrix[y].loc[abs(correlation_matrix[y])>=threshold].index)

    return list_of_selected_columns


def plot_corr_matrix(df: pd.DataFrame):
    '''

    This function takes a DataFrame and generates a correlation matrix between its numerical columns.

    Input
        df: pd.DataFrame

    Output
        correlation matrix plot

    '''
    # Calculate the correlation matrix
    correlation_matrix = df.corr(numeric_only = True)

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the correlation matrix as a heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='magma', linewidths=0.5, linecolor='lightgray',
                cbar=True, square=True, xticklabels=True, yticklabels=True, ax=ax)

    # Customize the plot
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()


def remove_outliers(df: pd.DataFrame, columns: list, lower_percentile: float=25, upper_percentile: float =75) -> pd.DataFrame:
    '''

    This function removes the outliers of specified columns of a 
    dataset that lie out of the limits provided as inputs.
    
    Input:
    df: input DataFrame
    columns: list of columns to remove outliers
    lower_percentile: lower limit percentile to remove outliers
    upper_percentile: upper limit percentile to remove outliers
    
    Output:
    DataFrame with removed outliers
    
    '''
    filtered_data = df.copy()
    
    for column in columns:
        lower_limit = np.percentile(filtered_data[column], lower_percentile)
        upper_limit = np.percentile(filtered_data[column], upper_percentile)
        iqr = upper_limit - lower_limit
        lower_whisker = lower_limit - 1.5 * iqr
        upper_whisker = upper_limit + 1.5 * iqr

        filtered_data = filtered_data[(filtered_data[column] > lower_whisker) & (filtered_data[column] < upper_whisker)]

    return filtered_data

def save_pickle_file(path: str, file_name: str, file_to_save):
    '''

    This function saves pickle files such as scalers, transformers, encoders or models 
    to later retrieve them using the read_pickle_file function.
    
    Input:
        path: path to save the file. If the directory doesn't exist, it will create one. i.e 'encoders/'
        file_name: name of the pickle file. Must have .pkl extension. i.e encoder.pkl
        file_to_save: the .pkl file to save
    
    Output:
        It will save the file. If a directory is created it will notify via print.
    
    '''
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist
      os.makedirs(path)
      print("The new directory is created!")

    with open(path + file_name, "wb") as file:
        pickle.dump(file_to_save, file)

def read_pickle_file(path: str):
    '''

    This function reads pickle files such as scalers, transformers, encoders or models 
    from a previsouly saved file using save_pickle_file.
    
    Input:
        path: the file's location with .pkl extension. i.e encoders/encoder.pkl
    
    Output:
        file: the file read at the directory
    
    '''
    import pickle

    with open(path, "rb") as file:
        file_read = pickle.load(file)

    return file_read

def create_results_table(X_train: pd.DataFrame, 
                        X_test: pd.DataFrame, 
                        y_train: pd.Series, 
                        y_test: pd.Series, 
                        y_train_pred: pd.Series, 
                        y_test_pred: pd.Series) -> pd.DataFrame:
    '''

    This function creates a DataFrame with the results, and  its absolute and relative errors of a model prediction.
    
    Input:
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        y_train_pred: pd.Series
        y_test_pred: pd.Series
    
    Output:
        Dataframe with results and errors
    
    '''

    # Create a table with the results - Real vs. Predicted
    results = {"Set": ["Train"]*X_train.shape[0] + 
               ["Test"]*X_test.shape[0], 
               "Real": list(y_train) + list(y_test),
               "Predicted": list(y_train_pred) + list(y_test_pred)}

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Add absolute and relative errors to DataFrame
    results_df['Absolute Error'] = (results_df['Real'] - results_df['Predicted']).round(2)
    results_df['Relative Error'] = (((results_df['Real'] - results_df['Predicted']) / results_df['Predicted'])*100).round()

    return results_df

def plot_linear_regression_errors(results: pd.DataFrame):
    '''

    This function generates two plots:
    (1) Predicted vs. Real scatter plot including train and test data + line fit
    (2) Histogram of the absolute error residuals of train and test data

    Input:
        results: DataFrame output of function create_results_table
    
    Output:
        Predicted vs. Real and Histogram of Residuals plots
    
    '''
    
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    
    sns.scatterplot(results, x="Real", y="Predicted", hue="Set", ax=ax[0])
    sns.lineplot(results, x="Real", y="Real", color="black", ax=ax[0])
    ax[0].set_title("Predicted vs real")
    sns.histplot(results, x="abs_error", bins=50, hue="Set", ax=ax[1])
    ax[1].set_title("Histogram of residuals")
    
    plt.tight_layout()
    plt.show()


def error_metrics_report(y_real_train: list, y_real_test: list, y_pred_train: list, y_pred_test: list) -> pd.DataFrame:
    '''
    
    This function calculates the MAE, MSE, RMSE and R2 error metrics 
    and returns a df with the results.
    
    Inputs:
    y_real_train: real y values used as training data
    y_real_test: real y values used as test data
    y_pred_train: predicted y values obtained from training data
    y_pred_test: predicted y values obtained from test data
    
    Outputs:
    DataFrame with the results for each error metric 
    for real and predicted data.
    
    '''
    # Mean absolute error
    MAE_train = mean_absolute_error(y_real_train, y_pred_train)
    MAE_test  = mean_absolute_error(y_real_test, y_pred_test)

    # Mean squared error
    MSE_train = mean_squared_error(y_real_train, y_pred_train)
    MSE_test  = mean_squared_error(y_real_test, y_pred_test)

    # Root mean squared error
    RMSE_train = mean_squared_error(y_real_train, y_pred_train, squared = False)
    RMSE_test  = mean_squared_error(y_real_test, y_pred_test, squared = False)

    # R2 error
    R2_train = r2_score(y_real_train, y_pred_train)
    R2_test  = r2_score(y_real_test, y_pred_test)

    results = {"Metric": ['MAE','MSE','RMSE','R2'], 
               "Train": [MAE_train, MSE_train, RMSE_train, R2_train],
               "Test":  [MAE_test, MSE_test, RMSE_test, R2_test]}

    results_df = pd.DataFrame(results).round(2)

    return results_df


