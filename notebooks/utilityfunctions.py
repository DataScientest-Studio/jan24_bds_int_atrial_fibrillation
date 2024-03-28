# Importing Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
from matplotlib import rc
from cycler import cycler

from sklearn.metrics import (
    confusion_matrix, recall_score, accuracy_score, precision_score,
    f1_score, roc_auc_score, make_scorer
)
from sklearn.model_selection import RandomizedSearchCV

import time

from typing import Optional

from IPython.display import display

# Setting the Styles

def set_custom_palette(palette_name):
    if palette_name == "grayscale":
        # Define your grayscale color palette

    elif palette_name == "colorblind":
        # Define your colorblind-safe color palette
        colorblind_palette = sns.color_palette("colorblind")
        # Set the colorblind palette
        sns.set_palette(colorblind_palette)
    else:
        # Default to grayscale palette if palette_name is not recognized
        grayscale_palette = ['black', '0.8', '0.6', '0.4', '0.2', '0.1', '0.7', '0.3']
        sns.set_palette(grayscale_palette)


def set_styles():
    # Set APA style parameters for matplotlib
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'figure.titlesize': 'large',
        'figure.titleweight': 'bold',
        'figure.subplot.wspace': 0.3,
        'figure.subplot.hspace': 0.3,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.autolimit_mode': 'round_numbers',
        'axes.axisbelow': 'line',
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.labelpad': 5.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.bottom': True,
        'axes.spines.left': True,
        'axes.grid': False,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'errorbar.capsize': 10,
        'savefig.format': 'svg',
        'savefig.bbox': 'tight'
    })
    
    # Define grayscale color cycle
    grayscale_color_cycle = cycler('color', ['black', '0.8', '0.6', '0.4', '0.2', '0.1', '0.7', '0.3'])
    
    # Define line style cycle
    linestyle_cycle = cycler('linestyle', ['-', '--', '-.', ':', '-', '--', '-.', ':'])
    
    # Define line width cycle
    linewidth_cycle = cycler('linewidth', [1.2, 1.2, 1, 0.7, 0.5, 1, 0.8, 0.6])
    
    # Define hatch pattern cycle
    hatch_cycle = cycler('hatch', ['/', '\\', '|', '-', '+', 'x', 'o', 'O'])
    
    # Combine all cycles
    combined_cycle = grayscale_color_cycle + linestyle_cycle + linewidth_cycle + hatch_cycle
    
    # Set the style for seaborn plots
    sns.set_style("white", rc={
        "font.family": "sans-serif",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.prop_cycle": combined_cycle
    })

# Function to perform randomized hyperparameter search
def perform_randomized_search(clf, param_grid, X_train, y_train, X_test, y_test):
    scoring = {'recall': make_scorer(recall_score)}
    print("Starting hyperparameter search...")
    start_total = time.time()
    gcv = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=2)
    start_search = time.time()
    gcv.fit(X_train, y_train)
    end_search = time.time()

    print("Searching...")
    for current_iter in range(1, 11):
        elapsed_time = time.time() - start_search
        remaining_time = (elapsed_time / current_iter) * (10 - current_iter)
        print(f"Iteration {current_iter} completed. Estimated remaining time: {remaining_time:.2f} seconds.", end='\r')
        time.sleep(1)  # Simulate processing time

    end_total = time.time()
    elapsed_total = end_total - start_total
#   print(f"\nHyperparameter search completed. Total elapsed time: {elapsed_total:.2f} seconds.")
    fit_time = end_search - start_search
    cv_time = gcv.refit_time_

    train_predictions = gcv.predict(X_train)
    test_predictions = gcv.predict(X_test)
    train_metrics = get_metrics(y_train, train_predictions)
    test_metrics = get_metrics(y_test, test_predictions)
    return gcv.best_params_, train_metrics, test_metrics, elapsed_total

# Function to calculate evaluation metrics
def get_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    f1 = f1_score(true_labels, predicted_labels, average='binary')
    roc_auc = roc_auc_score(true_labels, predicted_labels)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}

def display_metrics(model_name, train_metrics, test_metrics):
    # Combine train and test metrics into a DataFrame
    metrics_df = pd.DataFrame({'Train': train_metrics, 'Test': test_metrics})
    
    # Add a row for the metric names and set it as the index
    metrics_df = metrics_df.T.rename(index={'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1 Score', 'roc_auc': 'ROC AUC'})
    
    # Format numerical values to display a maximum of 4 decimal points
    metrics_df = metrics_df.round(4)
    
    # Add model name as headline
    print(f"\n{'='*20}\n{model_name}\n{'='*20}\n")
    
    # Display the DataFrame
    display(metrics_df)