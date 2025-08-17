import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.helpers.config import train_data_path, test_data_path, output_base_path
from src.helpers.DataReader import filereader

def plot_frequency_distribution(df, column, title):
    plt.figure(figsize=(10, 6))
    df[column].value_counts().plot(kind='bar')
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return

def plot_scatter_plot(df, x_column, y_column, title, label_column=None):

    plt.figure(figsize=(10, 6))

    if label_column is not None:
        # Plot each label group separately for distinct colors + legend
        for label, group_data in df.groupby(label_column):
            plt.scatter(
                group_data[x_column],
                group_data[y_column],
                label=label,
                alpha=0.7
            )
    else:
        plt.scatter(df[x_column], df[y_column], alpha=0.7)

    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)

    if label_column is not None:
        plt.legend(title=label_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()

    plt.savefig(output_base_path / f"{title.replace(' ', '_').lower()}.png")
    plt.show()

    return



if __name__ == "__main__":

    train_df = filereader(train_data_path)
    test_df = filereader(test_data_path)

    plot_frequency_distribution(train_df, 'Item_Type', 'Item Type Frequency Distribution in Train Dataset')
    plot_frequency_distribution(train_df, 'Item_Fat_Content', 'Item Fat Content Frequency Distribution in Train Dataset')
    plot_frequency_distribution(test_df, 'Item_Weight', 'Item Type Frequency Distribution in Test Dataset')
    plot_scatter_plot(train_df, x_column='Item_MRP', y_column='Item_Outlet_Sales',
                      title='Item MRP vs Item Outlet Sales in Train Dataset',)