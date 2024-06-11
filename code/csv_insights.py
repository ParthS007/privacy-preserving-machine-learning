import ast
import os
import re

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def save_results_to_csv(
    **kwargs,
):
    file_name = kwargs.get("file_name")
    kwargs.pop("file_name")
    data = {key: [value] for key, value in kwargs.items()}

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)

    # If the file exists, load the existing data and append the new data
    if os.path.isfile(file_name):
        df_existing = pd.read_csv(file_name)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(file_name, index=False)


def extract_tensors_from_string(s):
    numbers = re.findall(r"tensor\((.*?)\)", s)
    return [torch.tensor(float(number)) for number in numbers]


def get_unique_keys(data):
    required_columns = ["algorithm", "noise_multiplier"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(
            f"DataFrame must contain the following columns: {required_columns}"
        )

    # Generate unique keys
    # Replace None with an empty string for noise_multiplier
    noise_multiplier = data["noise_multiplier"].fillna("").astype(str)
    unique_keys = data["algorithm"].astype(str) + " " + noise_multiplier

    # Include k-fold information if available
    if "k_fold_iteration" in data.columns and "k_fold" in data.columns:
        unique_keys += (
            " F:"
            + data["k_fold_iteration"].astype(str)
            + "/"
            + data["k_fold"].astype(str)
        )

    return unique_keys.unique()


def get_color_map(unique_keys):
    if len(unique_keys) == 0 or unique_keys is None:
        raise ValueError("The unique_keys list is empty or None.")

    # Ensure unique_keys is a list
    if not isinstance(unique_keys, (list, np.ndarray)):
        raise TypeError("unique_keys must be a list or numpy array.")

    # Generate colors using a colormap
    colors = cm.rainbow(np.linspace(0, 1, len(unique_keys)))

    return dict(zip(unique_keys, colors))


def get_plot_label(row):
    noise_multiplier = (
        "" if pd.isna(row["noise_multiplier"]) else row["noise_multiplier"]
    )
    plot_label = f"{row['algorithm']} {noise_multiplier}"
    if "k_fold_iteration" in row and "k_fold" in row:
        plot_label += f" F:{row['k_fold_iteration']}/{row['k_fold']}"
    return plot_label


def plot_iterations_train_times_vs_epoch(
    data,
    save_dir,
):
    unique_keys = get_unique_keys(data)
    color_map = get_color_map(unique_keys)

    plt.figure()
    for _, row in data.iterrows():
        epochs = list(range(1, len(row["iteration_train_times"]) + 1))
        plot_label = get_plot_label(row)
        color = color_map.get(plot_label, "black")
        plt.plot(epochs, row["iteration_train_times"], color=color, label=plot_label)

    plt.xlabel("Epoch")
    plt.ylabel("Time - Log(s)")
    plt.yscale("log")
    plt.title(f"{row['model_name']}: Iterations Train times Vs Epoch")
    plt.legend()
    plt.savefig(
        os.path.join(save_dir, f"Iterations_Train_times_Vs_Epoch_{row['model_name']}")
    )
    plt.close()


def plot_training_loss_vs_epoch(data, save_dir):
    unique_keys = get_unique_keys(data)
    color_map = get_color_map(unique_keys)

    plt.figure()
    for _, row in data.iterrows():
        epochs = list(range(1, len(row["training_losses"]) + 1))
        plot_label = get_plot_label(row)
        color = color_map.get(plot_label, "black")
        plt.plot(epochs, row["training_losses"], color=color, label=plot_label)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{row['model_name']}: Training Loss Vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"Training_Loss_Vs_Epoch_{row['model_name']}"))
    plt.close()


def plot_validation_loss_vs_epoch(data, save_dir):
    unique_keys = get_unique_keys(data)
    color_map = get_color_map(unique_keys)

    plt.figure()
    for _, row in data.iterrows():
        epochs = list(range(1, len(row["validation_losses"]) + 1))
        plot_label = get_plot_label(row)
        color = color_map.get(plot_label, "black")
        plt.plot(epochs, row["validation_losses"], color=color, label=plot_label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(f"{row['model_name']}: Validation Loss Vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"Validation_Loss_Vs_Epoch_{row['model_name']}"))
    plt.close()


def plot_epsilon_vs_epoch(data, save_dir):
    unique_keys = get_unique_keys(data)
    color_map = get_color_map(unique_keys)

    plt.figure()
    for _, row in data.iterrows():
        if row["algorithm"] == "Non-DP":
            continue
        # Literal eval for overall_privacy_spent
        row["overall_privacy_spent"] = ast.literal_eval(row["overall_privacy_spent"])
        epochs = list(range(1, len(row["iteration_train_times"]) + 1))
        plot_label = get_plot_label(row)
        color = color_map.get(plot_label, "black")
        plt.plot(epochs, row["overall_privacy_spent"], color=color, label=plot_label)

    plt.xlabel("Epoch")
    plt.ylabel("Overall Privacy Spent (eps_rdp)")
    plt.title(f"{row['model_name']}: Overall Privacy Spent per Iteration")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"Epsilon_Vs_Epoch_{row['model_name']}"))
    plt.close()


def plot_accuracy_vs_epoch(data, save_dir):
    unique_keys = get_unique_keys(data)
    color_map = get_color_map(unique_keys)

    plt.figure()
    for _, row in data.iterrows():
        epochs = list(range(1, len(row["validation_accuracies"]) + 1))
        plot_label = get_plot_label(row)
        color = color_map.get(plot_label, "black")
        plt.plot(epochs, row["validation_accuracies"], color=color, label=plot_label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{row['model_name']}: Validation Accuracy Vs Epoch")
    plt.legend()
    plt.savefig(
        os.path.join(save_dir, f"Validation_Accuracy_Vs_Epoch_{row['model_name']}")
    )
    plt.close()


def plot_metrics_from_csv():
    unet_results_file_names = [
        "results/Non-DP/unet_Duke.csv",
        "results/Opacus-AC/unet_Duke.csv",
        "results/Opacus-RS/unet_Duke.csv",
    ]
    nested_unet_results_file_names = [
        "results/Non-DP/NestedUnet_Duke.csv",
        "results/Opacus-AC/NestedUnet_Duke.csv",
        "results/Opacus-RS/NestedUnet_Duke.csv",
    ]

    # Overall:
    # Plot Iterations Train times Vs Epoch for DP(AC, RS) and Non-DP
    # Plot Training Loss Vs Epoch for DP(AC, RS) and Non-DP
    # Plot Validation Loss Vs Epoch for DP(AC, RS) and Non-DP
    # Plot mean validation loss per epoch for DP(AC, RS) and Non-DP
    # Plot accuracy vs epoch for DP(AC, RS) and Non-DP

    # Opacus results
    # Plot Epsilon vs Epoch for DP(AC, RS)

    # Load the results from the csv files
    unet_results = pd.DataFrame()
    for file_name in unet_results_file_names:
        unet_results = pd.concat(
            [unet_results, pd.read_csv(file_name)], ignore_index=True
        )

    nested_unet_results = pd.DataFrame()
    for file_name in nested_unet_results_file_names:
        nested_unet_results = pd.concat(
            [nested_unet_results, pd.read_csv(file_name)], ignore_index=True
        )

    # Use ast.literal_eval for the following columns:
    # iteration_train_times, training_losses, validation_losses, overall_privacy_spent, validation_accuracies

    unet_results["iteration_train_times"] = unet_results["iteration_train_times"].apply(
        ast.literal_eval
    )
    unet_results["training_losses"] = unet_results["training_losses"].apply(
        ast.literal_eval
    )
    unet_results["validation_losses"] = unet_results["validation_losses"].apply(
        ast.literal_eval
    )

    unet_results["validation_accuracies"] = unet_results["validation_accuracies"].apply(
        ast.literal_eval
    )

    nested_unet_results["iteration_train_times"] = nested_unet_results[
        "iteration_train_times"
    ].apply(ast.literal_eval)
    nested_unet_results["training_losses"] = nested_unet_results[
        "training_losses"
    ].apply(ast.literal_eval)
    nested_unet_results["validation_losses"] = nested_unet_results[
        "validation_losses"
    ].apply(ast.literal_eval)

    nested_unet_results["validation_accuracies"] = nested_unet_results[
        "validation_accuracies"
    ].apply(ast.literal_eval)

    # Save the plots in the results/csv_insights directory
    save_dir = "results/csv_insights"
    os.makedirs(save_dir, exist_ok=True)

    # Plot the metrics for UNet
    plot_iterations_train_times_vs_epoch(unet_results, save_dir)
    plot_training_loss_vs_epoch(unet_results, save_dir)
    plot_validation_loss_vs_epoch(unet_results, save_dir)
    plot_epsilon_vs_epoch(unet_results, save_dir)
    plot_accuracy_vs_epoch(unet_results, save_dir)

    # Plot the metrics for Nested UNet
    plot_iterations_train_times_vs_epoch(nested_unet_results, save_dir)
    plot_training_loss_vs_epoch(nested_unet_results, save_dir)
    plot_validation_loss_vs_epoch(nested_unet_results, save_dir)
    plot_epsilon_vs_epoch(nested_unet_results, save_dir)
    plot_accuracy_vs_epoch(nested_unet_results, save_dir)


if __name__ == "__main__":
    plot_metrics_from_csv()
