import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score,  average_precision_score
from sklearn.model_selection import train_test_split
import json
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import *


to_drop_metrics = ['Project', 'Version', 'id', 'Bugged', 'LOCClass_zscore',
       'OverrideRatio', 'TCC_zscore', 'LCOM', 'NumberOfFields_zscore',
       'NumberOfMethods_zscore', 'FANIN_zscore', 'FANOUT_zscore',
       'DepthOfInheritance_zscore', 'IsAbstract',
       'InterfaceMethodDeclarationCount_zscore', 'NumberOfPublicMethods_bool',
       'PublicFieldCount_zscore']

def train_origin_thresholds_model(df_train,seed, output_directory, run_number, batch_size, hidden_dim_class,layers_num_class, learning_rate_class, early_stop, num_of_epochs):
    X_train_origin = df_train.drop(to_drop_metrics, axis=1).values
    y_train_origin = df_train['Bugged'].values.astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X_train_origin, y_train_origin, test_size=0.1,
                                                      random_state=seed)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_resampled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_resampled, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset_val = TensorDataset(X_val_tensor, y_val_tensor)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    input_dim = X_resampled.shape[1]
    model = DynamicNet(input_dim, hidden_dim_class, layers_num_class, 1)


    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_class)

    # Add early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0

    num_epochs = num_of_epochs

    print("Starting training classification model with origin thresholds...")
    for epoch in range(num_epochs):
        all_loss_class = 0
        for batch in data_loader:
            # General preperation of the data
            X_batch, y_batch = batch
            X_batch = X_batch.to(torch.float32)
            y_batch = y_batch.view(-1, 1).to(torch.float32)

            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss_class += loss.item()

        # Validation loss
        all_loss_val = 0
        for batch in data_loader_val:
            # General preperation of the data
            X_batch_val, y_batch_val = batch
            X_batch_val = X_batch_val.to(torch.float32)
            y_batch_val = y_batch_val.view(-1, 1).to(torch.float32)

            with torch.no_grad():
                val_predictions = model(X_batch_val)
                val_loss = criterion(val_predictions, y_batch_val)
                all_loss_val += val_loss.item()

        val_loss = all_loss_val / len(data_loader_val)
        # Print the loss and validation loss every 100 epochs
        if (epoch + 1) % 10 == 0:
            epoch_loss_class = all_loss_class / len(data_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss_class:.4f}, Validation Loss: {val_loss:.4f}')

            # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), f'{output_directory}/{run_number}/best_model_{run_number}.pth')
        elif early_stop > 0:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stop:
                print(f"Early stopping! No improvement in validation loss. epoch: {epoch + 1}")
                break
        else:
            continue

    final_model = DynamicNet(input_dim, hidden_dim_class, layers_num_class, 1)
    final_model.load_state_dict(torch.load(f'{output_directory}/{run_number}/best_model_{run_number}.pth'))

    return final_model

def test_origin_thresholds_model(output_directory, run_number, df_test, final_model , to_save = True):
    results_origin = {'F1 Score': [], 'AUC-PRC': [], 'AUC-ROC': []}
    unique_projects = set(df_test['Project'].unique())

    for project in unique_projects:
        # Filter train and test dataframes based on the current project
        filtered_test = df_test[df_test['Project'] == project]

        x_test = filtered_test.drop(to_drop_metrics, axis=1).values
        x_test = torch.tensor(x_test)
        x_test = x_test.to(torch.float32)
        y_test = filtered_test['Bugged']
        if (y_test == 0).all():
            continue

        # Evaluate the model on the test set
        with torch.no_grad():
            test_predictions = final_model(x_test)
            predicted_labels = (test_predictions >= 0.5).float()

        # Convert predictions and labels to numpy arrays
        y_pred_np = predicted_labels.numpy().flatten()
        y_true_np = y_test

        try:
            # Calculate metrics
            f1 = f1_score(y_true_np, y_pred_np)
            roc_auc = roc_auc_score(y_true_np, test_predictions.numpy().flatten())
            prc_auc = average_precision_score(y_true_np, test_predictions.numpy().flatten())

            results_origin['F1 Score'].append(f1)
            results_origin['AUC-PRC'].append(prc_auc)
            results_origin['AUC-ROC'].append(roc_auc)


        except:
            continue

    if to_save:
        file_path_results_origin = f'{output_directory}/{run_number}/results_origin_{run_number}.json'
        # Save the dictionary to a JSON file
        with open(file_path_results_origin, 'w') as json_file:
            json.dump(results_origin, json_file)

    means_origin = [np.mean(results_origin[metric]) for metric in list(results_origin.keys())]

    return results_origin, means_origin
