import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_curve
from sklearn.model_selection import train_test_split
import json
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import *
import torch.nn as nn


to_drop = ['Project', 'Version', 'id', 'Bugged', 'lazy_class', 'refused_bequest',
           'large_class', 'god_class', 'multifaceted_abstraction',
           'hub-like_modularization', 'deep_hierarchy', 'swiss_army_knife',
           'unnecessary_abstraction', 'broken_modularization',
           'class_data_should_be_private']

input_dim_generator = 13  # The input dimension for the generator model = total number of metrics
output_dim_generator = 15  # The output dimension for the generator model = total number of thresholds
input_dim_classification = 11  # The input dimension for the classification model = number of smells


def train_thresholds_model(df_train, seed_arg, output_directory, run_number, num_of_epochs, batch_size, layers_num_class,
                layers_num_gen, layers_num_dis, hidden_dim_class, hidden_dim_gen, hidden_dim_dis, learning_rate_gen,
                learning_rate_dis, learning_rate_class, early_stop, loss_weight_generator, plot=True):
    # Prepare the data
    X_train_metrics = df_train.drop(to_drop, axis=1).values
    y_train_metrics = df_train['Bugged'].values.astype(int)
    y_train_metrics = y_train_metrics.reshape((-1, 1))

    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(X_train_metrics, y_train_metrics, test_size=0.1,
                                                                  random_state=seed_arg)

    X_train, y_train = SMOTE(random_state=seed_arg).fit_resample(X_train_nn, y_train_nn)

    # prepare the models

    # Instantiate the models
    generator = DynamicNet(input_dim_generator, hidden_dim_gen, layers_num_gen, output_dim_generator)
    discriminator = DynamicNet(output_dim_generator + 1, hidden_dim_dis, layers_num_dis, 1)
    classifier = DynamicNet(input_dim_classification, hidden_dim_class, layers_num_class, 1)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    criterion_BCE_separate = nn.BCELoss(reduction='none')

    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_gen)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_dis)
    optimizer_C = optim.Adam(classifier.parameters(), lr=learning_rate_class)

    # Training loop
    num_epochs = num_of_epochs
    batch_size = batch_size

    # Prepare the data - Train set
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).view(-1, 1).to(torch.float32)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set the losses lists -  THIS IS FOR GRAPH FOR EPOCHS IF ILL WANT TO
    results_train = {'generator Loss': [], 'classifier Loss': [], 'dis Loss': []}
    results_val = {'score': [], 'loss': []}

    # Prepare the data - Validation set
    X_val_tensor = torch.tensor(X_val_nn)
    X_val_tensor = torch.tensor(X_val_tensor, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_nn, dtype=torch.float32)

    dataset_val = TensorDataset(X_val_tensor, y_val_tensor)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    # Add early stopping
    best_val_score = 0
    early_stopping_counter = 0
    best_epoch = 0

    # Start the training process
    print("Starting model training...")
    for epoch in range(num_epochs):
        all_loss_class = 0
        all_loss_dis = 0
        all_loss_gen = 0
        for batch in data_loader:
            # General preperation of the data
            X_batch, y_batch = batch
            X_batch = X_batch.to(torch.float32)
            real_batch_size = X_batch.size(0)

            # Train the classification
            optimizer_C.zero_grad()
            thresholds = generator(X_batch)
            smells = calc_smells(thresholds.detach(), X_batch)
            output_class = classifier(smells)
            loss_class = criterion(output_class, y_batch)
            loss_class.backward()
            optimizer_C.step()
            all_loss_class += loss_class.item()

            # Get real and fake data
            with torch.no_grad():
                output_class = classifier(smells)
                loss_per_row = criterion_BCE_separate(output_class, y_batch)

            # Real data
            real_labels = torch.ones(real_batch_size, 1)
            real_data = torch.cat((thresholds.detach(), loss_per_row), dim=1)

            # Fake data
            fake_labels = torch.zeros(real_batch_size, 1)
            random_perturbation = (torch.rand_like(
                loss_per_row) - 0.5) * 2 * 0.2  # Creates a random tensor that it's values is in range [-0.2,0.2] (think if make the range not include 0 and very small numbers)
            fake_data = torch.cat((thresholds.detach(), loss_per_row + random_perturbation), dim=1)

            # Train the Discriminator
            optimizer_D.zero_grad()
            real_outputs = discriminator(real_data)
            real_loss = criterion(real_outputs, real_labels)
            fake_outputs = discriminator(fake_data)
            fake_loss = criterion(fake_outputs, fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
            all_loss_dis += d_loss.item()

            # Train Generator
            optimizer_G.zero_grad()
            thresholds = generator(X_batch)
            smells = calc_smells(thresholds.detach(), X_batch)
            output_class = classifier(smells)
            loss_per_row = criterion_BCE_separate(output_class, y_batch)
            real_data = torch.cat((thresholds, loss_per_row), dim=1)

            generator_loss_Discriminator_part = criterion_BCE_separate(discriminator(real_data), real_labels)
            loss_per_row_mul = loss_weight_generator * loss_per_row + 1
            real_generator_loss = generator_loss_Discriminator_part * loss_per_row_mul
            generator_loss = real_generator_loss.mean()
            # Backpropagation
            generator_loss.backward()
            optimizer_G.step()
            all_loss_gen += generator_loss.item()

        # Validation set
        all_loss_val = 0
        all_scores_val = 0
        for batch_val in data_loader_val:
            X_batch_val, y_batch_val = batch_val
            X_batch_val = X_batch_val.to(torch.float32)
            y_batch_val_np = y_batch_val.numpy()
            with torch.no_grad():
                thresholds_val = generator(X_batch_val)
            smells = calc_smells(thresholds_val, X_batch_val)
            with torch.no_grad():
                val_predictions = classifier(smells)
            val_loss = criterion(val_predictions, y_batch_val)
            # calc f1 score for val
            predicted_labels_val = (val_predictions > 0.5).float()  # TO DO think if it is the right way to do so
            predicted_labels_val_np = predicted_labels_val.numpy()
            all_probs_val = val_predictions.squeeze().numpy()
            f1_val = f1_score(y_batch_val_np, predicted_labels_val_np)
            precision, recall, _ = precision_recall_curve(y_batch_val_np, all_probs_val)
            prc_auc_val = auc(recall, precision)
            fpr, tpr, _ = roc_curve(y_batch_val_np, all_probs_val)
            roc_auc_val = auc(fpr, tpr)
            score_val = (f1_val + prc_auc_val + roc_auc_val) / 3

            all_scores_val += score_val
            all_loss_val += val_loss.item()

        # Calc the epoch metrics
        epoch_loss_class = all_loss_class / len(data_loader)
        epoch_loss_dis = all_loss_dis / len(data_loader)
        epoch_loss_gen = all_loss_gen / len(data_loader)
        val_scores = all_scores_val / len(data_loader_val)
        val_losses = all_loss_val / len(data_loader_val)

        # Add the metric to the dicrionares for plottig
        results_train['generator Loss'].append(epoch_loss_gen)
        results_train['classifier Loss'].append(epoch_loss_class)
        results_train['dis Loss'].append(epoch_loss_dis)
        results_val['score'].append(val_scores)
        results_val['loss'].append(val_losses)

        # Print progress every few epochs
        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], generator Loss: {epoch_loss_gen:.4f}, dis Loss: {epoch_loss_dis:.4f}, classifier Loss: {epoch_loss_class:.4f} ; val Loss: {val_losses:.4f}, val score: {val_scores:.4f}")

        # Check for early stopping
        if val_scores > best_val_score:
            best_val_score = val_scores
            early_stopping_counter = 0  # Reset patience if there's an improvement
            torch.save(generator.state_dict(), f'{output_directory}/{run_number}/best_generator_{run_number}.pth')
            torch.save(classifier.state_dict(), f'{output_directory}/{run_number}/best_classifier_{run_number}.pth')
            best_epoch = epoch + 1
        elif early_stop > 0:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stop:
                print(f'Early stopping after {epoch + 1} epochs. The best epoch is {best_epoch}')
                break
        else:
            continue

    # save the json results
    file_path_train = f'{output_directory}/{run_number}/results_train_dict_{run_number}.json'
    file_path_val = f'{output_directory}/{run_number}/results_val_dict_{run_number}.json'
    with open(file_path_train, 'w') as json_file:
        json.dump(results_train, json_file)
    with open(file_path_val, 'w') as json_file:
        json.dump(results_val, json_file)

    if plot:
        plt.style.use('ggplot')
        num_of_epochs = len(results_train['generator Loss']) + 1
        plt.plot(list(range(1, num_of_epochs)), results_train['generator Loss'], label='Train', color='salmon')
        # plt.plot(list(range(1, num_of_epochs)), results_val['score'], label='Val', color='cornflowerblue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Generator')
        plt.savefig(f'{output_directory}/{run_number}/loss_generator_train_{run_number}.png')
        plt.clf()

        plt.plot(list(range(1, num_of_epochs)), results_val['score'], label='Val', color='cornflowerblue')
        plt.xlabel('Epoch')
        plt.ylabel('Score Validation')
        # plt.legend()
        plt.savefig(f'{output_directory}/{run_number}/score_val_{run_number}.png')
        plt.clf()

        plt.plot(list(range(1, num_of_epochs - 1)), results_train['classifier Loss'][1:], label='Train',
                 color='orange')
        plt.plot(list(range(1, num_of_epochs - 1)), results_val['loss'][:-1], label='Validation', color='cornflowerblue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Classifier')
        plt.legend()
        plt.savefig(f'{output_directory}/{run_number}/loss_train_val_{run_number}.png')
        plt.clf()

    # Load the best models
    best_generator = DynamicNet(input_dim_generator, hidden_dim_gen, layers_num_gen, output_dim_generator)
    best_generator.load_state_dict(torch.load(f'{output_directory}/{run_number}/best_generator_{run_number}.pth'))

    best_classifier = DynamicNet(input_dim_classification, hidden_dim_class, layers_num_class, 1)
    best_classifier.load_state_dict(torch.load(f'{output_directory}/{run_number}/best_classifier_{run_number}.pth'))

    return best_generator, best_classifier

def test_thresholds_model(output_directory, run_number, df_test, best_generator, best_classifier, to_save = True):
    unique_projects = set(df_test['Project'].unique())
    best_classifier.eval()  # Set the model to evaluation mode

    # Iterate through each project
    results_model = {'F1 Score': [], 'AUC-PRC': [], 'AUC-ROC': []}
    for project in unique_projects:

        # Filter train and test dataframes based on the current project
        filtered_test = df_test[df_test['Project'] == project]

        x_test = filtered_test.drop(to_drop, axis=1).values
        x_test = torch.tensor(x_test)
        x_test = x_test.to(torch.float32)
        y_test = filtered_test['Bugged']
        if (y_test == 0).all():
            continue

        # Smells of test
        X_test_costume = x_test.clone().detach()

        with torch.no_grad():
            thresholds = best_generator(X_test_costume)
        smells = calc_smells(thresholds, X_test_costume)

        try:

            with torch.no_grad():
                outputs = best_classifier(smells)
                all_probs = outputs.squeeze().cpu().numpy()  # Squeeze removes singleton dimensions

            # Calculate Precision-Recall Curve and AUC
            precision, recall, _ = precision_recall_curve(y_test, all_probs)
            prc_auc = auc(recall, precision)

            # Calculate ROC Curve and AUC
            fpr, tpr, _ = roc_curve(y_test, all_probs)
            roc_auc = auc(fpr, tpr)

            # Calculate F1 Score
            threshold = 0.5  # You can adjust the threshold based on your requirements
            predicted_labels = (all_probs >= threshold).astype(int)
            f1 = f1_score(y_test, predicted_labels)


            results_model['F1 Score'].append(f1)
            results_model['AUC-PRC'].append(prc_auc)
            results_model['AUC-ROC'].append(roc_auc)


        except:
            continue

    if to_save:
        file_path_results_model = f'{output_directory}/{run_number}/results_model_{run_number}.json'
        # Save the dictionary to a JSON file
        with open(file_path_results_model, 'w') as json_file:
            json.dump(results_model, json_file)

    means_metrics = [np.mean(results_model[metric]) for metric in list(results_model.keys())]

    return results_model, means_metrics

