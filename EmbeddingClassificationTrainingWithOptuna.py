"""Train Optuna-tuned MLP classifiers on precomputed MRI embeddings."""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

random_seed = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.set_float32_matmul_precision('high')
print(f'PyTorch device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

def setup_logging(log_file_path):
    """Configure the training logger to write to a file and the console."""
    # Create logger
    logger = logging.getLogger('classification_optuna')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_data(df, tag, split_col_name, data_folder, binary_flag=False):
    """Load cached splits or build them from embedding files referenced by the CSV."""
    file_names = ['train_features.npy', 'train_labels.npy', 'val_features.npy', 'val_labels.npy', 'test_features.npy', 'test_labels.npy']
    recalculate_flag = False
    for file_name in file_names:
        if os.path.exists(os.path.join(data_folder, file_name)):
            print(os.path.join(data_folder, file_name))
            print(f"load data from {os.path.join(data_folder, file_name)}")
            data = np.load(os.path.join(data_folder, file_name))
            data = np.squeeze(data)
            
            if file_name == 'train_features.npy':
                train_feature_array = data
            elif file_name == 'train_labels.npy':
                train_label_array = data
            elif file_name == 'val_features.npy':
                val_feature_array = data
            elif file_name == 'val_labels.npy':
                val_label_array = data
            elif file_name == 'test_features.npy':
                test_feature_array = data
            elif file_name == 'test_labels.npy':
                test_label_array = data
        else:
            recalculate_flag = True
            break
    
    if recalculate_flag:
        # Rebuild all splits when even one cached array is missing.
        train_feature_array, train_label_array, val_feature_array, val_label_array, test_feature_array, test_label_array = [], [], [], [], [], []
        path_index = df.columns.get_loc('flywheel_info.nifti_path')
        split_index = df.columns.get_loc(split_col_name)
        tag_index = df.columns.get_loc(tag)
        records = []
        for row in df.itertuples(index=False, name=None):
            emb_path = row[path_index].replace('.nii.gz', '_GE_emb_11220225.npy')
            emb_path = emb_path.replace('./flywheel_data/', './')
            records.append((emb_path, row[split_index], row[tag_index]))

        def load_embedding(record):
            emb_path, split, label = record

            if not os.path.exists(emb_path):
                return "missing", split, label, emb_path

            emb = np.load(emb_path)

            if np.isnan(emb).any():
                return "nan", split, label, emb_path

            return "ok", split, label, emb_path, emb

        missing_counts = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        nan_counts = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        loaded_counts = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        worker_count = min(4, max(1, os.cpu_count() or 1))
        
        with ThreadPoolExecutor(max_workers=worker_count) as executor:

            loaded_records = executor.map(
                load_embedding,
                records
            )

            for result in tqdm(
                loaded_records,
                desc="Loading data",
                total=len(records)
            ):

                status = result[0]
                split = result[1]

                if status == "missing":
                    missing_counts[split] += 1
                    continue

                if status == "nan":
                    nan_counts[split] += 1
                    continue

                _, split, label, emb_path, emb = result

                loaded_counts[split] += 1

                if split == "train":
                    train_feature_array.append(emb)
                    train_label_array.append(label)

                elif split == "val":
                    val_feature_array.append(emb)
                    val_label_array.append(label)

                elif split == "test":
                    test_feature_array.append(emb)
                    test_label_array.append(label)
                    
        print("\n--- EMBEDDING AVAILABILITY ---")

        for split in ["train", "val", "test"]:

            total = (
                loaded_counts[split]
                + missing_counts[split]
                + nan_counts[split]
            )

            print(f"\n{split.upper()}")
            print(f"  Expected: {total}")
            print(f"  Loaded:   {loaded_counts[split]}")
            print(f"  Missing:  {missing_counts[split]}")
            print(f"  NaN:      {nan_counts[split]}")
            
        train_feature_array = np.squeeze(np.array(train_feature_array))
        val_feature_array = np.squeeze(np.array(val_feature_array))
        test_feature_array = np.squeeze(np.array(test_feature_array))

        train_label_array = np.squeeze(np.array(train_label_array))
        val_label_array = np.squeeze(np.array(val_label_array))
        test_label_array = np.squeeze(np.array(test_label_array))

        np.save(os.path.join(data_folder, 'train_features.npy'), train_feature_array)
        np.save(os.path.join(data_folder, 'train_labels.npy'), train_label_array)
        np.save(os.path.join(data_folder, 'val_features.npy'), val_feature_array)
        np.save(os.path.join(data_folder, 'val_labels.npy'), val_label_array)
        np.save(os.path.join(data_folder, 'test_features.npy'), test_feature_array)
        np.save(os.path.join(data_folder, 'test_labels.npy'), test_label_array)
    
    label_encoder = LabelEncoder()
    train_label_array = label_encoder.fit_transform(train_label_array)
    val_label_array = label_encoder.transform(val_label_array)
    test_label_array = label_encoder.transform(test_label_array)
    print(f"Using split column: {split_col_name}")
    print(f"Train: {len(train_label_array)}")
    print(f"Val:   {len(val_label_array)}")
    print(f"Test:  {len(test_label_array)}")
    fold_splits = [(train_feature_array, train_label_array, val_feature_array, val_label_array, test_feature_array, test_label_array)]
    return fold_splits, label_encoder


def convert_optuna_params_to_mlp_params(optuna_params):
    """Convert a trial's layer parameters into the format used by TorchMLP."""
    n_layers = optuna_params['n_layers']
    hidden_layer_sizes = []
    for i in range(n_layers):
        n_neurons = optuna_params[f'n_neurons_l{i}']
        hidden_layer_sizes.append(n_neurons)
    hidden_layer_sizes = tuple(hidden_layer_sizes)
    
    return {
        'hidden_layer_sizes': hidden_layer_sizes,
        'learning_rate_init': optuna_params['learning_rate_init'],
        'alpha': optuna_params['alpha'],
        'max_iter': optuna_params['max_iter']
    }


class TorchMLP(nn.Module):
    """Feed-forward classifier whose hidden-layer sizes are selected by Optuna."""

    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        layers = []
        previous_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.extend([nn.Linear(previous_size, hidden_size), nn.ReLU()])
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)

    def predict_proba(self, features):
        self.eval()
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features, dtype=torch.float32)
        features = features.to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(self(features), dim=1)
        return probabilities.cpu().numpy()

    def predict(self, features):
        return np.argmax(self.predict_proba(features), axis=1)


def train_and_val(train_feature_array, train_label_array, val_feature_array, val_label_array, model_params, average_method='weighted'):
    """Train one model and return its validation metrics, model, and training time."""
    torch.manual_seed(random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(random_seed)

    train_features = torch.as_tensor(train_feature_array, dtype=torch.float32)
    train_labels = torch.as_tensor(train_label_array, dtype=torch.long)
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(1024, len(train_dataset)),
        shuffle=True,
        pin_memory=device.type == 'cuda'
    )

    model = TorchMLP(
        input_size=train_features.shape[1],
        hidden_layer_sizes=model_params['hidden_layer_sizes'],
        output_size=len(np.unique(train_label_array))
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_params['learning_rate_init'],
        weight_decay=model_params['alpha']
    )
    criterion = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    model.train()
    for _ in range(model_params['max_iter']):
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    train_time = end_time - start_time

    val_features = torch.as_tensor(val_feature_array, dtype=torch.float32).to(device)
    val_pred = model.predict(val_features)
    val_pred_proba = model.predict_proba(val_features)
    
    val_accuracy = accuracy_score(val_label_array, val_pred)
    val_precision = precision_score(val_label_array, val_pred, average=average_method, zero_division=0)
    val_recall = recall_score(val_label_array, val_pred, average=average_method)
    val_f1 = f1_score(val_label_array, val_pred, average=average_method)
    
    # Calculate ROC-AUC (handle multi-class case)
    if len(np.unique(val_label_array)) > 2:
        val_roc_auc = roc_auc_score(val_label_array, val_pred_proba, multi_class='ovr', average=average_method)
        val_pr_auc = average_precision_score(val_label_array, val_pred_proba, average=average_method)
    else:
        val_roc_auc = roc_auc_score(val_label_array, val_pred_proba[:, 1])
        val_pr_auc = average_precision_score(val_label_array, val_pred_proba[:, 1], average=average_method)

    return val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc, model, train_time


def test(model, feature_array, label_array, average_method='weighted'):
    """Evaluate a trained model and return metrics, predictions, and elapsed time."""
    start_time = time.time()
    pred = model.predict(feature_array)
    end_time = time.time()
    test_time = end_time - start_time
    pred_proba = model.predict_proba(feature_array)
    
    accuracy = accuracy_score(label_array, pred)
    precision = precision_score(label_array, pred, average=average_method, zero_division=0)
    recall = recall_score(label_array, pred, average=average_method)
    f1 = f1_score(label_array, pred, average=average_method)
    
    # Calculate ROC-AUC (handle multi-class case)
    if len(np.unique(label_array)) > 2:
        roc_auc = roc_auc_score(label_array, pred_proba, multi_class='ovr', average=average_method)
        pr_auc = average_precision_score(label_array, pred_proba, average=average_method)
    else:
        roc_auc = roc_auc_score(label_array, pred_proba[:, 1], average=average_method)
        pr_auc = average_precision_score(label_array, pred_proba[:, 1], average=average_method)
    
    return accuracy, precision, recall, f1, roc_auc, pr_auc, pred, pred_proba, test_time


def optuna_objective(trial, fold_splits, model_selection_criteria='f1_roc_auc', average_method='weighted'):
    """Train and score a trial across the supplied train/validation splits."""
    
    # Suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layer_sizes = []
    for i in range(n_layers):
        n_neurons = trial.suggest_int(f'n_neurons_l{i}', 10, 500)
        hidden_layer_sizes.append(n_neurons)
    hidden_layer_sizes = tuple(hidden_layer_sizes)
    
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-5, 1e-1, log=True)
    alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)
    max_iter = trial.suggest_int('max_iter', 20, 200)
    
    model_params = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'learning_rate_init': learning_rate_init,
        'alpha': alpha,
        'max_iter': max_iter
    }
    
    # Evaluate on all folds
    fold_scores = []
    for train_features, train_labels, val_features, val_labels, test_features, test_labels in fold_splits:
        try:
            val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc, model, train_time = train_and_val(
                train_features, train_labels, val_features, val_labels, model_params, average_method=average_method)
            
            if model_selection_criteria == 'accuracy':
                fold_scores.append(val_accuracy)
            elif model_selection_criteria == 'precision':
                fold_scores.append(val_precision)
            elif model_selection_criteria == 'recall':
                fold_scores.append(val_recall)
            elif model_selection_criteria == 'f1':
                fold_scores.append(val_f1)
            elif model_selection_criteria == 'roc_auc':
                fold_scores.append(val_roc_auc)
            elif model_selection_criteria == 'f1_roc_auc':
                # Combined score: average of F1 and ROC-AUC
                combined_score = (val_f1 + val_roc_auc) / 2
                fold_scores.append(combined_score)
            elif model_selection_criteria == 'f1_pr_auc':
                combined_score = (val_f1 + val_pr_auc) / 2
                fold_scores.append(combined_score)
            else:
                raise ValueError(f"Invalid model selection criteria: {model_selection_criteria}")
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('-inf')  # Failed trials must not win a maximize study
    
    mean_score = np.mean(fold_scores)
    
    # Report intermediate results for pruning
    trial.report(mean_score, 0)
    
    # Check if trial should be pruned
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return mean_score


def cross_validation_optuna(fold_splits, key, work_log, ratio=1.0, model_name=None, n_trials=100, model_selection_criteria='f1_roc_auc', average_method='weighted', n_folds=1):
    """Perform cross-validation with Optuna hyperparameter optimization"""
    # Setup logging
    logger = setup_logging(work_log)
    logger.info(f"The label is {key}")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_seed),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    logger.info(f"Using PyTorch device: {device}; max_iter search range: 20-200 epochs")
    
    # Optimize hyperparameters
    start_time = time.time()
    study.optimize(
        lambda trial: optuna_objective(trial, fold_splits, model_selection_criteria=model_selection_criteria, average_method=average_method),
        n_trials=n_trials,
        show_progress_bar=True
    )
    end_time = time.time()
    optimize_time = end_time - start_time
    logger.info(f"Optuna optimization time: {end_time - start_time:.2f} seconds")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best {model_selection_criteria}: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    # Train final models with best parameters on each fold
    fold_results = []
    all_predictions = []
    all_pred_proba = []
    all_labels = []
    all_train_times = []
    all_test_times = []
    
    for fold_idx, (train_features, train_labels, val_features, val_labels, test_features, test_labels) in enumerate(fold_splits):
        logger.info(f"=== Fold {fold_idx + 1}/{n_folds} ===")
        
        # Train with best parameters
        best_mlp_params = convert_optuna_params_to_mlp_params(study.best_params)
        val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc, best_model, train_time = train_and_val(
            train_features, train_labels, val_features, val_labels, best_mlp_params, average_method=average_method)
        
        # Test on the held-out test set
        test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc, test_pred, test_pred_proba, test_time = test(best_model, test_features, test_labels, average_method=average_method)
        
        # Store results
        fold_result = {
            'fold': fold_idx + 1,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'roc_auc': test_roc_auc,
            'pr_auc': test_pr_auc
        }
        fold_results.append(fold_result)
        
        # Store predictions for overall analysis
        all_predictions.extend(test_pred)
        all_pred_proba.extend(test_pred_proba)
        all_labels.extend(test_labels)

        all_train_times.append(train_time)
        all_test_times.append(test_time)

        logger.info(f"Fold {fold_idx + 1} Results:")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Precision: {test_precision:.4f}")
        logger.info(f"  Recall: {test_recall:.4f}")
        logger.info(f"  F1: {test_f1:.4f}")
        logger.info(f"  ROC-AUC: {test_roc_auc:.4f}")
        logger.info(f"  PR-AUC: {test_pr_auc:.4f}")
        logger.info(f"  Train time: {train_time:.2f} seconds")
        logger.info(f"  Test time: {test_time:.2f} seconds")
    
    # Calculate average metrics across all folds
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_precision = np.mean([r['precision'] for r in fold_results])
    avg_recall = np.mean([r['recall'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    avg_roc_auc = np.mean([r['roc_auc'] for r in fold_results])
    avg_pr_auc = np.mean([r['pr_auc'] for r in fold_results])
    avg_train_time = np.mean(np.array(all_train_times))
    avg_test_time = np.mean(np.array(all_test_times))

    # Calculate standard deviations
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    std_precision = np.std([r['precision'] for r in fold_results])
    std_recall = np.std([r['recall'] for r in fold_results])
    std_f1 = np.std([r['f1'] for r in fold_results])
    std_roc_auc = np.std([r['roc_auc'] for r in fold_results])
    std_pr_auc = np.std([r['pr_auc'] for r in fold_results])
    std_train_time = np.std(np.array(all_train_times))
    std_test_time = np.std(np.array(all_test_times))
    
    # Calculate overall metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    overall_precision = precision_score(all_labels, all_predictions, average=average_method, zero_division=0)
    overall_recall = recall_score(all_labels, all_predictions, average=average_method)
    overall_f1 = f1_score(all_labels, all_predictions, average=average_method)

    # Calculate overall ROC-AUC using collected probabilities
    all_pred_proba = np.array(all_pred_proba)
    if len(np.unique(all_labels)) > 2:
        overall_roc_auc = roc_auc_score(all_labels, all_pred_proba, multi_class='ovr', average=average_method)
        overall_pr_auc = average_precision_score(all_labels, all_pred_proba, average=average_method)
    else:
        overall_roc_auc = roc_auc_score(all_labels, all_pred_proba[:, 1], average=average_method)
        overall_pr_auc = average_precision_score(all_labels, all_pred_proba[:, 1], average=average_method)
    
    # Log detailed results
    logger.info("=== Optuna Optimization Results ===")
    logger.info(f"Optuna optimization time: {optimize_time:.2f} seconds")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best {model_selection_criteria}: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Averaged method: {average_method}")
    logger.info(f"Model selection criteria: {model_selection_criteria}")

    logger.info(f"=== Cross-Validation Results ({n_folds} folds) ===")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    logger.info(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    logger.info(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
    logger.info(f"Average F1: {avg_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"Average ROC-AUC: {avg_roc_auc:.4f} ± {std_roc_auc:.4f}")
    logger.info(f"Average PR-AUC: {avg_pr_auc:.4f} ± {std_pr_auc:.4f}")
    logger.info(f"Average Train time: {avg_train_time:.2f} seconds ± {std_train_time:.2f} seconds")
    logger.info(f"Average Test time: {avg_test_time:.2f} seconds ± {std_test_time:.2f} seconds")
    
    logger.info("=== Overall Results ===")
    logger.info(f"Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Precision: {overall_precision:.4f}")
    logger.info(f"Recall: {overall_recall:.4f}")
    logger.info(f"F1: {overall_f1:.4f}")
    logger.info(f"ROC-AUC: {overall_roc_auc:.4f}")
    logger.info(f"PR-AUC: {overall_pr_auc:.4f}")
    logger.info(f"{overall_accuracy:.4f}, {overall_precision:.4f}, {overall_recall:.4f}, {overall_f1:.4f}, {overall_roc_auc:.4f}, {overall_pr_auc:.4f}")
    
    logger.info("=== Individual Fold Results ===")
    for result in fold_results:
        logger.info(f"Fold {result['fold']}: Accuracy={result['accuracy']:.4f}, "
                   f"Precision={result['precision']:.4f}, Recall={result['recall']:.4f}, "
                   f"F1={result['f1']:.4f}, ROC-AUC={result['roc_auc']:.4f}")
    logger.info("--------------------------------------------------")
    
    # Save prediction confusion matrix
    if model_name:
        save_confusion_matrix(all_labels, all_predictions, model_name, os.path.dirname(work_log), key)
    
    # Save Optuna study
    if model_name:
        study_path = os.path.join(os.path.dirname(work_log), f'optuna_study_{model_name}.pkl')
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        logger.info(f"Optuna study saved to: {study_path}")
    
    return fold_results, (avg_accuracy, avg_precision, avg_recall, avg_f1), study


def save_confusion_matrix(true_labels, predictions, model_name, save_folder, key):
    """Save a confusion-matrix plot and its summary metrics."""
    plt.figure(figsize=(10, 8))
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(true_labels), 
                yticklabels=np.unique(true_labels))
    
    # plt.title(f'Confusion Matrix - {key} Classification - {model_name} (Optuna Optimized)')
    title_task = key.split('.')[1].replace('combined', 'label')
    plt.title(f'Confusion Matrix - {title_task} Classification - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Calculate and display metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    # Note: ROC-AUC calculation requires probabilities, which we don't have here
    # We'll use a placeholder or calculate it separately if needed
    metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}'
    plt.figtext(0.02, 0.98, metrics_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
                verticalalignment='top')
    
    plt.tight_layout()
    plot_path = os.path.join(save_folder, f'confusion_matrix_{model_name}_optuna.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log the plot save
    logger = logging.getLogger('classification_optuna')
    logger.info(f"Confusion matrix saved to: {plot_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, required=True, help='Train ratio (e.g. 0.25)')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--save_subfolder', type=str, required=True, help='Subfolder name under classification_experiments (e.g. fat_sup_0.25)')
    parser.add_argument(
    '--tag',
    type=str,
    required=True,
    choices=[
        'sequence',
        'contrast',
        'fat_suppression',
        'region'
    ],
    help='Classification task to run'
    )
    args = parser.parse_args()

    ratios = [args.ratio]
    # tags = ['classification.contrast_combined', 'classification.fat_suppression_label']  # 'classification.region_label', 'classification.sequence_label'
    # tags = ['classification.fat_suppression_label']
    tag_mapping = {
    'sequence': 'classification.sequence_label',
    'contrast': 'classification.contrast_combined',
    'fat_suppression': 'classification.fat_suppression_label',
    'region': 'classification.region_label'
    }

    tags = [tag_mapping[args.tag]]
    split_cols_mapping = {
        'classification.region_label': 'split.region',
        'classification.sequence_label': 'split.region', # 'split.sequence',
        'classification.contrast_combined': 'split.region', # 'split.contrast',
        'classification.fat_suppression_label': 'split.region', # 'split.fat_sup',
    }

    binary_flags = {
        'classification.sequence_label': False,
        'classification.region_label': False,
        'classification.contrast_combined': True,
        'classification.fat_suppression_label': True
    }

    exp_folder = 'MRI_Anatomy_Cohort_metadata'
    csv_file_path = args.csv_file
    df = pd.read_csv(csv_file_path, low_memory=False)
    data_folder = exp_folder + '/data/GE_FM_V1'
    os.makedirs(data_folder, exist_ok=True)

    n_trials = 50 #50  # Number of Optuna trials 50
    model_selection_criteria = 'f1_pr_auc'  # Use combined F1 and ROC-AUC for classification
    # average_methods = ['weighted', 'macro']
    average_methods = ['macro']
    model_name = 'GE_MRI_3D_FM'
    n_folds = 1

    save_folder = exp_folder + f'/classification_experiments/{args.save_subfolder}'
    os.makedirs(save_folder, exist_ok=True)

    for ratio in ratios:
        for tag in tags:
            for average_method in average_methods:
                cur_save_folder = os.path.join(save_folder, tag.split('.')[1] + f'_{average_method}_{model_selection_criteria}')
                os.makedirs(cur_save_folder, exist_ok=True)
                work_log = os.path.join(cur_save_folder, f'work_log_{int(ratio*100)}.txt')
                cur_data_folder = os.path.join(data_folder, tag)
                os.makedirs(cur_data_folder, exist_ok=True)
                
                # Setup logging for this run
                logger = setup_logging(work_log)
                logger.info(f"===== Running {tag} classification for {model_name} with ratio {ratio*100}% and Optuna optimization =====")
                logger.info(f"{model_name} MLP for {tag} classification - Optuna optimization")

                fold_splits, label_encoder = load_data(df, tag, split_cols_mapping[tag], cur_data_folder, binary_flags[tag])
                for fold_idx, (train_features, train_labels, val_features, val_labels, test_features, test_labels) in enumerate(fold_splits):
                    logger.info(f"=== Fold {fold_idx + 1}/{len(fold_splits)} ===")
                    logger.info(f"Train features: {train_features.shape}, Train labels: {train_labels.shape}, Val features: {val_features.shape}, Val labels: {val_labels.shape}, Test features: {test_features.shape}, Test labels: {test_labels.shape}")

                cross_validation_optuna(fold_splits, tag, work_log, ratio=ratio, model_name=model_name, n_trials=n_trials, model_selection_criteria=model_selection_criteria, average_method=average_method, n_folds=n_folds)
