import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from utils.utils import *

def test(kwargs):
    #scaler = StandardScaler()
    if kwargs.exp_name not in ['pre-train', 'calibrate']:
        raise ValueError(f"Explicitly provide 'exp_name' argument from CLI when testing! Allowed values are 'pre-train' and 'calibrate'. Instead '{kwargs.exp_name}' was given!")
    
    elif kwargs.exp_name == 'pre-train':
        n_bins = kwargs.n_bins_calibration_metrics        
        appendix =  kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.checkpoint.num_classes}' + '_classes_' + f'{kwargs.checkpoint.num_features}'
        test_file_name = 'multicalss_calibration_' + appendix + '_train_cal'+'.png'                
        cal_file_name = 'multicalss_calibration_' + appendix + '_eval_cal'+'.png'        
        save_path = kwargs.save_path_calibration_plots
        os.makedirs(save_path, exist_ok=True)    
        test_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                kwargs.exp_name,
                kwargs.checkpoint.data,
                kwargs.checkpoint.num_classes,
                kwargs.checkpoint.num_features,
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature            
            )
        cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                kwargs.exp_name,
                kwargs.checkpoint.data,
                kwargs.checkpoint.num_classes,
                kwargs.checkpoint.num_features,
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature            
            )
        
        # Load your data
        df_test = pd.read_csv(test_results)
        df_cal = pd.read_csv(cal_results)

        # Compute accuracy
        accuracy_test = (df_test['preds'] == df_test['true']).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')
        accuracy_cal = (df_cal['preds'] == df_cal['true']).mean()
        print(f'Cal accuracy: {accuracy_cal:.2%}')    
        
        # Extract logits and true labels
        logits_test = df_test.drop(columns=['preds', 'true'])
        labels_test = df_test['true']
        logits_cal = df_cal.drop(columns=['preds', 'true'])
        labels_cal = df_cal['true']
        
        logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
        y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)

        logits_cal_ = torch.tensor(logits_cal.values, dtype=torch.float32)
        y_true_cal_ = torch.tensor(labels_cal.values, dtype=torch.long)

        # Convert logits to probabilities
        probs_test = F.softmax(logits_test_, dim=1)
        probs_cal = F.softmax(logits_cal_, dim=1)

        # Compute calibration metrics
        ece_test, mce_test, brier_test = compute_multiclass_calibration_metrics(probs_test, y_true_test_, n_bins)
        results = {
            "ECE": [ece_test],
            "MCE": [mce_test],
            "Brier": [brier_test]
        }

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Specify your directory and filename
        output_dir = "results/calibration_metrics"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "metric_" + appendix + "_train_cal_.csv")

        # Save to CSV
        df.to_csv(output_file, index=False)  
        
        ece_cal, mce_cal, brier_cal = compute_multiclass_calibration_metrics(probs_cal, y_true_cal_, n_bins)
        
        results = {
            "ECE": [ece_cal],
            "MCE": [mce_cal],
            "Brier": [brier_cal]
        }

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Specify your directory and filename
        output_dir = "results/calibration_metrics"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "metric_" + appendix + "_eval_cal_.csv")

        # Save to CSV
        df.to_csv(output_file, index=False)  

        # Print results
        print(f"Test Calibration — ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}")
        print(f"Cal Calibration — ECE: {ece_cal:.4f}, MCE: {mce_cal:.4f}, Brier: {brier_cal:.4f}")
        multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name)
        multiclass_calibration_plot(y_true_cal_, probs_cal, n_bins=n_bins, save_path=save_path, filename=cal_file_name)\
            
    elif kwargs.exp_name == 'calibrate':
        n_bins = kwargs.n_bins_calibration_metrics
        appendix = kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.checkpoint.num_classes}' + '_classes_' + f'{kwargs.checkpoint.num_features}' + '_test'
        test_file_name = 'multicalss_calibration_' + appendix + + '.png'        
        save_path = kwargs.save_path_calibration_plots
        os.makedirs(save_path, exist_ok=True)    
        test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                kwargs.exp_name,
                kwargs.checkpoint.data,
                kwargs.checkpoint.num_classes,
                kwargs.checkpoint.num_features,
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,                
            )        
        
        # Load your data
        df_test = pd.read_csv(test_results)        

        # Compute accuracy
        accuracy_test = (df_test['preds'] == df_test['true']).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')        
        
        # Extract logits and true labels
        logits_test = df_test.drop(columns=['preds', 'true'])
        labels_test = df_test['true']
        
        logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
        y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)

        # Convert logits to probabilities
        probs_test = F.softmax(logits_test_, dim=1)        

        # Compute calibration metrics
        ece_test, mce_test, brier_test = compute_multiclass_calibration_metrics(probs_test, y_true_test_, n_bins) 
        results = {
            "ECE": [ece_test],
            "MCE": [mce_test],
            "Brier": [brier_test]
        }

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Specify your directory and filename
        output_dir = "results/calibration_metrics"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'metric_' + appendix + '.csv')

        # Save to CSV
        df.to_csv(output_file, index=False)        
   
        # Print results
        print(f"Test Calibration — ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}")        
        multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name)                
                
        
