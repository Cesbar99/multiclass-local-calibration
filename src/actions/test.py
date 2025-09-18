import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from utils.utils import *

def test(kwargs):
    #scaler = StandardScaler()    
    if kwargs.data == 'mnist' and kwargs.dataset.variant:
            kwargs.data = kwargs.data + '_' + kwargs.dataset.variant 
            
    if kwargs.exp_name == 'pre-train':                     
        n_bins = kwargs.n_bins_calibration_metrics        
        appendix =  kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
        test_file_name = 'multicalss_calibration_train_cal'+'.png'                
        cal_file_name = 'multicalss_calibration_eval_cal'+'.png'        
        save_path = join(kwargs.save_path_calibration_plots, appendix)
        os.makedirs(save_path, exist_ok=True)    
        test_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.checkpoint.num_classes,
                kwargs.checkpoint.num_features,
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature            
            )
        cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.checkpoint.num_classes,
                kwargs.checkpoint.num_features,
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature            
            )
        
        # Load your data
        df_test = pd.read_csv(test_results)
        #for i in range(len(df_test.columns)):
        #    if not df_test.columns[i].startswith('features'):
        #        print(df_test.columns[i])
        df_cal = pd.read_csv(cal_results)

        # Compute accuracy
        accuracy_test = (df_test['preds'] == df_test['true']).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')
        accuracy_cal = (df_cal['preds'] == df_cal['true']).mean()
        print(f'Cal accuracy: {accuracy_cal:.2%}')    
        
        # Extract logits and true labels        
        # if kwargs.return_features:
        #     logits_test = logits_test.drop(columns=logits_test.filter(regex=r'^features').columns)
        #     logits_test = logits_test.drop(columns=logits_test.filter(regex=r'^pca').columns)
        logits_test = df_test.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
        labels_test = df_test['true']
        #logits_cal = df_cal.drop(columns=['preds', 'true'])
        #logits_cal = logits_cal.drop(columns=logits_cal.filter(regex=r'^features').columns)
        logits_cal = df_cal.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
        labels_cal = df_cal['true']
        
        logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
        y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)

        logits_cal_ = torch.tensor(logits_cal.values, dtype=torch.float32)
        y_true_cal_ = torch.tensor(labels_cal.values, dtype=torch.long)

        # Convert logits to probabilities
        probs_test = F.softmax(logits_test_, dim=1)
        probs_cal = F.softmax(logits_cal_, dim=1)

        # Compute calibration metrics
        ecce_test, ece_test, mce_test, brier_test, nll_test = compute_multiclass_calibration_metrics(probs_test, y_true_test_, n_bins)
        results = {
            "ECCE": [ecce_test],
            "ECE": [ece_test],
            "MCE": [mce_test],
            "Brier": [brier_test],
            "NLL": [nll_test]
        }

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "metric_train_cal_.csv")

        # Save to CSV
        df.to_csv(output_file, index=False)  
        
        ecce_cal, ece_cal, mce_cal, brier_cal, nll_cal = compute_multiclass_calibration_metrics(probs_cal, y_true_cal_, n_bins)
        
        results = {
            "ECCE": [ecce_cal],
            "ECE": [ece_cal],
            "MCE": [mce_cal],
            "Brier": [brier_cal],
            "NLL": [nll_cal]
        }

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "metric_eval_cal_.csv")

        # Save to CSV
        df.to_csv(output_file, index=False)  

        # Print results
        print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}")
        print(f"Cal Calibration — ECCE: {ecce_cal:.4f}, ECE: {ece_cal:.4f}, MCE: {mce_cal:.4f}, Brier: {brier_cal:.4f}, NLL: {nll_cal:.4f}")
        multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name)
        multiclass_calibration_plot(y_true_cal_, probs_cal, n_bins=n_bins, save_path=save_path, filename=cal_file_name)
            
    elif kwargs.exp_name == 'calibrate':
        if kwargs.calibrate:
            total_epochs = kwargs.models.epochs
        else:
            total_epochs =  kwargs.checkpoint.epochs
        n_bins = kwargs.n_bins_calibration_metrics  
        gamma = kwargs.gamma              
        if kwargs.data == 'synthetic':
            appendix = kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
            test_file_name = 'multicalss_calibration_test' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)    
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.checkpoint.seed,
                    total_epochs,                
                )        
        else:
            appendix = kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
            test_file_name = 'multicalss_calibration_test' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)                
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.checkpoint.seed,
                    total_epochs,                
                )
                   
        # Load your data
        df_test = pd.read_csv(test_results)        

        # Compute accuracy
        accuracy_test = (df_test['preds'] == df_test['true']).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')        
        
        # Extract logits and true labels
        logits_test = df_test.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
        pca_test = df_test.filter(regex=r'^features')
        labels_test = df_test['true']
        
        logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
        pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
        y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)

        # Convert logits to probabilities
        probs_test = F.softmax(logits_test_, dim=1)        

        # Compute calibration metrics        
        ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, n_bins, gamma=gamma) 
        results = {
            "ECCE": [ecce_test],       
            "ECE": [ece_test],
            "MCE": [mce_test],
            "Brier": [brier_test],
            "NLL": [nll_test],
            "LCE": [lce_test],
            "MLCE": [mlce_test]
        }

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'metrics.csv') #'metric_' + appendix + 

        # Save to CSV
        df.to_csv(output_file, index=False)        
    
        # Print results
        print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
        multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name)                
        
        train_file_name = 'multicalss_calibration_train' + '.png'        
        train_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.checkpoint.seed,
                    total_epochs,                
                )
        
        # Load your data
        df_train = pd.read_csv(train_results)        

        # Compute accuracy
        accuracy_train = (df_train['preds'] == df_train['true']).mean()
        print(f'Test accuracy: {accuracy_train:.2%}')        
        
        # Extract logits and true labels
        logits_train = df_train.filter(regex=r'^logits') #df_train.drop(columns=['preds', 'true'])
        pca_train = df_train.filter(regex=r'^features')
        labels_train = df_train['true']
        
        logits_train_ = torch.tensor(logits_train.values, dtype=torch.float32)
        pca_train_ = torch.tensor(pca_train.values, dtype=torch.float32)
        y_true_train_ = torch.tensor(labels_train.values, dtype=torch.long)

        # Convert logits to probabilities
        probs_train = F.softmax(logits_train_, dim=1)        

        # Compute calibration metrics
        ecce_train, ece_train, mce_train, brier_train, nll_train, lce_train, mlce_train = compute_multiclass_calibration_metrics_w_lce(probs_train, y_true_train_, pca_train_, n_bins, gamma=gamma) 
        results = {            
            "ECCE": [ecce_train],       
            "ECE": [ece_train],            
            "MCE": [mce_train],
            "Brier": [brier_train],
            "NLL": [nll_train],
            "LCE": [lce_train],
            "MLCE": [mlce_train]
        }
    
        # Print results
        print(f"Test Calibration — ECCE: {ecce_train:.4f}, ECE: {ece_train:.4f}, MCE: {mce_train:.4f}, Brier: {brier_train:.4f}, NLL: {nll_train:.4f}, LCE: {lce_train:.4f}") #, MLCE: {mlce_train:.4f}")        
        multiclass_calibration_plot(y_true_train_, probs_train, n_bins=n_bins, save_path=save_path, filename=train_file_name)   
                
    elif kwargs.exp_name == 'competition':                     
        n_bins = kwargs.n_bins_calibration_metrics 
        gamma = kwargs.gamma 
            
        appendix = kwargs.exp_name + '_' + kwargs.method + '_'+ kwargs.data + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
        test_file_name = 'multicalss_calibration_test' + '.png'        
        save_path = join(kwargs.save_path_calibration_plots, appendix)
        os.makedirs(save_path, exist_ok=True)                
        test_results = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.checkpoint.seed,
                    kwargs.models.max_iter,                
                )
        
        # Load your data
        df_test = pd.read_csv(test_results)        

        # Compute accuracy
        accuracy_test = (df_test['preds'] == df_test['true']).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')        
        
        # Extract logits and true labels
        logits_test = df_test.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
        pca_test = df_test.filter(regex=r'^features')
        labels_test = df_test['true']
        
        logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
        pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
        y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)

        # Convert logits to probabilities
        probs_test = F.softmax(logits_test_, dim=1)              
                  
        # Compute calibration metrics                     
        ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, n_bins, gamma=gamma) 
        results = {
            "ECCE": [ecce_test],       
            "ECE": [ece_test],
            "MCE": [mce_test],
            "Brier": [brier_test],
            "NLL": [nll_test],
            "LCE": [lce_test],
            "MLCE": [mlce_test]
        }

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'metrics.csv') #'metric_' + appendix + 

        # Save to CSV
        df.to_csv(output_file, index=False)        
        
        # Print results
        print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
        multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name)                
        
        train_file_name = 'multicalss_calibration_train' + '.png'        
        train_results = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.checkpoint.seed,
                    kwargs.models.max_iter,                
                )
        
        # Load your data
        df_train = pd.read_csv(train_results)        

        # Compute accuracy
        accuracy_train = (df_train['preds'] == df_train['true']).mean()
        print(f'Test accuracy: {accuracy_train:.2%}')        
        
        # Extract logits and true labels
        logits_train = df_train.filter(regex=r'^logits') #df_train.drop(columns=['preds', 'true'])
        pca_train = df_train.filter(regex=r'^features')
        labels_train = df_train['true']
        
        logits_train_ = torch.tensor(logits_train.values, dtype=torch.float32)
        pca_train_ = torch.tensor(pca_train.values, dtype=torch.float32)
        y_true_train_ = torch.tensor(labels_train.values, dtype=torch.long)

        # Convert logits to probabilities
        probs_train = F.softmax(logits_train_, dim=1)        

        # Compute calibration metrics
        ecce_train, ece_train, mce_train, brier_train, nll_train, lce_train, mlce_train = compute_multiclass_calibration_metrics_w_lce(probs_train, y_true_train_, pca_train_, n_bins, gamma=gamma) 
        results = {            
            "ECCE": [ecce_train],       
            "ECE": [ece_train],            
            "MCE": [mce_train],
            "Brier": [brier_train],
            "NLL": [nll_train],
            "LCE": [lce_train],
            "MLCE": [mlce_train]
        }
    
        # Print results
        print(f"Test Calibration — ECCE: {ecce_train:.4f}, ECE: {ece_train:.4f}, MCE: {mce_train:.4f}, Brier: {brier_train:.4f}, NLL: {nll_train:.4f}, LCE: {lce_train:.4f}") #, MLCE: {mlce_train:.4f}")        
        multiclass_calibration_plot(y_true_train_, probs_train, n_bins=n_bins, save_path=save_path, filename=train_file_name)   
                
