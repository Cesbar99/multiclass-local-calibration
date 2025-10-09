import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from utils.utils import *

def test(kwargs):
    class_freqs = kwargs.dataset.class_freqs  
    
    if kwargs.exp_name == 'pre-train':              
        gamma = kwargs.gamma            
        n_bins = kwargs.n_bins_calibration_metrics        
        appendix =  kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
        test_file_name = 'multicalss_calibration_train_cal'+'.png'                
        cal_file_name = 'multicalss_calibration_eval_cal'+'.png'        
        save_path = join(kwargs.save_path_calibration_plots, appendix)
        os.makedirs(save_path, exist_ok=True)    
        
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
        df_cal = pd.read_csv(cal_results)

        # Compute accuracy
        accuracy_cal = (df_cal['preds'] == df_cal['true']).mean()
        print(f'Cal accuracy: {accuracy_cal:.2%}')   
        
        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "accs_eval_cal.csv")
        
        if not kwargs.only_test: 
        
            # Extract logits and true labels        
            logits_cal = df_cal.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
            pca_cal = df_cal.filter(regex=r'^features')
            labels_cal = df_cal['true']

            logits_cal_ = torch.tensor(logits_cal.values, dtype=torch.float32)
            pca_cal_ = torch.tensor(pca_cal.values, dtype=torch.float32)
            y_true_cal_ = torch.tensor(labels_cal.values, dtype=torch.long)

            # Convert logits to probabilities
            probs_cal = F.softmax(logits_cal_, dim=1)

            # Compute calibration metrics
            ecce_cal, ece_cal, mce_cal, brier_cal, nll_cal, lce_cal, mlce_cal = compute_multiclass_calibration_metrics_w_lce(probs_cal, y_true_cal_, pca_cal_, class_freqs, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy)         
            results = {
                "ECCE": [ecce_cal],       
                "ECE": [ece_cal],
                "MCE": [mce_cal],
                "Brier": [brier_cal],
                "NLL": [nll_cal],
                "LCE": [lce_cal],
                "MLCE": [mlce_cal]
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
            print(f"Cal Calibration — ECCE: {ecce_cal:.4f}, ECE: {ece_cal:.4f}, MCE: {mce_cal:.4f}, Brier: {brier_cal:.4f}, NLL: {nll_cal:.4f}, LCE: {lce_cal:.4f}") #, MLCE: {mlce_cal:.4f}")
            multiclass_calibration_plot(y_true_cal_, probs_cal, n_bins=n_bins, save_path=save_path, filename=cal_file_name)
                
    elif kwargs.exp_name == 'calibrate':
        if kwargs.calibrate:
            total_epochs = kwargs.models.epochs
        else:
            total_epochs =  kwargs.checkpoint.epochs
        n_bins = kwargs.n_bins_calibration_metrics  
        gamma = kwargs.gamma         
        
        appendix = kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
        test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
        save_path = join(kwargs.save_path_calibration_plots, appendix)
        os.makedirs(save_path, exist_ok=True)                
        test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                kwargs.seed, #kwargs.checkpoint.seed,
                total_epochs,                
            )
                   
        # Load your data
        df_test = pd.read_csv(test_results)        

        # Compute accuracy
        accuracy_test = (df_test['preds'] == df_test['true']).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')  
        accs = {'acc': [accuracy_test]}
        # Convert to DataFrame
        df_accs = pd.DataFrame(accs)
        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}.csv') 
        # Save to CSV
        df_accs.to_csv(output_file, index=False)      
        
        if not kwargs.only_test: 
            # Extract logits and true labels
            logits_test = df_test.filter(regex=r'^logits') 
            pca_test = df_test.filter(regex=r'^features')
            labels_test = df_test['true']
            
            logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
            pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
            y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)
            
            # Convert logits to probabilities
            probs_test = F.softmax(logits_test_, dim=1)    
            
            # Compute calibration metrics
            if kwargs.calibrate:
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy) 
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
                output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}.csv') #'metric_' + appendix + 

                # Save to CSV
                df.to_csv(output_file, index=False)  
                
                # Print results
                print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
            else:    
                all_lce = []
                all_mlce = []                
                for gamma in kwargs.gammas:
                    print(f'Computing metrics with gamma {gamma}')
                    ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
                    all_lce.append(lce_test)
                    all_mlce.append(mlce_test)
                    if gamma == kwargs.gamma:
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
                        output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}.csv') #'metric_' + appendix + 

                        # Save to CSV
                        df.to_csv(output_file, index=False)  
                        
                        # Print results
                        print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                
                gamma_plot = {
                    'GAMMA': kwargs.gammas,
                    'LCE': all_lce,
                    'MLCE': all_mlce
                }
                
                # Convert to DataFrame
                df = pd.DataFrame(gamma_plot)

                # Specify your directory and filename
                output_dir = join(kwargs.save_path_calibration_metrics, appendix)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}.csv') #'metric_' + appendix + 

                # Save to CSV
                df.to_csv(output_file, index=False)                                                             
            
            # Calibration plot        
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy)                
            
    elif kwargs.exp_name == 'competition':                     
        n_bins = kwargs.n_bins_calibration_metrics 
        gamma = kwargs.gamma 
                                
        appendix = kwargs.exp_name + '_' 
        appendix += kwargs.method 
        appendix += '_'+ kwargs.data + '_' 
        appendix += f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
        test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
        save_path = join(kwargs.save_path_calibration_plots, appendix)
        os.makedirs(save_path, exist_ok=True)                
        test_results = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.seed,
                    kwargs.models.max_iter,                
                )
        
        # Load your data
        df_test = pd.read_csv(test_results)        

        # Compute accuracy
        accuracy_test = (df_test['preds'] == df_test['true']).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')   
        accs = {'acc': [accuracy_test]}
        # Convert to DataFrame
        df_accs = pd.DataFrame(accs)
        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}.csv') #'metric_' + appendix +  
        # Save to CSV
        df_accs.to_csv(output_file, index=False)    
               
        if not kwargs.only_test: 
            # Extract logits and true labels
            logits_test = df_test.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
            pca_test = df_test.filter(regex=r'^features')
            labels_test = df_test['true']
            
            logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
            pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
            y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)

            # Convert logits to probabilities
            if kwargs.method in ['DC', 'IR', 'PS']:
                probs_test = logits_test_              
            else:
                probs_test = F.softmax(logits_test_, dim=1)              
                    
            # Compute calibration metrics
            all_lce = []
            all_mlce = []
            for gamma in kwargs.gammas:
                print(f'Computing metrics with gamma {gamma}')                    
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
                all_lce.append(lce_test)
                all_mlce.append(mlce_test)
                
                if gamma == kwargs.gamma:
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
                    output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}.csv') #'metric_' + appendix + 

                    # Save to CSV
                    df.to_csv(output_file, index=False)   
                    
                    # Print results
                    print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                            
            gamma_plot = {
                'GAMMA': kwargs.gammas,
                'LCE': all_lce,
                'MLCE': all_mlce
            }
            
            # Convert to DataFrame
            df = pd.DataFrame(gamma_plot)

            # Specify your directory and filename
            output_dir = join(kwargs.save_path_calibration_metrics, appendix)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}.csv') #'metric_' + appendix + 

            # Save to CSV
            df.to_csv(output_file, index=False)                                   
            
            # Calibration plot            
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy)                
            
   
   
                
