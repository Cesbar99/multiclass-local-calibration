import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from utils.utils import *


def test(kwargs):
    class_freqs = kwargs.dataset.class_freqs
    #scaler = StandardScaler()    
    if kwargs.data == 'mnist' and kwargs.dataset.variant:
            kwargs.data = kwargs.data + '_' + kwargs.dataset.variant 
            
    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "fog",
        "snow",
        "frost", # try        
        "brightness", # good
        "contrast",
        "pixelate",        
    ]
    
    if (kwargs.corruption_type) and (kwargs.corruption_type not in corruptions):
        raise ValueError(f'Unknown corruption type! {kwargs.corruption_type} was given.')
    
    epochs = kwargs.checkpoint.epochs
    if epochs == 9:
        model_class = 'resnet'
    elif kwargs.checkpoint.epochs == 5:
        model_class = 'vit'
    else:
        raise ValueError(f'Checkpoint not corresponding to a trained modl! {kwargs.checkpoint.epochs} was given but only 9 and 20 are supported')
            
    if kwargs.exp_name == 'pre-train':   
        # if kwargs.data != 'food101':        
        temperature = kwargs.checkpoint.temperature
        # else:
        #     epochs = 'None'
        #     temperature = 1.0    
                   
        gamma = kwargs.gamma            
        n_bins = kwargs.n_bins_calibration_metrics        
        appendix =  kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
        test_file_name = 'multicalss_calibration_train_cal'+'.png'                
        cal_file_name = 'multicalss_calibration_eval_cal'+'.png'        
        save_path = join(kwargs.save_path_calibration_plots, appendix)
        os.makedirs(save_path, exist_ok=True)   
        if kwargs.corruption_type: 
            cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.corruption_type,
                    kwargs.seed,
                    epochs,
                    temperature            
                )
            
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.corruption_type,
                    kwargs.seed,
                    epochs,
                    temperature            
                )
        else:
            cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed,
                    epochs,
                    temperature            
                )
            
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed,
                    epochs,
                    temperature            
                )
        
        # Load your data
        df_cal = pd.read_csv(cal_results)
        df_test = pd.read_csv(test_results)
        
        # Extract features and labels
        cols = df_cal.columns
        # Single pass grouping
        features_cols = [c for c in cols if c.startswith("features")]
        logits_cols   = [c for c in cols if c.startswith("logits")]
        pca_cols      = [c for c in cols if c.startswith("pca")]
        # Extract values
        feats_train_cal  = df_cal[features_cols].values
        logits_train_cal = df_cal[logits_cols].values
        pca_train_cal    = df_cal[pca_cols].values

        y_train_cal = df_cal["true"].values
        p_train_cal = df_cal["preds"].values

        cols = df_test.columns
        # Single pass grouping
        features_cols = [c for c in cols if c.startswith("features")]
        logits_cols   = [c for c in cols if c.startswith("logits")]
        pca_cols      = [c for c in cols if c.startswith("pca")]
        # Extract values
        feats_eval_cal  = df_test[features_cols].values
        logits_eval_cal = df_test[logits_cols].values
        pca_eval_cal    = df_test[pca_cols].values
        
        y_eval_cal = df_test["true"].values
        p_eval_cal = df_test["preds"].values

        # if kwargs.data != 'food101':
        # Split into 90% test and 10% val
        (feats_test, feats_val,
        logits_test, logits_val,
        pca_test, pca_val,
        y_test, y_val,
        p_test, p_val) = train_test_split(
            feats_eval_cal,
            logits_eval_cal,
            pca_eval_cal,
            y_eval_cal,
            p_eval_cal,
            test_size=0.1, #0.1  # 10% for validation
            random_state=kwargs.seed, # for reproducibility
            shuffle=True) 
        
        # df_test = pd.DataFrame(
        #     np.hstack([
        #         y_test.reshape(-1, 1),
        #         p_test.reshape(-1, 1),
        #         logits_test,
        #         feats_test,                                
        #         pca_test                                
        #     ]),
        #     columns=(
        #         ["true"] +
        #         [f"preds"] +
        #         [f"logits_{i}" for i in range(logits_test.shape[1])] +
        #         [f"features_{i}" for i in range(feats_test.shape[1])] +                
        #         [f"pca_{i}" for i in range(pca_test.shape[1])]                                 
        #     )
        # )
        
        # filename = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}_only_test.csv".format(
        #         kwargs.exp_name,
        #         kwargs.data,
        #         kwargs.checkpoint.num_classes,
        #         kwargs.checkpoint.num_features,
        #         kwargs.seed,
        #         epochs,
        #         temperature            
        #     )
        # df_test.to_csv(filename, index=False)  
        # else:
        #     feats_test = feats_eval_cal            
        #     logits_test = logits_eval_cal
        #     pca_test = pca_eval_cal
        #     y_test = y_eval_cal
        #     p_test = p_eval_cal                     
        
        # Compute accuracy
        #accuracy_test = (df_test['preds'] == df_test['true']).mean()
        accuracy_test = (y_test == p_test).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')
        accs = {'acc': [accuracy_test]}
        df_accs = pd.DataFrame(accs)
        #accuracy_cal = (df_cal['preds'] == df_cal['true']).mean()
        accuracy_cal = (y_train_cal == p_train_cal).mean()
        print(f'Cal accuracy: {accuracy_cal:.2%}')   
        
        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"accs_eval_cal_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv")
        df_accs.to_csv(output_file, index=False) 
        
        if not kwargs.only_test: # COMPUTE METRIC IN ADDITION TO ACCURACY
        
            # Extract logits and true labels                    
            labels_test = y_test #df_test['true']
            #pca_test = df_test.filter(regex=r'^features')
            
            logits_cal = logits_train_cal #df_cal.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
            pca_cal = pca_train_cal #df_cal.filter(regex=r'^features')
            labels_cal = y_train_cal #df_cal['true']
            
            #logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
            logits_test_ = torch.tensor(logits_test, dtype=torch.float32)
            #pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
            pca_test_ = torch.tensor(pca_test, dtype=torch.float32)
            #y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)
            y_true_test_ = torch.tensor(labels_test, dtype=torch.long)

            #logits_cal_ = torch.tensor(logits_cal.values, dtype=torch.float32)
            logits_cal_ = torch.tensor(logits_cal, dtype=torch.float32)
            #pca_cal_ = torch.tensor(pca_cal.values, dtype=torch.float32)
            pca_cal_ = torch.tensor(pca_cal, dtype=torch.float32)
            #y_true_cal_ = torch.tensor(labels_cal.values, dtype=torch.long)
            y_true_cal_ = torch.tensor(labels_cal, dtype=torch.long)

            # Convert logits to probabilities
            probs_test = F.softmax(logits_test_, dim=1)
            probs_cal = F.softmax(logits_cal_, dim=1)

            # Compute calibration metrics
            ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
            output_file = os.path.join(output_dir, f"metric_eval_cal_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv")

            # Save to CSV
            df.to_csv(output_file, index=False)  
            
            #ecce_cal, ece_cal, mce_cal, brier_cal, nll_cal = compute_multiclass_calibration_metrics(probs_cal, y_true_cal_, n_bins)
            # ecce_cal, ece_cal, mce_cal, brier_cal, nll_cal, lce_cal, mlce_cal = compute_multiclass_calibration_metrics_w_lce(probs_cal, y_true_cal_, pca_cal_, class_freqs, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy)         
            # results = {
            #     "ECCE": [ecce_cal],       
            #     "ECE": [ece_cal],
            #     "MCE": [mce_cal],
            #     "Brier": [brier_cal],
            #     "NLL": [nll_cal],
            #     "LCE": [lce_cal],
            #     "MLCE": [mlce_cal]
            # }

            # # Convert to DataFrame
            # df = pd.DataFrame(results)

            # # Specify your directory and filename
            # output_dir = join(kwargs.save_path_calibration_metrics, appendix)
            # os.makedirs(output_dir, exist_ok=True)
            # output_file = os.path.join(output_dir, f"metric_train_cal_seed_{kwargs.seed}.csv")

            # # Save to CSV
            # df.to_csv(output_file, index=False)  

            # Print results
            #print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}")
            print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")                
            #print(f"Cal Calibration — ECCE: {ecce_cal:.4f}, ECE: {ece_cal:.4f}, MCE: {mce_cal:.4f}, Brier: {brier_cal:.4f}, NLL: {nll_cal:.4f}")
            #print(f"Cal Calibration — ECCE: {ecce_cal:.4f}, ECE: {ece_cal:.4f}, MCE: {mce_cal:.4f}, Brier: {brier_cal:.4f}, NLL: {nll_cal:.4f}, LCE: {lce_cal:.4f}") #, MLCE: {mlce_cal:.4f}")
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name)
            #multiclass_calibration_plot(y_true_cal_, probs_cal, n_bins=n_bins, save_path=save_path, filename=cal_file_name)
            
    elif kwargs.exp_name == 'quantize':
        total_epochs = kwargs.models.epochs
        # if kwargs.quantize:
        #     total_epochs = kwargs.models.epochs
        # else:
        #     total_epochs =  kwargs.checkpoint.epochs
        n_bins = kwargs.n_bins_calibration_metrics  
        gamma = kwargs.gamma             
        name = kwargs.exp_name
        if kwargs.models.S != 64:
            name += f'slot-{kwargs.models.S}'
        if kwargs.models.K != 64:
            name += f'kappa-{kwargs.models.K}'
        if kwargs.models.random:
            name += '_random'
        if kwargs.models.L1:
            name += '_L1'       
        if kwargs.models.quantization_only: 
            name += '_quantonly' 
        if kwargs.models.standard_dirichlet:
            name += '_stdcal'    
        if kwargs.models.quadratic:
            name += '_quadratic'
            
        if kwargs.data == 'synthetic':
            appendix = name + '_' + kwargs.data + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
            test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)    
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_seed-{}_ep-{}_{}.csv".format(
                    name, #kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed, #kwargs.checkpoint.seed,
                    total_epochs,                
                    model_class
                )        
        else: 
            appendix = name + '_' + kwargs.data + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'            
            test_file_name = 'multicalss_quantisation_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)       
            if kwargs.corruption_type:
                test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    name, #kwargs.exp_name,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.corruption_type,
                    kwargs.seed, #kwargs.checkpoint.seed,
                    total_epochs,                
                    model_class
                )        
            else:                     
                test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_seed-{}_ep-{}_{}.csv".format(
                        name, #kwargs.exp_name,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
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
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix +  
        # Save to CSV
        print(f"saved accuracy to {output_file}")
        df_accs.to_csv(output_file, index=False) 
                
        # === codeword usage statistics ===
        idx_np = df_test.filter(regex=r'^indices').to_numpy().ravel()
        unique, counts = np.unique(idx_np, return_counts=True)
        freq = counts / counts.sum()

        print("\n=== Codeword usage statistics ===")
        print(f"Total codewords used: {len(unique)} / {idx_np.max() + 1}")
        print(f"Min freq: {freq.min():.6f}")
        print(f"Max freq: {freq.max():.6f}")
        print(f"Mean freq (ideal): {1.0 / (idx_np.max() + 1):.6f}")

        # Entropy (max = log K)
        entropy = -np.sum(freq * np.log(freq + 1e-12))
        max_entropy = np.log(idx_np.max() + 1)
        print(f"Entropy: {entropy:.4f} / {max_entropy:.4f}")

        # Optional: print full histogram (comment out if too verbose)
        usage_df = pd.DataFrame({
            "codeword": unique,
            "count": counts,
            "frequency": freq
        }).sort_values("codeword")
        # print(usage_df)
        if kwargs.corruption_type:
            output_file = os.path.join(output_dir, f'usage_stats_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix +  
        else:
            output_file = os.path.join(output_dir, f'usage_stats_seed_{kwargs.seed}_{model_class}.csv') #'metric_' + appendix +  
        usage_df.to_csv(output_file, index=False)  
        # ================================
        
        # === standard deviation of learned region dependent calibration parameters ===
        alpha_test = df_test.filter(regex=r'^alpha')
        alpha_test_ = torch.tensor(alpha_test.values, dtype=torch.float32)
        
        alpha_std = torch.std(alpha_test_, dim=0)
        alpha_mean = torch.mean(alpha_test_, dim=0)        
        
        # print("\n=== Standard deviation of learned region-dependent calibration parameters ===")
        # #print(f"{alpha_std.item():.6f}")
        # #print('\n')
        # for i in range(alpha_std.shape[0]):
        #     print(f"Class {i}:  mean = {alpha_mean[i].item():.4f},  std = {alpha_std[i].item():.4f}, cv = {alpha_std[i].item() / (alpha_mean[i].item() + 1e-12):.4f}")                        
        # print('\n')
        # ================================
        
        if not kwargs.only_test:             
            # Extract logits and true labels
            logits_test = df_test.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
            pca_test = df_test.filter(regex=r'^pca')
            labels_test = df_test['true']
                        
            logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
            pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
            y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)            
            
            # Convert logits to probabilities            
            probs_test = F.softmax(logits_test_, dim=1)    
            
            # Compute calibration metrics
            if kwargs.models.adabw:
                bw_test = df_test.filter(regex=r'^bandwidth')
                bw_test = torch.tensor(bw_test.values, dtype=torch.float32).squeeze() 
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce_adabw(probs_test, y_true_test_, pca_test_, bw_test, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
            else:
                if kwargs.quantize or kwargs.test:
                    ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
                    output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix + 

                    # Save to CSV
                    df.to_csv(output_file, index=False)  
                    
                    # Print results
                    print(f"Test Quantisation — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                #else:    
                    all_lce = []
                    all_mlce = []                
                    for gamma in kwargs.gammas:
                        print(f'Computing metrics with gamma {gamma}')
                        ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                        all_lce.append(lce_test)
                        all_mlce.append(mlce_test)
                        # if gamma == kwargs.gamma:
                        #     results = {
                        #         "ECCE": [ecce_test],       
                        #         "ECE": [ece_test],
                        #         "MCE": [mce_test],
                        #         "Brier": [brier_test],
                        #         "NLL": [nll_test],
                        #         "LCE": [lce_test],
                        #         "MLCE": [mlce_test]
                        #     }

                        #     # Convert to DataFrame
                        #     df = pd.DataFrame(results)

                        #     # Specify your directory and filename
                        #     output_dir = join(kwargs.save_path_calibration_metrics, appendix)
                        #     os.makedirs(output_dir, exist_ok=True)
                        #     output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}.csv') #'metric_' + appendix + 

                        #     # Save to CSV
                        #     df.to_csv(output_file, index=False)  
                            
                        #     # Print results
                        #     print(f"Test Quantisation — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                    
                    if gamma != kwargs.gamma:
                        
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
                        output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix + 

                        # Save to CSV
                        df.to_csv(output_file, index=False)                                                             
                        
            print("probs_test min/max:", probs_test.min().item(), probs_test.max().item())
            # Calibration plot        
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy)                
        """    
        train_file_name = 'multicalss_quantisation_train_' + f'{kwargs.bin_strategy}' + '.png'        
        train_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_calquant_seed-{}_ep-{}.csv".format(
                    name, #kwargs.exp_name,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.seed,
                    total_epochs,                
                )        
        
        # Load your data
        df_train = pd.read_csv(train_results)        

        # Compute accuracy
        accuracy_train = (df_train['preds'] == df_train['true']).mean()
        print(f'Test accuracy: {accuracy_train:.2%}')        
        
        if not kwargs.only_test: 
            # Extract logits and true labels
            logits_train = df_train.filter(regex=r'^logits') #df_train.drop(columns=['preds', 'true'])
            pca_train = df_train.filter(regex=r'^pca')
            labels_train = df_train['true']
            
            logits_train_ = torch.tensor(logits_train.values, dtype=torch.float32)
            pca_train_ = torch.tensor(pca_train.values, dtype=torch.float32)
            y_true_train_ = torch.tensor(labels_train.values, dtype=torch.long)
            
            probs_train = F.softmax(logits_train_, dim=1)      
            
            # Compute calibration metrics
            if kwargs.models.adabw:
                bw_train = df_train.filter(regex=r'^bandwidth')
                bw_train = torch.tensor(bw_train.values, dtype=torch.float32).squeeze() 
                ecce_train, ece_train, mce_train, brier_train, nll_train, lce_train, mlce_train = compute_multiclass_calibration_metrics_w_lce_adabw(probs_train, y_true_train_, pca_train_, bw_train, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
            else:
                ecce_train, ece_train, mce_train, brier_train, nll_train, lce_train, mlce_train = compute_multiclass_calibration_metrics_w_lce(probs_train, y_true_train_, pca_train_, class_freqs, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy, model_type=model_class) 
                    
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
            print(f"Test Quantisation — ECCE: {ecce_train:.4f}, ECE: {ece_train:.4f}, MCE: {mce_train:.4f}, Brier: {brier_train:.4f}, NLL: {nll_train:.4f}, LCE: {lce_train:.4f}") #, MLCE: {mlce_train:.4f}")        
            multiclass_calibration_plot(y_true_train_, probs_train, n_bins=n_bins, save_path=save_path, filename=train_file_name, bin_strategy=kwargs.bin_strategy)   
        """                
        
    elif kwargs.exp_name == 'calibrate':
        if kwargs.calibrate:
            total_epochs = kwargs.models.epochs
        else:
            total_epochs =  kwargs.checkpoint.epochs
        n_bins = kwargs.n_bins_calibration_metrics  
        gamma = kwargs.gamma              
        if kwargs.data == 'synthetic':
            appendix = kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
            test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)    
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed, #kwargs.checkpoint.seed,
                    total_epochs,                
                )        
        else: 
            appendix = kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
            if kwargs.models.lambda_kl == 0:
                appendix = 'reference_kernel' + '_' + kwargs.data + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
            if kwargs.models.kernel_only:
                appendix = 'kernel_only' + '_' + kwargs.data + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
            test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)    
            if kwargs.corruption_type:
                test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.corruption_type,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
                    )
                if kwargs.models.lambda_kl == 0:
                    test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        'reference_kernel',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.corruption_type,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
                    )
                if kwargs.models.kernel_only:
                    test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        'kernel_only',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.corruption_type,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
                    )
            else:                        
                test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
                    )
                if kwargs.models.lambda_kl == 0:
                    test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                        'reference_kernel',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
                    )
                if kwargs.models.kernel_only:
                    test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                        'kernel_only',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
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
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix +  
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
            if kwargs.models.lambda_kl == 0 or kwargs.models.kernel_only:
                probs_test = logits_test_             
            else:
                probs_test = F.softmax(logits_test_, dim=1)    
            
            # Compute calibration metrics
            if kwargs.models.adabw:
                bw_test = df_test.filter(regex=r'^bandwidth')
                bw_test = torch.tensor(bw_test.values, dtype=torch.float32).squeeze() 
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce_adabw(probs_test, y_true_test_, pca_test_, bw_test, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
            else:
                if kwargs.calibrate:
                    ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
                    output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix + 

                    # Save to CSV
                    df.to_csv(output_file, index=False)  
                    
                    # Print results
                    print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                else:    
                    all_lce = []
                    all_mlce = []                
                    for gamma in kwargs.gammas:
                        print(f'Computing metrics with gamma {gamma}')
                        ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
                            output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix + 

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
                    output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix + 

                    # Save to CSV
                    df.to_csv(output_file, index=False)                                                             
                    
            print("probs_test min/max:", probs_test.min().item(), probs_test.max().item())
            # Calibration plot        
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy)                
        
        # SKIP PERFORMANCE ON CALIBRATION SET TO SAVE COMPUTING TIME     
        # train_file_name = 'multicalss_calibration_train_' + f'{kwargs.bin_strategy}' + '.png'        
        # train_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}.csv".format(
        #             kwargs.exp_name,
        #             kwargs.data,
        #             kwargs.dataset.num_classes,
        #             kwargs.dataset.num_features,
        #             kwargs.seed,
        #             total_epochs,                
        #         )
        # if kwargs.models.lambda_kl == 0:
        #     train_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}.csv".format(
        #             'reference_kernel',
        #             kwargs.data,
        #             kwargs.dataset.num_classes,
        #             kwargs.dataset.num_features,
        #             kwargs.seed,
        #             total_epochs,                
        #         )
        # if kwargs.models.kernel_only:
        #     train_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}.csv".format(
        #             'kernel_only',
        #             kwargs.data,
        #             kwargs.dataset.num_classes,
        #             kwargs.dataset.num_features,
        #             kwargs.seed,
        #             total_epochs,                
        #         )
        
        # # Load your data
        # df_train = pd.read_csv(train_results)        

        # # Compute accuracy
        # accuracy_train = (df_train['preds'] == df_train['true']).mean()
        # print(f'Test accuracy: {accuracy_train:.2%}')        
        
        # if not kwargs.only_test: 
        #     # Extract logits and true labels
        #     logits_train = df_train.filter(regex=r'^logits') #df_train.drop(columns=['preds', 'true'])
        #     pca_train = df_train.filter(regex=r'^features')
        #     labels_train = df_train['true']
            
        #     logits_train_ = torch.tensor(logits_train.values, dtype=torch.float32)
        #     pca_train_ = torch.tensor(pca_train.values, dtype=torch.float32)
        #     y_true_train_ = torch.tensor(labels_train.values, dtype=torch.long)

        #     # Convert logits to probabilities
        #     if kwargs.models.lambda_kl == 0 or kwargs.models.kernel_only:
        #         probs_train = logits_train_ 
        #     else:
        #         probs_train = F.softmax(logits_train_, dim=1)      
            
        #     # Compute calibration metrics
        #     if kwargs.models.adabw:
        #         bw_train = df_train.filter(regex=r'^bandwidth')
        #         bw_train = torch.tensor(bw_train.values, dtype=torch.float32).squeeze() 
        #         ecce_train, ece_train, mce_train, brier_train, nll_train, lce_train, mlce_train = compute_multiclass_calibration_metrics_w_lce_adabw(probs_train, y_true_train_, pca_train_, bw_train, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
        #     else:
        #         ecce_train, ece_train, mce_train, brier_train, nll_train, lce_train, mlce_train = compute_multiclass_calibration_metrics_w_lce(probs_train, y_true_train_, pca_train_, class_freqs, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                    
        #     results = {            
        #         "ECCE": [ecce_train],       
        #         "ECE": [ece_train],            
        #         "MCE": [mce_train],
        #         "Brier": [brier_train],
        #         "NLL": [nll_train],
        #         "LCE": [lce_train],
        #         "MLCE": [mlce_train]
        #     }
        
        #     # Print results
        #     print(f"Test Calibration — ECCE: {ecce_train:.4f}, ECE: {ece_train:.4f}, MCE: {mce_train:.4f}, Brier: {brier_train:.4f}, NLL: {nll_train:.4f}, LCE: {lce_train:.4f}") #, MLCE: {mlce_train:.4f}")        
        #     multiclass_calibration_plot(y_true_train_, probs_train, n_bins=n_bins, save_path=save_path, filename=train_file_name, bin_strategy=kwargs.bin_strategy)   
                
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
        if kwargs.corruption_type:              
            test_results = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.corruption_type,
                        kwargs.seed,
                        kwargs.models.max_iter,                
                        model_class
                    )
        else:
            test_results = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.seed,
                        kwargs.models.max_iter,                
                        model_class
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
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix +  
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
            if kwargs.method in ['SMS', 'DC', 'IR', 'PS', 'PC']:
                probs_test = logits_test_ #F.softmax(logits_test_, dim=1)              
            else:
                probs_test = F.softmax(logits_test_, dim=1)              
                    
            # Compute calibration metrics
            if kwargs.models.adabw:
                bw_data = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                        'calibrate',                    
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.seed,
                        kwargs.checkpoint.epochs_bw,                
                    )        
                # Load your data
                df_bw = pd.read_csv(bw_data)               
                bw_test = df_bw.filter(regex=r'^bandwidth')
                bw_test = torch.tensor(bw_test.values, dtype=torch.float32).squeeze() 
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce_adabw(probs_test, y_true_test_, pca_test_, bw_test, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
            else:                               
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
            
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
                output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix + 

                # Save to CSV
                df.to_csv(output_file, index=False)   
                
                # Print results
                print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                    
                all_lce = []
                all_mlce = []
                for gamma in kwargs.gammas:
                    print(f'Computing metrics with gamma {gamma}')     
                    ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                    all_lce.append(lce_test)
                    all_mlce.append(mlce_test)

                    if gamma != kwargs.gamma:
                        
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
                        output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_corrupt_{kwargs.corruption_type}_seed_{kwargs.seed}_{model_class}.csv') #'metric_' + appendix + 

                        # Save to CSV
                        df.to_csv(output_file, index=False)                                   
            
            # Calibration plot            
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy)                
            
    elif kwargs.exp_name == 'replicate':
        total_epochs = kwargs.models.max_iter   
        n_bins = kwargs.n_bins_calibration_metrics  
        gamma = kwargs.gamma             
        name = kwargs.exp_name
        name += f'{kwargs.models.n_steps}'
        if kwargs.models.kl_reg > 0:
            name += '_KL'
        if kwargs.models.state_dependent:
            name += '_DEP'
            
        if kwargs.data == 'synthetic':
            appendix = name + '_' + kwargs.data + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
            test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)    
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_replicate_seed-{}_ep-{}.csv".format(
                    name, 
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed, #kwargs.checkpoint.seed,
                    total_epochs,                
                )        
        else: 
            appendix = name + '_' + kwargs.data + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'            
            test_file_name = 'multicalss_replicate_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)                            
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_replicate_seed-{}_ep-{}_{}.csv".format(
                    name, 
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.seed,
                    kwargs.models.max_iter,
                    model_class           
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
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix +  
        # Save to CSV
        df_accs.to_csv(output_file, index=False) 
                
        if not kwargs.only_test: # if only_test only computes accuracy            
            # Extract logits and true labels
            logits_test = df_test.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
            pca_test = df_test.filter(regex=r'^features')
            labels_test = df_test['true']
                        
            logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
            pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
            y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)            
            
            # Convert logits to probabilities            
            probs_test = logits_test_ #F.softmax(logits_test_, dim=1)    
            
            # Compute calibration metrics
            if kwargs.models.adabw:
                bw_test = df_test.filter(regex=r'^bandwidth')
                bw_test = torch.tensor(bw_test.values, dtype=torch.float32).squeeze() 
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce_adabw(probs_test, y_true_test_, pca_test_, bw_test, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
            else:
                if kwargs.replicate or kwargs.test:
                    if kwargs.data == 'cubic':
                        ecce_test, ece_test, mce_test, brier_test, nll_test = compute_multiclass_calibration_metrics(probs_test, y_true_test_, n_bins, class_freqs) 
                        results = {
                            "ECCE": [ecce_test],       
                            "ECE": [ece_test],
                            "MCE": [mce_test],
                            "Brier": [brier_test],
                            "NLL": [nll_test]                            
                        }
                    else:
                        ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
                    output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix + 

                    # Save to CSV
                    df.to_csv(output_file, index=False)  
                    
                    # Print results
                    if kwargs.data == 'cubic':
                        print(f"Test Replicator — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}") #, LCE: {lce_test:.4f}, MLCE: {mlce_test:.4f}")        
                    else:
                        print(f"Test Replicator — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                 
                    all_lce = []
                    all_mlce = []       
                    if kwargs.data != 'cubic':         
                        for gamma in kwargs.gammas:
                            print(f'Computing metrics with gamma {gamma}')
                            ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                            all_lce.append(lce_test)
                            all_mlce.append(mlce_test)

                        if gamma != kwargs.gamma:
                            
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
                            output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{kwargs.corruption_type}_{model_class}.csv') #'metric_' + appendix + 

                            # Save to CSV
                            df.to_csv(output_file, index=False)                                                             
                        
            print("probs_test min/max:", probs_test.min().item(), probs_test.max().item())
            # Calibration plot        
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy) 