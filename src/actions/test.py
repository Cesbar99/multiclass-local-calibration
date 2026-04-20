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
    # if (kwargs.corruption_type) and (kwargs.corruption_type not in corruptions):
    #     raise ValueError(f'Unknown corruption type! {kwargs.corruption_type} was given.')
    sev = kwargs.severity
    if kwargs.corruption_type:
        corruption_text = kwargs.corruption_type
        if sev > 0:
            severity = kwargs.severity
            corruption_text += f'_severity_{severity}'
        print("CORRUPTION TEXT: ", corruption_text)
    else:
        corruption_text = "None"
    epochs = kwargs.checkpoint.epochs
    if epochs == 9:
        model_class = 'resnet'
    elif kwargs.checkpoint.epochs == 5:
        model_class = 'vit'
    elif kwargs.checkpoint.epochs == 20:
        model_class = 'convnext'
    else: # ftt uses 50 
        model_class = 'ftt'
        if not kwargs.data == 'weather':
            raise ValueError(
                f'Checkpoint not corresponding to a trained modl! {kwargs.checkpoint.epochs} was given but only 9 and 20 are supported')
            
    if kwargs.exp_name == 'pre-train':   
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            to_add = kwargs.data + '_' + 'shift' 
        else:
            to_add = kwargs.data
      
        temperature = kwargs.checkpoint.temperature
        # else:
        #     epochs = 'None'
        #     temperature = 1.0    
                   
        gamma = kwargs.gamma            
        n_bins = kwargs.n_bins_calibration_metrics  
        n_bins_esse = kwargs.n_bins_esse
        appendix =  kwargs.exp_name + '_' + to_add + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
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
                    corruption_text,
                    kwargs.seed,
                    epochs,
                    temperature            
                )
            
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    corruption_text,
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
            if kwargs.data == 'weather' and kwargs.dataset.shift:
                test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_shift_seed-{}_ep-{}_tmp_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed,
                    epochs,
                    temperature            
                )
            else:
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
        output_file = os.path.join(output_dir, f"accs_eval_cal_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv")
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
            ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
            output_file = os.path.join(output_dir, f"metric_eval_cal_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv")

            # Save to CSV
            df.to_csv(output_file, index=False)  
            
            # ---- Save aggregated ESS profile ----
            ess_results = {
                "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                "avg_ess": ess_profile["avg_ess_per_bin"],
                "count": ess_profile["count_per_bin"]
            }

            df_ess = pd.DataFrame(ess_results)

            ess_output_file = os.path.join(
                output_dir,
                f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
            )

            df_ess.to_csv(ess_output_file, index=False)
            print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")                
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name)
            #multiclass_calibration_plot(y_true_cal_, probs_cal, n_bins=n_bins, save_path=save_path, filename=cal_file_name)
            
    elif kwargs.exp_name == 'quantize':
         
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            to_add = kwargs.data + '_' + 'shift' + '_calsize_' + f'{kwargs.dataset.subsample}'
        else:
            to_add = kwargs.data + '_calsize_' + f'{kwargs.dataset.subsample}'
        total_epochs = kwargs.models.epochs
        # if kwargs.quantize:
        #     total_epochs = kwargs.models.epochs
        # else:
        #     total_epochs =  kwargs.checkpoint.epochs
        n_bins = kwargs.n_bins_calibration_metrics  
        n_bins_esse = kwargs.n_bins_esse
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
            appendix = name + '_' + to_add + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
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
            appendix = name + '_' + to_add + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'            
            test_file_name = 'multicalss_quantisation_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)       
            if kwargs.corruption_type:
                test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    name, #kwargs.exp_name,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    corruption_text,
                    kwargs.seed, #kwargs.checkpoint.seed,
                    total_epochs,                
                    model_class
                )        
            else:      
                if kwargs.data == 'weather' and kwargs.dataset.shift:
                    test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_shift_seed-{}_ep-{}_{}.csv".format(
                        name, #kwargs.exp_name,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
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
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +
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
            output_file = os.path.join(output_dir, f'usage_stats_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +
        else:
            output_file = os.path.join(output_dir, f'usage_stats_seed_{kwargs.seed}_{model_class}.csv') #'metric_' + appendix +  
        usage_df.to_csv(output_file, index=False)  
        # ================================
        
        # === standard deviation of learned region dependent calibration parameters ===
        alpha_test = df_test.filter(regex=r'^alpha')
        alpha_test_ = torch.tensor(alpha_test.values, dtype=torch.float32)
        
        alpha_std = torch.std(alpha_test_, dim=0)
        alpha_mean = torch.mean(alpha_test_, dim=0)
        
        if not kwargs.only_test:             
            # Extract logits and true labels
            logits_test = df_test.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
            pca_test = df_test.filter(regex=r'^pca')            
            labels_test = df_test['true']
                        
            logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
            pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
            l2_test_ = torch.tensor(df_test.filter(regex=r'^l2').values, dtype=torch.float32)
            y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)            
            
            # Convert logits to probabilities            
            probs_test = F.softmax(logits_test_, dim=1)    
            
            # Compute calibration metrics
            if kwargs.models.adabw:
                bw_test = df_test.filter(regex=r'^bandwidth')
                bw_test = torch.tensor(bw_test.values, dtype=torch.float32).squeeze() 
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce_adabw(probs_test, y_true_test_, pca_test_, bw_test, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
            else:
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_,class_freqs, n_bins, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class) #compute_multiclass_calibration_metrics_w_lce_quantv2(probs_test, y_true_test_, pca_test_, l2_test_,class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                # ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile, l2_profile = compute_multiclass_calibration_metrics_w_lce_quantv2(probs_test, y_true_test_, pca_test_, l2_test_,class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
                output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                # Save to CSV
                df.to_csv(output_file, index=False)

                # ---- Save aggregated ESS profile ----
                ess_results = {
                    "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                    "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                    "avg_ess": ess_profile["avg_ess_per_bin"],
                    "count": ess_profile["count_per_bin"]
                }

                df_ess = pd.DataFrame(ess_results)

                ess_output_file = os.path.join(
                    output_dir,
                    f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
                )

                df_ess.to_csv(ess_output_file, index=False)

                # Save to CSV
                df.to_csv(output_file, index=False)
                print(f"Test Quantisation — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")
            #else:
                all_lce = []
                all_mlce = []
                for gamma in kwargs.gammas:
                    print(f'Computing metrics with gamma {gamma}')
                    ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
                    output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                    # Save to CSV
                    df.to_csv(output_file, index=False)
                        
            print("probs_test min/max:", probs_test.min().item(), probs_test.max().item())
            # Calibration plot        
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy)
    elif kwargs.exp_name == 'calibrate':          
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            to_add = kwargs.data + '_' + 'shift' + '_calsize_' + f'{kwargs.dataset.subsample}'
        else:
            to_add = kwargs.data + '_calsize_' + f'{kwargs.dataset.subsample}'
        if kwargs.calibrate:
            total_epochs = kwargs.models.epochs
        else:
            total_epochs =  kwargs.checkpoint.epochs
        n_bins = kwargs.n_bins_calibration_metrics  
        n_bins_esse = kwargs.n_bins_esse
        gamma = kwargs.gamma              
        if kwargs.data == 'synthetic':
            appendix = kwargs.exp_name + '_' + to_add + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
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
            appendix = kwargs.exp_name + '_' + to_add + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
            if kwargs.models.lambda_kl == 0:
                appendix = 'reference_kernel' + '_' + to_add + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
            if kwargs.models.kernel_only:
                appendix = 'kernel_only' + '_' + to_add + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
            test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)    
            if kwargs.corruption_type:
                test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        corruption_text,
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
                        corruption_text,
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
                        corruption_text,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
                    )
            else:       
                root = "results/{}/{}_{}_classes_{}_features/".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features
                )                          
                if kwargs.models.lambda_kl == 0:
                    root = "results/{}/{}_{}_classes_{}_features/".format(
                        'reference_kernel',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features
                    )
                if kwargs.models.kernel_only:
                    root = "results/{}/{}_{}_classes_{}_features/".format(
                        'kernel_only',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features
                    )
                piece = f"raw_results_test_cal_seed-{kwargs.seed}_ep-{total_epochs}_{model_class}.csv"
                if kwargs.data == 'weather' and kwargs.dataset.shift:
                    piece = f"raw_results_test_cal_shift_seed-{kwargs.seed}_ep-{total_epochs}_{model_class}.csv"  
                test_results = root + piece   

                   
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
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +
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
                    ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
                    output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                    # Save to CSV
                    df.to_csv(output_file, index=False)  
                    
                    # ---- Save aggregated ESS profile ----
                    ess_results = {
                        "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                        "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                        "avg_ess": ess_profile["avg_ess_per_bin"],
                        "count": ess_profile["count_per_bin"]
                    }

                    df_ess = pd.DataFrame(ess_results)

                    ess_output_file = os.path.join(
                        output_dir,
                        f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
                    )
                    
                    df_ess.to_csv(ess_output_file, index=False)
                    print(f"Saved ESS profile to {ess_output_file}")
                    # print(f"Not Saved ESS profile to {ess_output_file}. Uncomment the line!")
                    
                    # Print results
                    print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                else:    
                    all_lce = []
                    all_mlce = []                
                    for gamma in kwargs.gammas:
                        print(f'Computing metrics with gamma {gamma}')
                        ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
                            output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                            # Save to CSV
                            df.to_csv(output_file, index=False)  
                            
                            # ---- Save aggregated ESS profile ----
                            ess_results = {
                                "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                                "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                                "avg_ess": ess_profile["avg_ess_per_bin"],
                                "count": ess_profile["count_per_bin"]
                            }

                            df_ess = pd.DataFrame(ess_results)

                            ess_output_file = os.path.join(
                                output_dir,
                                f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
                            )
                            
                            df_ess.to_csv(ess_output_file, index=False)
                            
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
                    output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                    # Save to CSV
                    df.to_csv(output_file, index=False)                                                             
                    
            print("probs_test min/max:", probs_test.min().item(), probs_test.max().item())
            # Calibration plot        
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy)
                
    elif kwargs.exp_name == 'competition':            
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            to_add = kwargs.data + '_' + 'shift' + '_calsize_' + f'{kwargs.dataset.subsample}'
        else:
            to_add = kwargs.data + '_calsize_' + f'{kwargs.dataset.subsample}'                 
        n_bins = kwargs.n_bins_calibration_metrics 
        n_bins_esse = kwargs.n_bins_esse
        gamma = kwargs.gamma 
                                
        appendix = kwargs.exp_name + '_' 
        appendix += kwargs.method 
        appendix += '_'+ to_add + '_' 
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
                        corruption_text,
                        kwargs.seed,
                        kwargs.models.max_iter,                
                        model_class
                    )
        else:
            if kwargs.data == 'weather' and kwargs.dataset.shift:
                test_results = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_shift_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
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
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +
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
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
            
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
                output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                # Save to CSV
                df.to_csv(output_file, index=False)   
                
                # ---- Save aggregated ESS profile ----
                ess_results = {
                    "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                    "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                    "avg_ess": ess_profile["avg_ess_per_bin"],
                    "count": ess_profile["count_per_bin"]
                }

                df_ess = pd.DataFrame(ess_results)

                ess_output_file = os.path.join(
                    output_dir,
                    f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
                )
                
                df_ess.to_csv(ess_output_file, index=False)
                
                # Print results
                print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                    
                all_lce = []
                all_mlce = []
                for gamma in kwargs.gammas:
                    print(f'Computing metrics with gamma {gamma}')     
                    ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
                        output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_corrupt_{corruption_text}_seed_{kwargs.seed}_{model_class}.csv') #'metric_' + appendix +

                        # Save to CSV
                        df.to_csv(output_file, index=False)                                   
            
            # Calibration plot            
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy)                
            
    elif kwargs.exp_name == 'replicate':
        
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            to_add = kwargs.data + '_' + 'shift' + '_calsize_' + f'{kwargs.dataset.subsample}'
        else:
            to_add = kwargs.data + '_calsize_' + f'{kwargs.dataset.subsample}'
        total_epochs = kwargs.models.max_iter   
        n_bins = kwargs.n_bins_calibration_metrics  
        n_bins_esse = kwargs.n_bins_esse
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
            appendix = name + '_' + to_add + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'            
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
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +
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
                        ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
                    output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                    # Save to CSV
                    df.to_csv(output_file, index=False)  
                    
                    # ---- Save aggregated ESS profile ----
                    ess_results = {
                        "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                        "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                        "avg_ess": ess_profile["avg_ess_per_bin"],
                        "count": ess_profile["count_per_bin"]
                    }

                    df_ess = pd.DataFrame(ess_results)

                    ess_output_file = os.path.join(
                        output_dir,
                        f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
                    )
                    
                    df_ess.to_csv(ess_output_file, index=False)
                    
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
                            ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
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
                            output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                            # Save to CSV
                            df.to_csv(output_file, index=False)                                                             
                        
            print("probs_test min/max:", probs_test.min().item(), probs_test.max().item())
            # Calibration plot        
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy) 
    
    elif kwargs.exp_name == 'ess_plot': 
        if kwargs.l2_plot:
            save_path = "results/metrics"  # change if needed
            summary = plot_entropy_lce(args=kwargs, save_path=save_path)         
            save_pipe_table(summary, save_dir=save_path, filename=f"entropy_lce_table_{kwargs.data}.txt")
        else:
            if kwargs.data == 'cifar100':
                model_class = 'ResNet152'
                data_name = 'CIFAR-100'
            elif kwargs.data == 'cifar10':
                model_class = 'ResNet50'
                data_name = 'CIFAR-10'
            else:
                model_class = 'ResNet50'
                data_name = 'TissueMNIST'
            metrics_root = "results/metrics"  # change if needed

            method_to_runs = collect_ess_profiles(metrics_root, kwargs.data)
            agg_dict = aggregate_method_runs(method_to_runs)

            print("Methods found:")
            for method, runs in method_to_runs.items():
                print(f"  {method}: {len(runs)} runs")

            plot_ess_profiles(
                agg_dict,
                save_path=os.path.join(metrics_root, f"ess_profile_comparison_{kwargs.data}.png"),
                title="Average Absolute LCE Across Density Bins for "+ f"{data_name} with a " + f"{model_class}", # ,
                interval="std"   # use "sem95" for 95% confidence band
        )

    elif kwargs.exp_name == 'ablate_cal_size':
        method_names = ['VQ', 'LCN', 'RK', 'SMS', 'DC', 'PS', 'TS', 'IR', 'PC']
        data_names = ['weather', 'tissue']
        for data_name in data_names:
            for method_name in method_names: 
                # method_name = "RK" # change if needed
                # data_name = "tissue" # change if needed
                
                if method_name == "VQ":
                    method = "quantize"  
                elif method_name == "LCN":
                    method = "calibrate"
                elif method_name == "RK":
                    method = "reference_kernel"
                elif method_name == "SMS":
                    method = "competition_SMS"
                elif method_name == "DC":
                    method = "competition_DC"
                elif method_name == "PS":
                    method = "competition_PS"
                elif method_name == "TS":
                    method = "competition_TS"
                elif method_name == "IR":
                    method = "competition_IR"   
                elif method_name == "PC":
                    method = "competition_PC" 
                    
                if data_name == "tissue":
                    num_classes = 8
                elif data_name == "weather":
                    num_classes = 5
                
                calsizes = (0.05, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0)
                seeds = range(42, 47)
                
                result_table = summarize_vq_by_calsize(base_dir=kwargs.save_path_calibration_metrics,
                    calsizes=calsizes,
                    seeds=seeds,
                    method=method,
                    data_name=data_name,
                    num_classes=num_classes,        
                    method_name=method_name,                    
                )                                                    
                
                output_dir = os.path.join(kwargs.save_path_calibration_metrics, "ablate_cal_size_results")
                os.makedirs(output_dir, exist_ok=True)
                print(output_dir)
                result_table.to_csv(os.path.join(output_dir, f"ablate_{method_name}_{data_name}.csv"), index=False)     
                print(f"\nSaved ablation results for {method_name} on {data_name} to {output_dir}")
                
                if data_name == "weather":
                    result_table = summarize_vq_by_calsize(base_dir=kwargs.save_path_calibration_metrics,
                    calsizes=calsizes,
                    seeds=seeds,
                    method=method,
                    data_name=data_name+"_shift",
                    num_classes=num_classes,        
                    method_name=method_name,                    
                    )                                                    
                    
                    output_dir = os.path.join(kwargs.save_path_calibration_metrics, "ablate_cal_size_results")
                    os.makedirs(output_dir, exist_ok=True)
                    print(output_dir)
                    result_table.to_csv(os.path.join(output_dir, f"ablate_{method_name}_{data_name}_shift.csv"), index=False)     
                    print(f"\nSaved ablation results for {method_name} on {data_name} to {output_dir}")
                    
    elif kwargs.exp_name == 'ablate_s_k_tissue':
        slots=(16, 32, 64, 128, 256)
        kappas=(16, 32, 64, 128, 256)
        seeds=range(42, 47)
        result_slots, result_kappa = summarize_vq_by_slot_kappa(
            base_dir=kwargs.save_path_calibration_metrics,
            slots=slots,
            kappas=kappas,
            seeds=seeds,            
            method_name="VQ"
        )
        output_dir = os.path.join(kwargs.save_path_calibration_metrics, "ablate_s_k_tissue_results")
        os.makedirs(output_dir, exist_ok=True)
        print(output_dir)
        result_slots.to_csv(os.path.join(output_dir, f"ablate_slots_tissue.csv"), index=False)
        result_kappa.to_csv(os.path.join(output_dir, f"ablate_kappas_tissue.csv"), index=False)
        print(f"\nSaved ablation of S and K results for tissue to {output_dir}")
        
    elif kwargs.exp_name == 'convnext_results':
        method_names = ['VQ', 'LCN', 'RK', 'SMS', 'DC', 'PS', 'TS', 'IR', 'PC', 'NC']
        data_names = ['cifar10', 'cifar100', 'tissue']
                
        for data_name in data_names:
            reusults_tables = []
            for method_name in method_names: 
                # method_name = "RK" # change if needed
                # data_name = "tissue" # change if needed
                
                if method_name == "VQ":
                    method = "quantize"  
                elif method_name == "LCN":
                    method = "calibrate"
                elif method_name == "RK":
                    method = "reference_kernel"
                elif method_name == "SMS":
                    method = "competition_SMS"
                elif method_name == "DC":
                    method = "competition_DC"
                elif method_name == "PS":
                    method = "competition_PS"
                elif method_name == "TS":
                    method = "competition_TS"
                elif method_name == "IR":
                    method = "competition_IR"   
                elif method_name == "PC":
                    method = "competition_PC" 
                elif method_name == "NC":
                    method = "pre-train"
                    
                if data_name == "tissue":
                    num_classes = 8
                elif data_name == "weather":
                    num_classes = 5
                elif data_name == "cifar10":
                    num_classes = 10
                elif data_name == "cifar100":
                    num_classes = 100
                
                calsizes = [1.0]
                seeds = range(42, 47)
                
                result_table = summarize_vq_by_calsize(base_dir=kwargs.save_path_calibration_metrics,
                    calsizes=calsizes,
                    seeds=seeds,
                    method=method,
                    data_name=data_name,
                    num_classes=num_classes,        
                    method_name=method_name,   
                    model_class="convnext"                 
                )                                                    
                
                reusults_tables.append(result_table)
        
            final_table = pd.concat(reusults_tables, ignore_index=True)   
            final_table.drop(columns=['calsize'], inplace=True)
                
            output_dir = os.path.join(kwargs.save_path_calibration_metrics, "summary_convnext_results")
            os.makedirs(output_dir, exist_ok=True) 
                
            final_table.to_csv(os.path.join(output_dir, f"{data_name}.csv"), index=False)     
            print(f"\nSaved ablation results for {method_name} on {data_name} to {output_dir}")
                
