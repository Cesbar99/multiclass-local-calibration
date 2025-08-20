import torch
import os
from os.path import join
import pandas as pd 
import sys

def get_raw_res(raws):
    preds = torch.cat([raws[j]["preds"].cpu() for j in range(len(raws))])
    probs = torch.cat([raws[j]["probs"].cpu() for j in range(len(raws))])
    logits = torch.cat([raws[j]["logits"].cpu() for j in range(len(raws))])
    true = torch.cat([raws[j]["true"].cpu() for j in range(len(raws))])
    
    raw_res = pd.DataFrame()
    raw_res["true"] = true.numpy().flatten()
    raw_res["preds"] = preds.numpy()
    raw_res["logits"] = logits.numpy()
    raw_res["probs"] = probs.numpy()
    tmp = pd.DataFrame()

    for i in range(probs.shape[1]):
        tmp["class_probs_{}".format(i)] = probs[:, i].cpu().numpy()
        tmp["logits_{}".format(i)] = logits[:, i].cpu().numpy()
        
    raw_res = pd.concat([raw_res, tmp], axis=1)    
    return raw_res

def create_logdir(name: str, resume_training: bool, wandb_logger):
    basepath = os.path.dirname(os.path.abspath(sys.argv[0]))
    basepath = os.path.join(os.path.dirname(os.path.dirname(basepath)), 'result')
    basepath = join(basepath, 'runs', name)
    # basepath = join(os.path.dirname(os.path.abspath(sys.argv[0])),'runs', name)
    run_name = wandb_logger.experiment.name
    logdir = join(basepath,run_name)
    if os.path.exists(logdir) and not resume_training:
        raise Exception(f'Run {run_name} already exists. Please delete the folder {logdir} or choose a different run name.')
    os.makedirs(logdir,exist_ok=True)
    return logdir



