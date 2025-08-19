import torch

def get_raw_res(raws, paradigm):
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


