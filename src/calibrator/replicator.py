from fileinput import filename
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import math
from utils.utils import multiclass_calibration_plot
import os


class EquivariantFitness(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.psi = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # Optional: start near-zero so initial behavior ~ identity
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, log_q):  # (N,C)
        x = log_q.unsqueeze(-1)           # (N,C,1)
        h = self.phi(x)                   # (N,C,H)
        m = h.mean(dim=1, keepdim=True)   # (N,1,H)
        m = m.expand_as(h)                # (N,C,H)
        z = torch.cat([h, x], dim=-1)     # (N,C,H+1)
        s = self.psi(z).squeeze(-1)       # (N,C)
        return s


class ReplicatorCalibrator(nn.Module):
    """
    Replicator-flow calibrator on the simplex.

    We learn a fitness field s(q) = A * [log(q), q, 1] (Dirichlet-style features),
    then apply L replicator / mirror-descent (exponentiated-gradient) steps:

        q^{l+1} ∝ q^{l} ⊙ exp(eta * s(q^{l}))

    This keeps q on the simplex by construction and defines a nonlinear calibration map
    via composition of steps.
    """
    def __init__(
        self,
        n_classes: int,
        lr: float = 0.01,
        max_iter: int = 500,
        n_steps: int = 3,
        step_size: float = 0.1,
        eps: float = 1e-12,
        # Optional trust-region regularizer weight; 0 disables
        kl_reg: float = 0.0,
        # If True, compute fitness from evolving q; if False, use initial probs only
        state_dependent: bool = False,
        feature_dim: int = 2048,
        weight_decay: float = 0.0,
        # Optional: clamp fitness magnitude for stability (None disables)
        # fitness_clip: float | None = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.feature_dim = feature_dim 
        
        #self.A = nn.Parameter(torch.zeros(self.n_classes, self.n_classes + 1))
        self.A_t = nn.Parameter(torch.zeros(n_steps, self.n_classes, self.n_classes + 1)) #+1
        # self.W = nn.Parameter(torch.zeros(n_steps, self.n_classes, self.n_classes + 1)) #+1
        # self.U = nn.Parameter(torch.zeros(n_steps, self.n_classes, self.n_classes + 1)) #+1
        # nn.init.normal_(self.W, mean=0.0, std=1e-2)
        # nn.init.normal_(self.U, mean=0.0, std=1e-2)
        # self.A_2 = nn.Parameter(torch.zeros(n_steps, self.n_classes, self.n_classes )) #+1
        # self.T = n_steps
        # self.M = 5  # start with 2 or 3
        # self.A0 = nn.Parameter(torch.zeros(self.n_classes, self.n_classes + 1))
        # self.A_basis = nn.Parameter(torch.zeros(self.M, self.n_classes, self.n_classes + 1))
        # self.beta = nn.Parameter(torch.zeros(self.T, self.M))  # Option A
        # self.hidden = 2048
        # self.gate = nn.Sequential(
        #     nn.Linear(self.n_classes, self.hidden),
        #     nn.ReLU(),
        #     # nn.Linear(self.hidden, self.hidden),
        #     # nn.ReLU(),
        #     # nn.Linear(self.hidden, self.hidden),
        #     # nn.ReLU(),
        #     nn.Linear(self.hidden, self.n_classes)
        # )
        # self.G = nn.Parameter(torch.zeros(self.n_classes, 3)) 
        # self.B_t = nn.Parameter(torch.zeros(n_steps, n_classes, feature_dim))     # acts on f
        # self.A_2 = nn.Parameter(torch.zeros(self.n_classes, self.n_classes))
        # self.bias = nn.Parameter(torch.zeros(self.n_classes))
        
        self.lr = lr
        self.max_iter = max_iter
        self.weight_decay = weight_decay

        self.n_steps = n_steps
        self.step_size = step_size #nn.Parameter(torch.zeros(self.n_steps)) #step_size
        # nn.init.normal_(self.step_size, mean=0., std=0.01)
        # self.log_eta = nn.Parameter(torch.log(torch.tensor([self.step_size] * self.n_steps))) #nn.Parameter(torch.log(torch.tensor([step_size]*n_steps)))
        self.eps = eps
        self.kl_reg = kl_reg
        self.state_dependent = state_dependent
        # self.fitness_clip = fitness_clip

    def _features(self, probs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # probs: (N, C), assumed in simplex
        
        # IF WANT TO RUN ON LOG-PROBS USE:
        log_probs = torch.log(probs + self.eps)           
        # IF WANT TO RUN ON LOGITS USE:
        # log_probs = logits 
        
        ones = torch.ones(probs.shape[0], 1, device=probs.device, dtype=probs.dtype)
        return torch.cat([log_probs, ones], dim=1)  # (N, C+1) #torch.cat([log_probs, probs, ones], dim=1)  # (N, 2C+1)
    
    def gate_feats(self, q):
        eps = self.eps
        ent = -(q * torch.log(q + eps)).sum(dim=1, keepdim=True)
        top2 = torch.topk(q, k=2, dim=1).values
        maxp = top2[:, :1]
        margin = (top2[:, :1] - top2[:, 1:2])
        return torch.cat([ent, maxp, margin], dim=1)  # (N,3)

    def fitness(self, probs: torch.Tensor, logits: torch.Tensor, features: torch.Tensor,  timestep: int = 0) -> torch.Tensor:
        # Dirichlet-style linear fitness on features
        feats = self._features(probs, logits)          # (N, 2C+1)
        
        # TO USE SYMMETRIC (GRADIENT BASED MODELLING) RUN:
        # self.W = torch.matmul(self.A.transpose(1,1), self.A_2) 
        # s = feats @ self.W.t() + self.bias
        
        # TO USE FEATURES RUN:
        # s = features @ self.B.t() + self.bias               # (N, C) 
        # s = features @ self.B_t[timestep].t() + self.bias
        
        # TO USE PROBS RUN:
        # s = feats @ self.A.t()                # (N, C) 
        # NN PARAMETERIZATION INSTEAD:
        # s = self.gate(feats)
        
        # TO USE TIMESTEPS RUN:
        # self.W_t = torch.matmul(self.A_t[timestep].transpose(1,1), self.A_2[timestep]) 
        s = feats @ self.A_t[timestep].t() # + self.bias[timestep] #self.W_t s
        
        # ones = torch.ones(feats.size(0), 1, device=feats.device, dtype=feats.dtype)
        # h = F.logsigmoid(feats @ self.W[timestep].t()) 
        # h_cat = torch.cat([h, ones], dim=1)
        # s = h_cat @ self.U[timestep].t() #h + 0.1 *( h_cat @ self.U[timestep].t() )
        
        # A_t = self.A_at(t=timestep) 
        # s = feats @ A_t.t()
        
        # if self.fitness_clip is not None:
        #     s = torch.clamp(s, -self.fitness_clip, self.fitness_clip)
        return s

    @torch.no_grad()
    def _normalize_simplex(self, q: torch.Tensor) -> torch.Tensor:
        # numerical safety: ensure nonneg and sum=1
        q = torch.clamp(q, min=self.eps)
        return q / q.sum(dim=1, keepdim=True)

    def forward(self, probs: torch.Tensor, logits: torch.Tensor, features: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """
        probs: (N, C) uncalibrated probabilities (e.g., softmax of base logits)
        returns: (N, C) calibrated probabilities after replicator flow
        """
        q = probs
        qs = []
        features = F.layer_norm(features, (self.feature_dim,))
        if not self.state_dependent:
            # Cache fitness computed from initial probs only
            s0 = self.fitness(probs, logits, features)

        for timestep in range(self.n_steps):
            s = self.fitness(q, logits, features, timestep) if self.state_dependent else s0            
            s = s - (q * s).sum(dim=1, keepdim=True) 
            s = s - s.max(dim=1, keepdim=True).values
            #s = torch.clamp(s, -10.0, 10.0)
            # Exponentiated-gradient / replicator step
            # q = self.euler_step(q, s, self.step_size, self.eps) #F.softplus(self.log_eta[timestep])
            q = q * torch.exp(F.softplus(self.log_eta[timestep]) * s) #torch.exp(F.softplus(self.log_eta) * s) # q * torch.exp(self.step_size * s) # #
            q = q / (q.sum(dim=1, keepdim=True) + self.eps)
            
            if return_all:
                qs.append(q)
                
        return (qs, q) if return_all else q
    
    def A_at(self, t: int):
        # A_t = A0 + sum_m beta[t,m] * A_basis[m]
        # returns (C, C+1)
        return self.A0 + torch.einsum("m,mcd->cd", self.beta[t], self.A_basis)
    
    @staticmethod
    def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        # KL(p || q) averaged over batch
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=1).mean()
    
    @staticmethod
    def euler_step(q, s, eta, eps=1e-12):
        # q: (N,C) on simplex
        mean = (q * s).sum(dim=1, keepdim=True)                             # <q,s>
        q = q + eta * q * s #already centerd no need for (s - mean)         # Euler
        q = torch.clamp(q, min=eps)
        q = q / (q.sum(dim=1, keepdim=True) + eps)
        return torch.clamp(q, min=eps, max=1.0)
    
    # def fit(self, val_loader, device="cuda"):
    #     self.to(device)
    #     self.train()

    #     optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     nll_criterion = nn.NLLLoss()

    #     for ep in range(self.max_iter):
    #         total = 0.0
    #         count = 0
    #         for batch in tqdm(val_loader, desc='calibration', leave=False):
    #             init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
    #             init_feats = init_feats.to(device)
    #             init_logits = init_logits.to(device)
    #             probs = F.softmax(init_logits, dim=1)
    #             labels = y_one_hot.to(device).argmax(dim=1)

    #             optimizer.zero_grad()
    #             qs, q = self.forward(probs, init_logits, init_feats, return_all=True)

    #             loss = nll_criterion(torch.log(q + self.eps), labels) # torch.zeros((), device=probs.device) #
                
    #             # for t, qt in enumerate(qs, start=1):
    #             #     w = 0.8 ** (self.n_steps - t) #t / len(qs)              # simple increasing weights
    #             #     last_step_value = nll_criterion(torch.log(qt + self.eps), labels)
    #             #     loss = loss + w * last_step_value #nll_criterion(torch.log(qt + self.eps), labels)
                    
    #             if self.kl_reg > 0:
    #                 loss = loss + self.kl_reg * self._kl(probs, q, eps=self.eps)
                
    #             loss.backward()
    #             optimizer.step()

    #             total += loss.item() * probs.size(0) # last_step_value.item() * probs.size(0)
    #             count += probs.size(0)

    #         print(f"Epoch {ep+1}/{self.max_iter} | NLL: {total/count:.4f}")

    #     self.eval()
    #     return self
    
    def fit(self, val_loader, device: str = "cuda"): 
        self.to(device) 
        probs_list, logits_list, feats_list, labels_list = [], [], [], [] 
        # Collect validation set outputs 
        with torch.no_grad(): 
            for batch in tqdm(val_loader, desc='accumulating'): 
            # Adjust depending on your dataloader format 
                init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch 
                init_feats = init_feats.to(device) 
                init_logits = init_logits.to(device) 
                probs = F.softmax(init_logits, dim=1) 
                y = y_one_hot.to(device).argmax(dim=1) 
            
                probs_list.append(probs) 
                feats_list.append(init_feats) 
                logits_list.append(init_logits) 
                labels_list.append(y) 
            
        probs = torch.cat(probs_list) # (N, C) 
        feats = torch.cat(feats_list) 
        logits = torch.cat(logits_list) # (N, C) 
        labels = torch.cat(labels_list) # (N,) 
    
        optimizer = optim.LBFGS([self.A_t], lr=self.lr, max_iter=self.max_iter) # self.A, self.W, self.U, self.log_eta
        nll_criterion = nn.NLLLoss() 
        
        def closure(): 
            optimizer.zero_grad() 
            q = self.forward(probs, logits, feats) 
            loss = nll_criterion(torch.log(q + self.eps), labels) 
    
            # Optional trust-region to stay close to base probs 
            if self.kl_reg > 0.0: 
                loss = loss + self.kl_reg * self._kl(probs, q, eps=self.eps) 
            loss.backward() 
            return loss 
      
        optimizer.step(closure) 
      
        with torch.no_grad(): 
            q = self.forward(probs, logits, feats) 
            loss = nll_criterion(torch.log(q + self.eps), labels) 
            if self.kl_reg > 0.0: 
                loss_total = loss + self.kl_reg * self._kl(probs, q, eps=self.eps) 
                print( f"Replicator calibration training done. " f"Final NLL: {loss.item():.4f} | Total (with KL): {loss_total.item():.4f}" ) 
            else: 
                print(f"Replicator calibration training done. Final NLL: {loss.item():.4f}") 
            return self

    
    def calibrated_predictions(self, batch, device="cuda"):
        """Return calibrated probabilities for a batch."""
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            init_feats = init_feats.to(device)
            init_logits = init_logits.to(device)
            probs = F.softmax(init_logits, dim=1)
            calibrated_probs = self.forward(probs, init_logits, init_feats)
            init_preds = init_preds.to(device)

            return {
                "features": init_pca.to(device),
                "logits": calibrated_probs,
                "preds": torch.argmax(calibrated_probs, dim=-1).view(-1, 1), 
                "true": torch.argmax(y_one_hot, dim=-1).view(-1, 1)
            }


# class BoostReplicatorCalibrator(nn.Module):
#     """
#     Replicator-flow calibrator on the simplex.

#     We learn a fitness field s(q) = A * [log(q), q, 1] (Dirichlet-style features),
#     then apply L replicator / mirror-descent (exponentiated-gradient) steps:

#         q^{l+1} ∝ q^{l} ⊙ exp(eta * s(q^{l}))

#     This keeps q on the simplex by construction and defines a nonlinear calibration map
#     via composition of steps.
#     """
#     def __init__(
#         self,
#         n_classes: int,
#         lr: float = 0.01,
#         max_iter: int = 500,
#         n_steps: int = 3,
#         step_size: float = 0.1,
#         eps: float = 1e-12,
#         # Optional trust-region regularizer weight; 0 disables
#         kl_reg: float = 0.0,
#         # If True, compute fitness from evolving q; if False, use initial probs only
#         state_dependent: bool = False,
#         feature_dim: int = 2048,
#         weight_decay: float = 0.0,
#         alpha: float = 0.05                     # weight for the auxiliary loss modelling residuals
#         # Optional: clamp fitness magnitude for stability (None disables)
#         # fitness_clip: float | None = None,
#     ):
#         super().__init__()
#         self.n_classes = n_classes
#         self.feature_dim = feature_dim 
        
#         #self.A = nn.Parameter(torch.zeros(self.n_classes, self.n_classes + 1))
#         # self.A_t = nn.Parameter(torch.zeros(n_steps, self.n_classes, self.n_classes + 1)) #+1
#         # nn.init.normal_(self.A_t, mean=0.0, std=1e-2)
#         self.W = nn.Parameter(torch.zeros(n_steps, self.n_classes, self.n_classes + 1)) #+1
#         self.U = nn.Parameter(torch.zeros(n_steps, self.n_classes, self.n_classes + 1)) #+1
#         nn.init.normal_(self.W, mean=0.0, std=1e-2)
#         nn.init.normal_(self.U, mean=0.0, std=1e-2)
#         # self.A_2 = nn.Parameter(torch.zeros(n_steps, self.n_classes, self.n_classes )) #+1                                        
#         # self.hidden = 2048
#         # self.gate = nn.Sequential(
#         #     nn.Linear(self.n_classes, self.hidden),
#         #     nn.ReLU(),
#         #     # nn.Linear(self.hidden, self.hidden),
#         #     # nn.ReLU(),
#         #     # nn.Linear(self.hidden, self.hidden),
#         #     # nn.ReLU(),
#         #     nn.Linear(self.hidden, self.n_classes)
#         # )        
#         # self.bias = nn.Parameter(torch.zeros(self.n_classes))
        
#         self.lr = lr
#         self.max_iter = max_iter
#         self.weight_decay = weight_decay

#         self.n_steps = n_steps
#         self.step_size = step_size 
#         # nn.init.normal_(self.step_size, mean=0., std=0.01)
#         self.log_eta = nn.Parameter(torch.log(torch.tensor([self.step_size] * self.n_steps))) #nn.Parameter(torch.log(torch.tensor([step_size]*n_steps)))
#         self.alpha = alpha
#         self.eps = eps
#         self.kl_reg = kl_reg
#         self.state_dependent = state_dependent
#         # self.fitness_clip = fitness_clip

#     def _features(self, probs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
#         # probs: (N, C), assumed in simplex
        
#         # IF WANT TO RUN ON LOG-PROBS USE:
#         log_probs = torch.log(probs + self.eps)           
#         # IF WANT TO RUN ON LOGITS USE:
#         # log_probs = logits 
        
#         ones = torch.ones(probs.shape[0], 1, device=probs.device, dtype=probs.dtype)
#         return torch.cat([log_probs, ones], dim=1)  # (N, C+1) #torch.cat([log_probs, probs, ones], dim=1)  # (N, 2C+1)
    
#     def gate_feats(self, q):
#         eps = self.eps
#         ent = -(q * torch.log(q + eps)).sum(dim=1, keepdim=True)
#         top2 = torch.topk(q, k=2, dim=1).values
#         maxp = top2[:, :1]
#         margin = (top2[:, :1] - top2[:, 1:2])
#         return torch.cat([ent, maxp, margin], dim=1)  # (N,3)

#     def fitness(self, probs: torch.Tensor, logits: torch.Tensor, features: torch.Tensor,  timestep: int = 0) -> torch.Tensor:
#         # Dirichlet-style linear fitness on features
#         feats = self._features(probs, logits)          # (N, 2C+1)
        
#         # TO USE SYMMETRIC (GRADIENT BASED MODELLING) RUN:
#         # self.W = torch.matmul(self.A.transpose(1,1), self.A_2) 
#         # s = feats @ self.W.t() + self.bias
        
#         # TO USE FEATURES RUN:
#         # s = features @ self.B.t() + self.bias               # (N, C) 
#         # s = features @ self.B_t[timestep].t() + self.bias
        
#         # TO USE PROBS RUN:
#         # s = feats @ self.A.t()                # (N, C) 
#         # NN PARAMETERIZATION INSTEAD:
#         # s = self.gate(feats)
        
#         # TO USE TIMESTEPS RUN:
#         # self.W_t = torch.matmul(self.A_t[timestep].transpose(1,1), self.A_2[timestep]) 
#         # s = feats @ self.A_t[timestep].t() # + self.bias[timestep] #self.W_t s
        
#         ones = torch.ones(feats.size(0), 1, device=feats.device, dtype=feats.dtype)
#         h = F.logsigmoid(feats @ self.W[timestep].t()) 
#         h_cat = torch.cat([h, ones], dim=1)
#         s = h_cat @ self.U[timestep].t() #h + 0.1 *( h_cat @ self.U[timestep].t() )
        
#         # A_t = self.A_at(t=timestep) 
#         # s = feats @ A_t.t()
        
#         # if self.fitness_clip is not None:
#         #     s = torch.clamp(s, -self.fitness_clip, self.fitness_clip)
#         return s

#     @torch.no_grad()
#     def _normalize_simplex(self, q: torch.Tensor) -> torch.Tensor:
#         # numerical safety: ensure nonneg and sum=1
#         q = torch.clamp(q, min=self.eps)
#         return q / q.sum(dim=1, keepdim=True)

#     def forward(self, probs: torch.Tensor, logits: torch.Tensor, features: torch.Tensor, return_all: bool = False) -> torch.Tensor:
#         """
#         probs: (N, C) uncalibrated probabilities (e.g., softmax of base logits)
#         returns: (N, C) calibrated probabilities after replicator flow
#         """
#         q_orig = probs
#         q = q_orig
#         ret = []
#         features = F.layer_norm(features, (self.feature_dim,))
#         if not self.state_dependent:
#             # Cache fitness computed from initial probs only
#             s0 = self.fitness(probs, logits, features)

#         for timestep in range(self.n_steps):
#             q_before = q 
#             s = self.fitness(q_orig, logits, features, timestep) if self.state_dependent else s0  #q_before                    
#             s = s - (q_before * s).sum(dim=1, keepdim=True)   #q_before
#             # s = 5.0 * torch.tanh(s / 5.0)                      
#             #s = torch.clamp(s, -10.0, 10.0)
#             eta = F.softplus(self.log_eta[timestep]) # 0.05 * torch.sigmoid(self.log_eta[timestep]) # self.step_size 
            
#             # EULER STEP (UNSTABLE):            
#             # q = self.euler_step(q_before, s, eta, self.eps) #self.euler_step(q, s, self.step_size, self.eps) #    
            
#             # Exponentiated-gradient / replicator step       
#             s = s - s.max(dim=1, keepdim=True).values    
#             s = s - (q_before * s).sum(dim=1, keepdim=True)   #q_before                          
#             q = q * torch.exp(eta * s) # torch.exp(F.softplus(self.log_eta) * s) # q * torch.exp(self.step_size * s) # #
#             q = q / (q.sum(dim=1, keepdim=True) + self.eps)
            
#             if return_all:
#                 tup = (q_before, q, eta, s)
#                 ret.append(tup)                
                
#         return (ret, q) if return_all else q
    
#     def A_at(self, t: int):
#         # A_t = A0 + sum_m beta[t,m] * A_basis[m]
#         # returns (C, C+1)
#         return self.A0 + torch.einsum("m,mcd->cd", self.beta[t], self.A_basis)
    
#     @staticmethod
#     def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#         # KL(p || q) averaged over batch
#         p = torch.clamp(p, min=eps)
#         q = torch.clamp(q, min=eps)
#         return (p * (torch.log(p) - torch.log(q))).sum(dim=1).mean()
    
#     @staticmethod
#     def euler_step(q, s, eta, eps=1e-12):
#         # q: (N,C) on simplex
#         mean = (q * s).sum(dim=1, keepdim=True)                             # <q,s>
#         q = q + eta * q * s #already centerd no need for (s - mean)         # Euler
#         q = torch.clamp(q, min=eps)
#         q = q / (q.sum(dim=1, keepdim=True) + eps)
#         return torch.clamp(q, min=eps, max=1.0)
    
#     @staticmethod
#     def cosine_sim(a, b, eps=1e-8):
#         a_norm = a / (a.norm(dim=1, keepdim=True) + eps)
#         b_norm = b / (b.norm(dim=1, keepdim=True) + eps)
#         return (a_norm * b_norm).sum(dim=1)
    
#     def fit(self, val_loader, device="cuda"):
#         self.to(device)
#         self.train()

#         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         nll_criterion = nn.NLLLoss()

#         for ep in range(self.max_iter):
            
#             total = 0.0
#             count = 0
#             # diagnostics
#             eta_max_epoch = 0.0
#             #eta_mean_epoch = 0.0
#             s_max_epoch = 0.0
#             q_min_epoch = 1.0
#             q_true_min_epoch = 1.0            
            
#             for batch in tqdm(val_loader, desc='calibration', leave=False):
#                 init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
#                 init_feats = init_feats.to(device)
#                 init_logits = init_logits.to(device)
#                 probs = F.softmax(init_logits, dim=1)
#                 one_hot_labels = y_one_hot.to(device)
#                 labels = one_hot_labels.argmax(dim=1)

#                 optimizer.zero_grad()
#                 ret, q = self.forward(probs, init_logits, init_feats, return_all=True)

#                 final_step_loss_value = nll_criterion(torch.log(q + self.eps), labels)
#                 loss = final_step_loss_value #torch.zeros((), device=probs.device # torch.zeros((), device=probs.device) #
                
#                 for t, tup in enumerate(ret, start=1):
#                     q_b, q_a, eta, fitness = tup 
                    
#                     #delta = q_a - q_b
#                     #residual = (one_hot_labels - q_b.detach())                    
#                     #target = eta * residual # residual 
#                     #aux_loss = aux_loss = F.mse_loss(delta, target)        
                           
#                     # inside your loop
#                     g = (one_hot_labels - q_b.detach())                   # (N,C) negative gradient direction
#                     g = g - (q_b.detach() * g).sum(dim=1, keepdim=True)   # project to tangent (optional but good)

#                     # normalize both to avoid scale games
#                     s = fitness
#                     s = s - (q_b.detach() * s).sum(dim=1, keepdim=True)

#                     loss_t = F.mse_loss(s, g) #F.mse_loss(one_hot_labels, q_a) # #(1 - self.cosine_sim(s, g)).mean()
#                     #loss = loss + self.alpha * aux_loss
                    
#                     loss = loss +  self.alpha * loss_t #  #+ loss_t #+ self.alpha * aux_loss #+ 0.1 * F.mse_loss(eta, torch.zeros_like(eta)) #nll_criterion(torch.log(qt + self.eps), labels)
                    
#                     eta_max_epoch = max(eta_max_epoch, eta.max().item())
#                     #eta_mean_epoch += eta.mean().item()

#                     s_max_epoch = max(s_max_epoch, fitness.abs().max().item())
                    
#                 q_min_epoch = min(q_min_epoch, q.min().item())

#                 q_true = q[torch.arange(q.size(0), device=q.device), labels]
#                 q_true_min_epoch = min(q_true_min_epoch, q_true.min().item())
                    
#                 if self.kl_reg > 0:
#                     loss = loss + self.kl_reg * self._kl(probs, q, eps=self.eps)
                
#                 loss.backward()
#                 optimizer.step()

#                 total += final_step_loss_value.item() * probs.size(0) # loss.item() * probs.size(0)
#                 count += probs.size(0)

#             print(
#                 f"Epoch {ep+1}/{self.max_iter} | "
#                 f"NLL: {total/count:.4f} | "
#                 f"eta_max: {eta_max_epoch:.4f} | "
#                 #f"eta_mean: {eta_mean_epoch:.4f} | "
#                 f"|s|_max: {s_max_epoch:.4f} | "
#                 f"q_min: {q_min_epoch:.6f} | "
#                 f"q_true_min: {q_true_min_epoch:.6f}"
#             )

#         self.eval()
#         return self
    
    # def fit(self, val_loader, device: str = "cuda"): 
    #     self.to(device) 
    #     probs_list, logits_list, feats_list, labels_list = [], [], [], [] 
    #     # Collect validation set outputs 
    #     with torch.no_grad(): 
    #         for batch in tqdm(val_loader, desc='accumulating'): 
    #         # Adjust depending on your dataloader format 
    #             init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch 
    #             init_feats = init_feats.to(device) 
    #             init_logits = init_logits.to(device) 
    #             probs = F.softmax(init_logits, dim=1) 
    #             y = y_one_hot.to(device).argmax(dim=1) 
            
    #             probs_list.append(probs) 
    #             feats_list.append(init_feats) 
    #             logits_list.append(init_logits) 
    #             labels_list.append(y) 
            
    #     probs = torch.cat(probs_list) # (N, C) 
    #     feats = torch.cat(feats_list) 
    #     logits = torch.cat(logits_list) # (N, C) 
    #     labels = torch.cat(labels_list) # (N,) 
    
    #     optimizer = optim.LBFGS([self.W, self.U, self.log_eta], lr=self.lr, max_iter=self.max_iter) # self.A, self.A_t, self.log_eta
    #     nll_criterion = nn.NLLLoss() 
        
    #     def closure(): 
    #         optimizer.zero_grad() 
    #         q = self.forward(probs, logits, feats) 
    #         loss = nll_criterion(torch.log(q + self.eps), labels) 
    
    #         # Optional trust-region to stay close to base probs 
    #         if self.kl_reg > 0.0: 
    #             loss = loss + self.kl_reg * self._kl(probs, q, eps=self.eps) 
    #         loss.backward() 
    #         return loss 
      
    #     optimizer.step(closure) 
      
    #     with torch.no_grad(): 
    #         q = self.forward(probs, logits, feats) 
    #         loss = nll_criterion(torch.log(q + self.eps), labels) 
    #         if self.kl_reg > 0.0: 
    #             loss_total = loss + self.kl_reg * self._kl(probs, q, eps=self.eps) 
    #             print( f"Replicator calibration training done. " f"Final NLL: {loss.item():.4f} | Total (with KL): {loss_total.item():.4f}" ) 
    #         else: 
    #             print(f"Replicator calibration training done. Final NLL: {loss.item():.4f}") 
    #         return self

    
    # def calibrated_predictions(self, batch, device="cuda"):
    #     """Return calibrated probabilities for a batch."""
    #     with torch.no_grad():
    #         init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
    #         init_feats = init_feats.to(device)
    #         init_logits = init_logits.to(device)
    #         probs = F.softmax(init_logits, dim=1)
    #         calibrated_probs = self.forward(probs, init_logits, init_feats)
    #         init_preds = init_preds.to(device)

    #         return {
    #             "features": init_pca.to(device),
    #             "logits": calibrated_probs,
    #             "preds": torch.argmax(calibrated_probs, dim=-1).view(-1, 1), 
    #             "true": torch.argmax(y_one_hot, dim=-1).view(-1, 1)
    #         }
    

class FitnessWeakLearner(nn.Module):
    """
    Weak learner h(q) -> s(q) in R^C.
    Default: linear map on Dirichlet-style features [log q, 1].
    """
    def __init__(self, n_classes: int, feature_dim: int, eps: float = 1e-12):
        super().__init__()
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.eps = eps
        #self.lin = nn.Linear(feature_dim, n_classes, bias=True) # nn.Linear(n_classes + 1, n_classes, bias=True)
        self.net = nn.Sequential(            
            nn.Linear(n_classes, feature_dim), # 2*
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, n_classes)
        )                
        # nn.init.zeros_(self.net[-1].weight)
        # nn.init.zeros_(self.net[-1].bias)
        # nn.init.zeros_(self.lin.weight)
        # nn.init.zeros_(self.lin.bias)

    def forward(self, q: torch.Tensor, logits: torch.Tensor):
        # q: (B,C) on simplex
        log_q = torch.log(q + self.eps)
        # log_q = log_q - log_q.mean(dim=1, keepdim=True)
        # log_q = log_q / (log_q.std(dim=1, keepdim=True) + 1e-6)
        #ones = torch.ones(q.size(0), 1, device=q.device, dtype=q.dtype)
        #feats = torch.cat([log_q, ones], dim=1)  # (B, C+1) #logits
        feats = log_q # (B, C+1) #logits
        return self.net(feats)  # (B,C) # lin
        # return self.lin(q) # q is last layer representations in this case !

class StepwiseReplicatorCalibrator(nn.Module):
    def __init__(
        self,
        n_classes: int,
        data: str,
        feature_dim: int,
        n_steps: int = 5,     # <-- number of steps == number of weak learners
        eta: float = 0.1,
        alpha: float = 1.0,   # optional shrinkage for each learner's output
        eps: float = 1e-12,
        kl_reg: float = 0.0,
        weak_epochs: int = 3,
        weak_lr: float = 5e-3,
        weight_decay: float = 0.0,
        finetune_epochs: int = 0,   # number of epochs for optional end-to-end fine-tuning after stagewise training
        finetune_grad_clip: float = 1.0,                
    ):
        super().__init__()
        self.n_classes = n_classes
        self.data = data
        self.feature_dim = feature_dim
        self.n_steps = n_steps
        self.eta = eta
        self.log_eta = nn.ParameterList([
            nn.Parameter(torch.tensor(math.log(eta), dtype=torch.float32)) for _ in range(n_steps)
        ])
        # self.log_eta = self.log_eta = nn.ParameterList([
        #     nn.Parameter(torch.full((n_classes,), math.log(eta), dtype=torch.float32)) for _ in range(n_steps)
        #    ]) #self.log_eta = nn.Parameter(torch.log(torch.tensor([eta] * n_steps))) 
        # self.bias = nn.ParameterList([
        #     nn.Parameter(torch.zeros(n_classes)) for _ in range(n_steps)
        # ])
        self.alpha = alpha
        self.eps = eps
        self.kl_reg = kl_reg
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = weak_lr * 0.1 # learning rate for optional end-to-end fine-tuning (default: smaller than weak learner LR)
        self.finetune_grad_clip = finetune_grad_clip
        
        self.weak_epochs = weak_epochs
        self.weak_lr = weak_lr
        self.weight_decay = weight_decay
        
        # one weak learner per step
        self.step_learners = nn.ModuleList([
            FitnessWeakLearner(n_classes, self.feature_dim, eps=eps) for _ in range(n_steps)
        ])

    @staticmethod
    def _kl(p, q, eps=1e-12):
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=1).mean()

    def _replicator_step(self, q, s, t):
        # project to tangent: s <- s - <q,s>
        s = s - (q * s).sum(dim=1, keepdim=True)        
        
        # s = s - s.max(dim=1, keepdim=True).values
        # q = q * torch.exp(self.eta * s) # torch.exp(F.softplus(self.log_eta[t]) * s) #        
        # q = q / (q.sum(dim=1, keepdim=True) + self.eps)
        
        # per-class positive step sizes
        eta_t = F.softplus(self.log_eta[t])      # shape: (C,)
        # expand for clarity (not strictly required)
        #eta_t = eta_t.unsqueeze(0)               # shape: (1, C)
        #bias_t = self.bias[t].unsqueeze(0)     # shape: (1, C)
        
        log_q = torch.log(q + self.eps) + eta_t * s # + bias_t # self.eta * s        
        q = F.softmax(log_q, dim=1)
        
        return q

    def forward(self, probs, logits):        
        q = probs        
        logits = logits            
        for t in range(self.n_steps):
            s = self.alpha * self.step_learners[t](q, logits) #q #feats
            q = self._replicator_step(q, s, t) #q        
        return q

    def fit(self, val_loader, test_loader, device="cuda"): #weak_epochs=3, weak_lr=5e-3, weight_decay=0.0):    
            
        self.to(device)
        nll = nn.NLLLoss()
        # ce = nn.CrossEntropyLoss()
        
        # Stagewise training
        for t in range(self.n_steps):
            all_probs = []
            all_labels = []      
            
            # freeze previous steps
            for j in range(t):
                for p in self.step_learners[j].parameters():
                    p.requires_grad_(False)
                self.step_learners[j].eval()

            # train only step t
            for p in self.step_learners[t].parameters():
                p.requires_grad_(True)
            self.step_learners[t].train()

            opt = optim.Adam(list(self.step_learners[t].parameters()) + [self.log_eta[t]], lr=self.weak_lr, weight_decay=self.weight_decay) #+ [self.bias[t]] #list(self.step_learners[t].parameters()) + [self.log_eta[t]] 

            for ep in range(self.weak_epochs):
                total, count = 0.0, 0
                for batch in tqdm(val_loader, desc=f"step {t+1}/{self.n_steps} ep {ep+1}/{self.weak_epochs}", leave=False):
                    init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
                    init_feats = init_feats.to(device)
                    init_logits = init_logits.to(device)
                    probs = F.softmax(init_logits, dim=1)                                        
                    labels = y_one_hot.to(device).argmax(dim=1)                    
                    
                    # build q up to step t-1 (no grad through frozen steps)
                    with torch.no_grad():                        
                        q = probs                        
                        for j in range(t):
                            s_j = self.alpha * self.step_learners[j](q, init_logits) #init_feats
                            q = self._replicator_step(q, s_j, j) #q

                    # now apply step t with grad
                    s_t = self.alpha * self.step_learners[t](q, init_logits) #init_feats
                    q_t = self._replicator_step(q, s_t, t) #q_t                    
                    if ep == self.weak_epochs - 1:
                        all_probs.append(q_t.detach().cpu())
                        all_labels.append(labels.cpu())

                    # loss = F.mse_loss(q_t, y_one_hot.to(device).float())      
                    loss = nll(torch.log(q_t + self.eps), labels)
                    if self.kl_reg > 0.0:
                        loss = loss + self.kl_reg * self._kl(probs, q_t, eps=self.eps)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    total += loss.item() * probs.size(0)
                    count += probs.size(0)

                print(f"[Step {t+1}] ep {ep+1}: NLL={total / max(count,1):.4f}")
                if ep == self.weak_epochs - 1:
                    all_probs = torch.cat(all_probs)
                    all_labels = torch.cat(all_labels)            

            # freeze this step after training
            self.step_learners[t].eval()
            for p in self.step_learners[t].parameters():
                p.requires_grad_(False)
            
            if self.weak_epochs > 0:
                save_path = f'results/plots/replicate{self.n_steps}_DEP_{self.data}_{self.n_classes}_classes_None_features'            
                full_path = os.path.join(save_path, 'in_training')
                os.makedirs(full_path, exist_ok=True)
                multiclass_calibration_plot(all_labels, all_probs, n_bins=15, bin_strategy='uniform',
                                            save_path=full_path, filename=f"{t}_multiclass_replicate_traincal_None.png")
                            
                with torch.no_grad():
                    all_test_probs = []
                    all_test_labels = []
                    for batch in tqdm(test_loader, desc=f"step {t+1}/{self.n_steps} ep {ep+1}/{self.weak_epochs}", leave=False):
                        init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
                        
                        init_feats = init_feats.to(device)
                        init_logits = init_logits.to(device)
                        probs = F.softmax(init_logits, dim=1)
                        
                        q = self.forward(probs, init_logits)# forward through this step only
                        all_test_probs.append(q.cpu())
                        
                        labels = y_one_hot.to(device).argmax(dim=1)
                        all_test_labels.append(labels.cpu())                                
                        
                    all_test_probs = torch.cat(all_test_probs)
                    all_test_labels = torch.cat(all_test_labels)

                    save_path = f'results/plots/replicate{self.n_steps}_DEP_{self.data}_{self.n_classes}_classes_None_features'            
                    full_path = os.path.join(save_path, 'in_training')
                    os.makedirs(full_path, exist_ok=True)
                    multiclass_calibration_plot(all_test_labels, all_test_probs, n_bins=15, bin_strategy='uniform',
                                                save_path=full_path, filename=f"{t}_multiclass_replicate_testcal_None.png")     
        # -------------------------
        # 2) Global fine-tuning (end-to-end)
        # -------------------------
        
        if self.finetune_epochs and self.finetune_epochs > 0:
            # unfreeze all steps
            for t in range(self.n_steps):
                self.step_learners[t].train()
                for p in self.step_learners[t].parameters():
                    p.requires_grad_(True)

            ft_wd = self.weight_decay if self.weight_decay is None else self.weight_decay
            ft_kl = self.kl_reg if self.kl_reg is None else self.kl_reg

            opt_ft = optim.Adam(
                list(p for t in range(self.n_steps) for p in self.step_learners[t].parameters()) + [self.log_eta[t] for t in range(self.n_steps)], #+ [self.bias[t] for t in range(self.n_steps)]
                lr=self.finetune_lr,
                weight_decay=ft_wd,
            )

            for ep in range(self.finetune_epochs):
                total, count = 0.0, 0
                for batch in tqdm(val_loader, desc=f"finetune ep {ep+1}/{self.finetune_epochs}", leave=False):
                    init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
                    init_feats = init_feats.to(device)
                    init_logits = init_logits.to(device)
                    probs = F.softmax(init_logits, dim=1)
                    labels = y_one_hot.to(device).argmax(dim=1)

                    # full forward through all steps with gradients
                    q = self.forward(probs, init_logits)

                    # brier = F.mse_loss(q, y_one_hot.to(device).float())                    
                    loss = nll(torch.log(q + self.eps), labels) #+ 0.1 * brier
                    if ft_kl and ft_kl > 0.0:
                        loss = loss + ft_kl * self._kl(probs, q, eps=self.eps)

                    opt_ft.zero_grad()
                    loss.backward()

                    if self.finetune_grad_clip is not None and self.finetune_grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.parameters(), self.finetune_grad_clip)

                    opt_ft.step()

                    total += loss.item() * probs.size(0)
                    count += probs.size(0)

                print(f"[Finetune] ep {ep+1}: NLL={total / max(count,1):.4f}")

                with torch.no_grad():
                    all_test_probs = []
                    all_test_labels = []
                    for batch in tqdm(test_loader, desc=f"step {t+1}/{self.n_steps} ep {ep+1}/{self.weak_epochs}", leave=False):
                        init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
                        
                        init_feats = init_feats.to(device)
                        init_logits = init_logits.to(device)
                        probs = F.softmax(init_logits, dim=1)
                        
                        q = self.forward(probs, init_logits)# forward through this step only
                        all_test_probs.append(q.cpu())
                        
                        labels = y_one_hot.to(device).argmax(dim=1)
                        all_test_labels.append(labels.cpu())                                
                        
                    all_test_probs = torch.cat(all_test_probs)
                    all_test_labels = torch.cat(all_test_labels)

                    save_path = f'results/plots/replicate{self.n_steps}_DEP_{self.data}_{self.n_classes}_classes_None_features'            
                    full_path = os.path.join(save_path, 'finetuning')
                    os.makedirs(full_path, exist_ok=True)
                    multiclass_calibration_plot(all_test_labels, all_test_probs, n_bins=15, bin_strategy='uniform',
                                                save_path=full_path, filename=f"{ep}_multiclass_replicate_testcal_None.png")     
            # set eval
            for t in range(self.n_steps):
                self.step_learners[t].eval()
                for p in self.step_learners[t].parameters():
                    p.requires_grad_(False)
                    
        self.eval()
        return self
    
    def calibrated_predictions(self, batch, device="cuda"):
        """
        Return calibrated probabilities for a batch.
        Uses learned replicator steps only (no supervision needed).
        """
        self.eval()
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch

            init_feats = init_feats.to(device)
            init_logits = init_logits.to(device)
            probs = F.softmax(init_logits, dim=1)

            # Run stepwise replicator flow
            calibrated_probs = self.forward(probs, init_logits)  # (B, C)
            
            return { 
                    "features": init_pca.to(device), 
                    "logits": calibrated_probs, 
                    "preds": torch.argmax(calibrated_probs, dim=-1).view(-1, 1), 
                    "true": torch.argmax(y_one_hot, dim=-1).view(-1, 1) 
                    }
            
            
# class FitnessWeakLearner(nn.Module):
#     """
#     Weak learner h(q) -> s(q) in R^C.
#     Default: linear map on Dirichlet-style features [log q, 1].
#     """
#     def __init__(self, n_classes: int, eps: float = 1e-12):
#         super().__init__()
#         self.n_classes = n_classes        
#         self.eps = eps
#         self.lin = nn.Linear(n_classes + 1, n_classes, bias=True) 
#         nn.init.zeros_(self.lin.weight)
#         nn.init.zeros_(self.lin.bias)

#     def forward(self, q: torch.Tensor) -> torch.Tensor:
#         # q: (B,C) on simplex
#         log_q = torch.log(q + self.eps)
#         ones = torch.ones(q.size(0), 1, device=q.device, dtype=q.dtype)
#         feats = torch.cat([log_q, ones], dim=1)  # (B, C+1)
#         # return self.net(feats)  # (B,C) # lin
#         return self.lin(feats) # q is last layer representations in this case !


class LinearLayer(nn.Module):
    """
    Dirichlet / matrix-scaling calibrator:
        log_q = log_softmax( W * log(p) + b )

    Input:  probs p  of shape (B, C) on simplex
    Output: log_q of shape (B, C) (log-probabilities), suitable for NLLLoss
    """
    def __init__(self, n_classes: int, eps: float = 1e-12, init_identity: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.eps = eps
        self.W = nn.Parameter(torch.zeros(n_classes, n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

        if init_identity:
            # Start close to identity map in log-space (a strong, stable default)
            with torch.no_grad():
                self.W.copy_(torch.eye(n_classes))

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        # probs: (B, C), assumed >=0 and sum to 1
        log_p = torch.log(probs.clamp_min(self.eps))        # (B, C)
        logits = log_p @ self.W.t() + self.b               # (B, C)
        log_q = F.log_softmax(logits, dim=1)               # (B, C)
        return log_q

class DirCalibrator(nn.Module):
    def __init__(
        self,
        n_classes: int,
        data: str,                        
        eps: float = 1e-12,
        kl_reg: float = 0.0,
        weak_epochs: int = 3,
        weak_lr: float = 5e-3,
        weight_decay: float = 0.0,             
    ):
        super().__init__()
        self.name = 'DIRICHLET'
        self.n_classes = n_classes                                
        self.data = data
        self.eps = eps
        self.kl_reg = kl_reg                                
        
        self.weak_epochs = weak_epochs
        self.weak_lr = weak_lr
        self.weight_decay = weight_decay
        
        # one weak learner per step
        self.step_learners = nn.ModuleList([
            LinearLayer(n_classes, eps=eps) 
        ])

    @staticmethod
    def _kl(p, q, eps=1e-12):
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=1).mean()    

    def forward(self, probs, feats):        
        q = probs                
        log_q = self.step_learners[0](q) #feats            
        return log_q

    # def fit(self, val_loader, test_loader, device="cuda"): #weak_epochs=3, weak_lr=5e-3, weight_decay=0.0):
    #     self.to(device)
    #     nll = nn.NLLLoss()  
    #     ce = nn.CrossEntropyLoss()                

    #     # train only step t
    #     for p in self.step_learners[0].parameters():
    #         p.requires_grad_(True)
    #     self.step_learners[0].train()

    #     opt = optim.Adam(self.step_learners[0].parameters(), lr=self.weak_lr, weight_decay=self.weight_decay) #list(self.step_learners[t].parameters()) + [self.log_eta[t]]

    #     for ep in range(self.weak_epochs):
    #         total, count = 0.0, 0
    #         for batch in tqdm(val_loader, desc=f"step {0+1}/{1} ep {ep+1}/{self.weak_epochs}", leave=False):
    #             init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
    #             init_feats = init_feats.to(device)
    #             init_logits = init_logits.to(device)
    #             probs = F.softmax(init_logits, dim=1)
    #             labels = y_one_hot.to(device).argmax(dim=1)

    #             # now apply step t with grad
    #             log_q = self.step_learners[0](probs) #init_feats                    

    #             # loss = F.mse_loss(q_t, y_one_hot.to(device).float())                    
    #             loss = nll(log_q, labels)                

    #             opt.zero_grad()
    #             loss.backward()
    #             opt.step()

    #             total += loss.item() * probs.size(0)
    #             count += probs.size(0)

    #         print(f"[Step {0+1}] ep {ep+1}: NLL={total / max(count,1):.4f}")
            
    #         if ep % 5 == 0:
    #             with torch.no_grad():
    #                 all_test_probs = []
    #                 all_test_labels = []
    #                 for batch in tqdm(test_loader, desc=f"epoch {ep+1}/{self.weak_epochs} ep {ep+1}/{self.weak_epochs}", leave=False):
    #                     init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
    #                     init_feats = init_feats.to(device)
    #                     init_logits = init_logits.to(device)
    #                     probs = F.softmax(init_logits, dim=1)
    #                     labels = y_one_hot.to(device).argmax(dim=1)

    #                     # now apply step t with grad
    #                     log_q = self.step_learners[0](probs) #init_feats    
    #                     q = F.softmax(log_q, dim=1)
                                            
    #                     all_test_probs.append(q.cpu())
                        
    #                     labels = y_one_hot.to(device).argmax(dim=1)
    #                     all_test_labels.append(labels.cpu())                                
                        
    #                 all_test_probs = torch.cat(all_test_probs)
    #                 all_test_labels = torch.cat(all_test_labels)

    #                 full_path = f'results/plots/replicateDC{self.weak_epochs}_potential_{self.data}_{self.n_classes}_classes_None_features'            
    #                 full_path = os.path.join(full_path, 'in_training')
    #                 full_path = os.path.join(full_path, 'joint')
    #                 os.makedirs(full_path, exist_ok=True)
    #                 multiclass_calibration_plot(all_test_labels, all_test_probs, n_bins=15, bin_strategy='uniform',
    #                                             save_path=full_path, filename=f"{ep}_multiclass_replicate_testcal_None.png")

    #     # freeze this step after training
    #     self.step_learners[0].eval()
    #     for p in self.step_learners[0].parameters():
    #         p.requires_grad_(False)                    
                
    #     self.eval()
    #     return self
        
    def fit(self, val_loader, device: str = "cuda"): 
        self.to(device) 
        probs_list, logits_list, feats_list, labels_list = [], [], [], [] 
        # Collect validation set outputs 
        with torch.no_grad(): 
            for batch in tqdm(val_loader, desc='accumulating'): 
            # Adjust depending on your dataloader format 
                init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch 
                init_feats = init_feats.to(device) 
                init_logits = init_logits.to(device) 
                probs = F.softmax(init_logits, dim=1) 
                y = y_one_hot.to(device).argmax(dim=1) 
            
                probs_list.append(probs) 
                feats_list.append(init_feats) 
                logits_list.append(init_logits) 
                labels_list.append(y) 
            
        probs = torch.cat(probs_list) # (N, C) 
        feats = torch.cat(feats_list) 
        logits = torch.cat(logits_list) # (N, C) 
        labels = torch.cat(labels_list) # (N,) 
    
        optimizer = optim.LBFGS([self.step_learners[0].W, self.step_learners[0].b], lr=self.weak_lr, max_iter=self.weak_epochs) # self.A, self.W, self.U, self.log_eta
        nll_criterion = nn.NLLLoss() 
        
        def closure(): 
            optimizer.zero_grad() 
            q = F.softmax(self.forward(probs, feats), dim=1)
            loss = nll_criterion(torch.log(q + self.eps), labels) 
    
            # Optional trust-region to stay close to base probs 
            if self.kl_reg > 0.0: 
                loss = loss + self.kl_reg * self._kl(probs, q, eps=self.eps) 
            loss.backward() 
            return loss 
        
        optimizer.step(closure) 
      
        with torch.no_grad(): 
            q = F.softmax(self.forward(probs, feats), dim=1)
            loss = nll_criterion(torch.log(q + self.eps), labels) 
            if self.kl_reg > 0.0: 
                loss_total = loss + self.kl_reg * self._kl(probs, q, eps=self.eps) 
                print( f"Dirichlet calibration training done. " f"Final NLL: {loss.item():.4f} | Total (with KL): {loss_total.item():.4f}" ) 
            else: 
                print(f"Dirichlet calibration training done. Final NLL: {loss.item():.4f}") 
            return self
    
    def calibrated_predictions(self, batch, device="cuda"):
        """
        Return calibrated probabilities for a batch.
        Uses learned replicator steps only (no supervision needed).
        """
        self.eval()
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch

            init_feats = init_feats.to(device)
            init_logits = init_logits.to(device)
            probs = F.softmax(init_logits, dim=1)

            # Run stepwise replicator flow
            calibrated_probs = F.softmax(self.forward(probs, init_feats), dim=1)  # (B, C)
                        
            return { 
                    "features": init_pca.to(device), 
                    "logits": calibrated_probs, 
                    "preds": torch.argmax(calibrated_probs, dim=-1).view(-1, 1), 
                    "true": torch.argmax(y_one_hot, dim=-1).view(-1, 1) 
                    }
                               
class FitnessNet(nn.Module):  
    def __init__(self, n_classes: int, hidden: int = 64, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(n_classes, hidden), # + 1 * 2
            nn.SiLU(),            
            nn.Linear(hidden, n_classes)  # scalar potential
        ) 
        
    def forward(self, x_feat: torch.Tensor, q: torch.Tensor):
        # q: (B, C) on simplex
        log_q = torch.log(q + self.eps)
        ones = torch.ones(q.size(0), 1, device=q.device, dtype=q.dtype)
        inp = torch.cat([log_q, ones], dim=1) # 
        # inp = torch.cat([log_q, x_feat], dim=1)      # (B, x_dim + C)
        return self.net(log_q).squeeze(-1)            #inp # (B,)
        
class PotentialNet(nn.Module):  
    def __init__(self, n_classes: int, hidden: int = 64, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(n_classes, hidden), # + 1 * 2
            nn.SiLU(),            
            # nn.Linear(hidden, hidden),
            # nn.SiLU(),
            # nn.Linear(hidden, hidden),
            # nn.SiLU(),
            # nn.Linear(hidden, hidden),
            # nn.SiLU(),
            # nn.Linear(hidden, hidden),
            # nn.SiLU(),
            # nn.Linear(hidden, hidden),
            # nn.SiLU(),
            # nn.Linear(hidden, hidden),
            # nn.SiLU(),
            # nn.Linear(hidden, hidden),
            # nn.SiLU(),
            # nn.Linear(hidden, hidden),
            # nn.SiLU(),
            # nn.Linear(hidden, hidden),
            # nn.SiLU(),
            nn.Linear(hidden, 1)  # scalar potential
        )

    def forward(self, x_feat: torch.Tensor, q: torch.Tensor):
        # q: (B, C) on simplex
        log_q = torch.log(q + self.eps)
        ones = torch.ones(q.size(0), 1, device=q.device, dtype=q.dtype)
        inp = torch.cat([log_q, ones], dim=1) # 
        # inp = torch.cat([log_q, x_feat], dim=1)      # (B, x_dim + C)
        return self.net(log_q).squeeze(-1)            #inp # (B,)


def fitness_from_potential(Ftheta, x_feat, q, eps=1e-12, create_graph=True, detach_q=False):
    with torch.enable_grad():
        log_q = torch.log(q + eps)
        if detach_q:
            log_q = log_q.detach()
        log_q = log_q.requires_grad_(True)          

        q_cur = torch.softmax(log_q, dim=1)
        F_val = Ftheta(x_feat, q_cur)  # (B,)

        grad_logq = torch.autograd.grad(
            outputs=F_val.sum(),
            inputs=log_q,
            create_graph=create_graph,
            retain_graph=create_graph,
            only_inputs=True
        )[0]

        # centering (optional)
        # grad_logq = grad_logq - (q_cur * grad_logq).sum(dim=1, keepdim=True)
        return grad_logq, q_cur


class PotentialReplicatorCalibrator(nn.Module):
    def __init__(self, n_classes, data='cifar10', n_steps=5, eta=0.1, hidden=64, lin_comb=0.9, ceiling=0.1, kl_reg=0.0, l2_reg=0.0, lr=1e-3, weight_decay=0.0, epochs=5, fit_stage=False, potential=True, eps=1e-12, detach_q_during_train=False):
        super().__init__()
        self.name = 'REPLICATOR'
        self.n_classes = n_classes               
        self.data = data        
        self.n_steps = n_steps 
        self.eps = eps
        self.detach_q_during_train = detach_q_during_train
        self.kl_reg = kl_reg
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.fit_stage = fit_stage  # whether to do stagewise training (default: False, i.e. train all steps end-to-end from the start)        
        self.potential = potential
        self.l2_reg = l2_reg
        self.lin_comb = lin_comb
        if self.lin_comb > 1:
            raise ValueError("linear combination weight should be in [0,1]. Instead {} was given!".format(self.lin_comb))
        self.ceiling = ceiling
        if self.ceiling > 1:
            raise ValueError("ceiling weight should be in [0,1]. Instead {} was given!".format(self.ceiling))

        self.init_eta = eta
        # self.log_eta = nn.Parameter(torch.tensor(torch.log(torch.tensor([eta]))))        
        self.log_eta = nn.ParameterList([
        nn.Parameter(torch.full((1,), math.log(eta), dtype=torch.float32)) for _ in range(n_steps) ]) #nn.Parameter(torch.tensor(torch.log(torch.tensor([eta] * self.n_steps)), dtype=torch.float32))
        # self.log_eta = nn.ParameterList([
        #     nn.Parameter(torch.full((n_classes,), math.log(eta), dtype=torch.float32)) for _ in range(n_steps)
        #     ])
        #self.Ftheta = PotentialNet(n_classes=n_classes, hidden=hidden, eps=eps)
        if self.potential:
            self.Ftheta = nn.ModuleList([PotentialNet(n_classes=n_classes, hidden=hidden, eps=eps) for t in range(n_steps)])
        else:
            self.Ftheta = nn.ModuleList([FitnessNet(n_classes=n_classes, hidden=hidden, eps=eps) for t in range(n_steps)])

    def replicator_step(self, q, s, eta, eps=1e-12):
        # eta = eta.unsqueeze(0) 
        # log_q = torch.log(q + eps) + eta * s
        # ret = F.softmax(log_q, dim=1)
        
        init_q = q
        
        s = s - (q * s).sum(dim=1, keepdim=True)        
            
        #s = s - s.max(dim=1, keepdim=True).values
        z = eta * s
        
        z = z - z.max(dim=1, keepdim=True).values
        
        q = q * torch.exp(z) # torch.exp(F.softplus(self.log_eta[t]) * s) #        
        q = q / (q.sum(dim=1, keepdim=True) + eps)
        ret = (1 - self.lin_comb) * q + self.lin_comb * init_q
        return ret
    
    def forward(self, p, x_feat):
        q = p
        #eta = F.softplus(self.log_eta)

        for t in range(self.n_steps):
            eta_t = self.init_eta + self.ceiling * torch.sigmoid(self.log_eta[t]) #F.softplus(self.log_eta[t]) # self.init_eta + 0.1 * torch.sigmoid(self.log_eta[t]) # # F.sigmoid(F.softplus(self.log_eta[t])) #F.softplus(self.log_eta)
            if self.potential:
                s, q_consistent = fitness_from_potential(
                    self.Ftheta[t], x_feat, q,
                    eps=self.eps,
                    create_graph=self.training,
                    detach_q=self.detach_q_during_train            
                )
            else:
                s = self.Ftheta[t](x_feat, q) # (B,) 
                q_consistent = q
                
            q = self.replicator_step(q_consistent, s, eta_t, eps=self.eps)

        return q

    @torch.no_grad()
    def _apply_steps_no_grad(self, p, x_feat, upto_t_exclusive: int):
        """Apply steps [0, upto_t_exclusive) with no grad (for building q_{t})."""
        q = p
        for j in range(upto_t_exclusive):
            eta_j = F.softplus(self.log_eta[j])
            # IMPORTANT: even though we are in no_grad, fitness_from_potential enables grad internally.
            # To truly keep no grad through previous steps, we compute s_j with create_graph=False and detach_q=True.
            s_j, q_consistent = fitness_from_potential(
                self.Ftheta[j], x_feat, q,
                eps=self.eps,
                create_graph=False,
                detach_q=True
            )
            q = self.replicator_step(q_consistent, s_j, eta_j, eps=self.eps)
        return q
    
    def l2_penalty(self):
        l2 = 0.0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                l2 = l2 + module.weight.pow(2).sum()
        return l2.mean()

    def fit(self, cal_loader, val_loader, test_loader, optuna_=False, trial=None, device="cuda"):
        if isinstance(device, int):
            device = f"cuda:{device}"
        elif isinstance(device, torch.device):
            device = str(device)
        self.to(device)
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.best_val_nll = float("inf")
        full_path = f'checkpoints/replicate{self.n_steps}_{self.data}_{self.n_classes}_classes_None_features'   
        os.makedirs(full_path, exist_ok=True)
        ckpt_name = "best_model.pt"
        ckpt_path = os.path.join(full_path, ckpt_name)   
        
        ######## TRAINING LOOP ########
        
        for ep in range(self.epochs):
            total, n = 0.0, 0
            for batch in tqdm(cal_loader, desc=f"epoch {ep+1}/{self.epochs}", leave=False):
                init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
                logits = init_logits.to(device)
                y = torch.argmax(y_one_hot, dim=1).to(device)

                p = F.softmax(logits, dim=1)
                x_feat = logits  # simplest choice; or init_feats/init_pca

                q = self.forward(p, x_feat)
                loss = F.nll_loss(torch.log(q + self.eps), y)
                if self.kl_reg > 0.0:
                    kl = (q * (torch.log(q + self.eps) - torch.log(p + self.eps))).sum(dim=1).mean()
                    loss = loss + self.kl_reg * kl
                if self.l2_reg > 0.0:
                    loss = loss + self.l2_reg * self.l2_penalty()

                opt.zero_grad()
                loss.backward()
                opt.step()

                total += loss.item() * logits.size(0)
                n += logits.size(0)

            print(f"ep {ep+1}/{self.epochs}: NLL={total/max(n,1):.4f}")
            
            ######## TEST LOOP ########
            if not optuna_:
                with torch.no_grad():
                    all_test_probs = []
                    all_test_labels = []
                    for batch in tqdm(test_loader, desc=f"epoch {ep+1}/{self.epochs}", leave=False):
                        init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
                        
                        logits = init_logits.to(device)
                        y = torch.argmax(y_one_hot, dim=1).to(device)

                        p = F.softmax(logits, dim=1)
                        x_feat = logits  # simplest choice; or init_feats/init_pca
                        q = self.forward(p, x_feat)
                        
                        all_test_probs.append(q.cpu())
                        
                        labels = y_one_hot.to(device).argmax(dim=1)
                        all_test_labels.append(labels.cpu())                                
                        
                    all_test_probs = torch.cat(all_test_probs)
                    all_test_labels = torch.cat(all_test_labels)
                    
                    nll_loss = torch.nn.NLLLoss()
                    nll = nll_loss(torch.log(all_test_probs + 1e-12), all_test_labels)
                    print("Test NLL:", nll.item())
                    
                    full_path = f'results/plots/replicate{self.n_steps}_potential_{self.data}_{self.n_classes}_classes_None_features'            
                    full_path = os.path.join(full_path, 'in_training')
                    full_path = os.path.join(full_path, 'joint')
                    os.makedirs(full_path, exist_ok=True)
                    multiclass_calibration_plot(all_test_labels, all_test_probs, n_bins=15, bin_strategy='uniform',
                                                save_path=full_path, filename=f"{ep}_multiclass_replicate_testcal_None.png")    
            
            ######## VAL LOOP ########                             
            with torch.no_grad():
                all_test_probs = []
                all_test_labels = []
                for batch in tqdm(val_loader, desc=f"epoch {ep+1}/{self.epochs}", leave=False):
                    init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
                    
                    logits = init_logits.to(device)
                    y = torch.argmax(y_one_hot, dim=1).to(device)

                    p = F.softmax(logits, dim=1)
                    x_feat = logits  # simplest choice; or init_feats/init_pca
                    q = self.forward(p, x_feat)
                    
                    all_test_probs.append(q.cpu())
                    
                    labels = y_one_hot.to(device).argmax(dim=1)
                    all_test_labels.append(labels.cpu())                                
                    
                all_test_probs = torch.cat(all_test_probs)
                all_test_labels = torch.cat(all_test_labels)
                
                nll_loss = torch.nn.NLLLoss()
                nll = nll_loss(torch.log(all_test_probs + 1e-12), all_test_labels)
                print("Val NLL:", nll.item())
                
                val_value = nll.item()

                # ---- OPTUNA REPORT + PRUNE ----
                if optuna_ and trial is not None:
                    trial.report(val_value, step=ep)
                    if trial.should_prune():
                        print(f"Trial pruned at epoch {ep+1} (Val NLL={val_value:.4f})")
                        raise optuna.TrialPruned()
            
            if nll.item() < self.best_val_nll:
                self.best_val_nll = nll.item()                    
                if not optuna_:
                    # -------- SAVE BEST --------                
                        torch.save({
                            "model_state_dict": self.state_dict(),
                            "epoch": ep,
                            "val_nll": self.best_val_nll
                        }, ckpt_path)

                        print(f"Saved new best model at epoch {ep+1} (Val NLL={self.best_val_nll:.4f})")       
                                             
        if not optuna_:
            # -------- LOAD BEST MODEL AT END --------
            if os.path.exists(ckpt_path):            
                checkpoint = torch.load(ckpt_path, map_location=device)
                self.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded best model from epoch {checkpoint['epoch']+1} (Val NLL={checkpoint['val_nll']:.4f})")

        self.eval()
        return self
        
        
    def fit_stagewise(self, val_loader, test_loader, device="cuda", weight_decay=0.0, grad_clip=None):
        """
        Stagewise training:
        for t in 0..T-1:
        freeze previous steps
        train only (Ftheta[t], log_eta[t]) using NLL (+ optional regs) on q_{t+1}
        """
        self.to(device)
        self.train()

        for t in range(self.n_steps):
            #---- freeze all steps ----
            for j in range(self.n_steps):
                for p in self.Ftheta[j].parameters():
                    p.requires_grad_(j == t)
                self.log_eta[j].requires_grad_(j == t)
                self.Ftheta[j].eval()

            opt = torch.optim.Adam(
                list(self.Ftheta[t].parameters()) + [self.log_eta[t]],
                lr=self.lr,
                weight_decay=weight_decay
            )

            for ep in range(self.epochs):
                total, n = 0.0, 0

                for batch in tqdm(val_loader, desc=f"stage {t+1}/{self.n_steps} ep {ep+1}/{self.epochs}", leave=False):
                    init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
                    logits = init_logits.to(device)
                    y = torch.argmax(y_one_hot, dim=1).to(device)
                    y_oh = y_one_hot.to(device).float()

                    p0 = F.softmax(logits, dim=1)
                    x_feat = logits  # or init_feats/init_pca

                    # ---- build q_t from previous frozen steps (no grad) ----
                    if t == 0:
                        q_t = p0
                    else:
                        q_t = self._apply_steps_no_grad(p0, x_feat, upto_t_exclusive=t)

                    # ---- apply step t with grad ----
                    eta_t = F.softplus(self.log_eta[t])
                    s_t, q_consistent = fitness_from_potential(
                        self.Ftheta[t], x_feat, q_t,
                        eps=self.eps,
                        create_graph=True,
                        detach_q=self.detach_q_during_train
                    )
                    q_tp1 = self.replicator_step(q_consistent, s_t, eta_t, eps=self.eps)

                    # ---- losses ----
                    nll = F.nll_loss(torch.log(q_tp1 + self.eps), y)

                    loss = nll

                    # if self.brier_reg and self.brier_reg > 0:
                    brier = (q_tp1 - y_oh).pow(2).sum(dim=1).mean()
                    loss = loss #+ brier

                    if self.kl_reg and self.kl_reg > 0:
                        # choose one direction; KL(p||q) often stabilizes calibration
                        kl_pq = (p0 * (torch.log(p0 + self.eps) - torch.log(q_tp1 + self.eps))).sum(dim=1).mean()
                        loss = loss + self.kl_reg * kl_pq

                    opt.zero_grad()
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        nn.utils.clip_grad_norm_(list(self.Ftheta[t].parameters()) + [self.log_eta[t]], grad_clip)
                    opt.step()

                    total += nll.item() * logits.size(0)
                    n += logits.size(0)

                print(f"[Stage {t+1}/{self.n_steps}] ep {ep+1}/{self.epochs}: NLL={total/max(n,1):.4f}")

            # after stage t, set it to eval mode (optional)
            self.Ftheta[t].eval()
            
            with torch.no_grad():
                all_test_probs = []
                all_test_labels = []
                for batch in tqdm(test_loader, desc=f"stage {t+1}/{self.n_steps} ep {ep+1}/{self.epochs}", leave=False):
                    init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
                    logits = init_logits.to(device)
                    y = torch.argmax(y_one_hot, dim=1).to(device)
                    y_oh = y_one_hot.to(device).float()

                    p0 = F.softmax(logits, dim=1)
                    x_feat = logits  # or init_feats/init_pca

                    # ---- build q_t from previous frozen steps (no grad) ----
                    if t == 0:
                        q_t = p0
                    else:
                        q_t = self._apply_steps_no_grad(p0, x_feat, upto_t_exclusive=t+1)
                    
                    # eta_t = F.softplus(self.log_eta[t]) 
                    # q_tp1 = replicator_step(q_t, s_t, eta_t, eps=self.eps)
                    
                    all_test_probs.append(q_t.cpu())                                        
                    all_test_labels.append(y.cpu())                                
                    
                all_test_probs = torch.cat(all_test_probs)
                all_test_labels = torch.cat(all_test_labels)

                full_path = f'results/plots/replicate{self.n_steps}_potential_{self.data}_{self.n_classes}_classes_None_features'            
                full_path = os.path.join(full_path, 'in_training')
                full_path = os.path.join(full_path, 'boosting')
                os.makedirs(full_path, exist_ok=True)
                multiclass_calibration_plot(all_test_labels, all_test_probs, n_bins=15, bin_strategy='uniform',
                                            save_path=full_path, filename=f"{t}_multiclass_replicate_testcal_None.png")     

        # finally freeze everything and set eval
        for j in range(self.n_steps):
            for p in self.Ftheta[j].parameters():
                p.requires_grad_(False)
            self.log_eta[j].requires_grad_(False)
            self.Ftheta[j].eval()

        self.eval()
        return self
        
    def calibrated_predictions(self, batch, device="cuda"):
        """
        Return calibrated probabilities for a batch.
        Uses learned replicator steps only (no supervision needed).
        """
        self.eval()
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            
            logits = init_logits.to(device)
            y = torch.argmax(y_one_hot, dim=1).to(device)
            y_oh = y_one_hot.to(device).float()

            p = F.softmax(logits, dim=1)
            x_feat = logits  # simplest choice; or init_feats/init_pca
            
            if self.fit_stage:
                # ---- build q_t from previous frozen steps (no grad) ----
                calibrated_probs = self._apply_steps_no_grad(p, x_feat, upto_t_exclusive=self.n_steps)
            else:                
                calibrated_probs = self.forward(p, x_feat)                                

            return { 
                    "features": init_pca.to(device), 
                    "logits": calibrated_probs, 
                    "preds": torch.argmax(calibrated_probs, dim=-1).view(-1, 1), 
                    "true": torch.argmax(y_one_hot, dim=-1).view(-1, 1) 
                    }
            
            