import torch
#from tqdm import tqdm
from tqdm.notebook import trange, tqdm
import numpy as np
from torch.distributions import MultivariateNormal as MVN
import math
import pdb
import matplotlib.pyplot as plt
 

torch.autograd.set_detect_anomaly(False)

class var_triple:

    def __init__(self,args):

        self.observations = args['observations'] 
        self.T = self.observations.shape[1]
        self.ground_truth = args['ground_truth']
        self.learning_rate = args['learning_rate']
        #print(self.learning_rate)
        #HMM parameters
        self.F = args['F']
        self.F.requires_grad = True
        self.H = args['H'] 
        self.H.requires_grad = False
        #self.H.double()
        #print(self.H)
        self.Q = args['Q'] 
        #self.Q.requires_grad = False
        self.R = args['R']
        #self.R.requires_grad = False
        #pr_r_1 need not be softmaxed
        self.pr_r_1 = args['pr_r_1'] 
        self.pr_r_1.requires_grad = True
        self.pr_0 = args['pr_0']
        
        self.inv_Q = torch.inverse(self.Q)
        self.inv_R = torch.inverse(self.R)
        
        #dimensions
        self.K     = self.F.shape[0]
        self.dim_x = self.H.shape[2]
        self.dim_y = self.H.shape[1]

        #x_t | x_t-1, y_{t-1:t}, r_t parameters
        self.C_x = torch.zeros(self.K, self.dim_x, self.dim_x)
        self.D_x = torch.zeros(self.K, self.dim_x, self.dim_y)
        self.D_prime_x = torch.zeros(self.K, self.dim_x, self.dim_y)
        self.cte_x = torch.randn(self.K, self.dim_x)
        self.sigma_x = torch.eye(self.dim_x).view(1,self.dim_x,-1).repeat(self.K,1,1)

        #y_t | r_t, y_t-1 parameters
        self.B_y = torch.zeros(self.K, self.dim_y, self.dim_y)
        self.cte_y = torch.randn(self.K, self.dim_y)
        #self.sigma_y = torch.zeros_like(self.B_y)
        self.sigma_y = torch.eye(self.dim_y).view(1,self.dim_y,-1).repeat(self.K,1,1)
        
        self.qr_0 = torch.randn(self.K,1)
        self.qr_r_1 = torch.randn(self.K,self.K)

        self.divergence_values = []
        self.mean_squared_error = []
        #become probs after applying softmax
        #dim0: r_t, dim1: r_t-1
        
    def _init_model(self, args):
        """args is a dictionary containing : 
        C_x, D_x, cte_x, sigma_x, B_y, cte_y, sigma_y, qr_0, qr_r_1""" 

        chol_Q = torch.cholesky(self.Q)
        self.tril_Q = self._triu_to_vect(chol_Q).requires_grad_(True)
        triu = self._vect_to_triu(self.tril_Q, self.dim_x)
        self.Q = (triu @ triu.transpose(-1,-2)).double()

        chol_R = torch.cholesky(self.R)
        self.tril_R = self._triu_to_vect(chol_R)[0].requires_grad_(True)
        triu = self._vect_to_triu(self.tril_R.repeat(self.K,1), self.dim_y)
        self.R = (triu @ triu.transpose(-1,-2)).double()

        self.x_init = args['x_init']
        self.C_x = args['C_x']
        self.C_x.requires_grad = True
        self.D_x = args['D_x']
        self.D_x.requires_grad = True
        self.D_prime_x = args['D_prime_x']
        self.D_prime_x.requires_grad = False
        self.cte_x = args['cte_x']
        self.cte_x.requires_grad = False

        self.chol_sigma_x = torch.cholesky(args['sigma_x'])
        self.tril_sigma_x = self._triu_to_vect(self.chol_sigma_x).requires_grad_()
        self.triu = self._vect_to_triu(self.tril_sigma_x, self.dim_x)
        self.sigma_x = (self.triu @ self.triu.transpose(-1,-2)).double()

        self.B_y = args['B_y']
        self.B_y.requires_grad = True
        self.cte_y = args['cte_y']
        self.cte_y.requires_grad = False

        self.chol_sigma_y = torch.cholesky(args['sigma_y'])
        self.tril_sigma_y = self._triu_to_vect(self.chol_sigma_y).requires_grad_()
        self.triu = self._vect_to_triu(self.tril_sigma_y, self.dim_y)
        self.sigma_y = (self.triu @ self.triu.transpose(-1,-2)).double()

        self.qr_0 = args['qr_0']
        self.qr_0.requires_grad = False
        self.qr_r_1 = args['qr_r_1']
        self.qr_r_1.requires_grad = True

        assert self.x_init.shape == (self.dim_x,)
        assert self.C_x.shape == (self.K, self.dim_x, self.dim_x)
        assert self.D_x.shape == (self.K, self.dim_x, self.dim_y)
        assert self.D_prime_x.shape == (self.K, self.dim_x, self.dim_y)
        assert self.cte_x.shape == (self.K, self.dim_x)
        assert self.sigma_x.shape == (self.K, self.dim_x, self.dim_x)
        
        assert self.B_y.shape == (self.K, self.dim_y, self.dim_y)
        assert self.cte_y.shape == (self.K, self.dim_y)
        assert self.sigma_y.shape == (self.K, self.dim_y, self.dim_y)

        assert self.qr_0.shape == (self.K, 1)
        assert self.qr_r_1.shape == (self.K, self.K)

        self.inv_Q = torch.inverse(self.Q)
        self.inv_R = torch.inverse(self.R)

        self.parameters = [self.C_x, self.D_x, self.D_prime_x, self.cte_x, self.cte_y, self.tril_sigma_x, self.tril_Q, self.tril_R,
                           self.tril_sigma_y, self.B_y, self.qr_0, self.qr_r_1,self.F,self.H,self.pr_r_1]

    
    def _vect_to_triu(self, vect, dim):
        
        triu = torch.zeros((self.K, dim, dim))

        tril_indices = torch.tril_indices(row=dim, col=dim, offset=0)
        triu[:,tril_indices[0], tril_indices[1]] = vect.float()

        return triu

    def _triu_to_vect(self, mat):
        
        dim = mat.shape[-1]
        triu_ones = torch.triu(torch.ones(dim,dim)).T

        return mat[:,triu_ones == 1.]

    def _conditionals_r2(self):

        """
        T : number of observations

        returns: log_qrGy  : log qr_t|y_0:t, K x T - 1
               : log_qrGry : log q(r_t-2|r_t-1, y_0:t-1), K x K x (T-1)
        """
        #pdb.set_trace()
        """Here we compute q(r_0 | y_0) = q(y_0 | r_0)q(r_0) / q(y_0)"""
        log_qrGry = torch.zeros(1,self.K,self.K)
        log_jump_cond = torch.softmax(self.qr_r_1,0).log()
        log_qrGy = self.pr_0.flatten().log().unsqueeze(1)

        for t in range(1,self.T):
            #pdb.set_trace()
            log_qyGry  = MVN(self.B_y @ self.observations[:,t-1] + self.cte_y,
                             self.sigma_y).log_prob(self.observations[:,t]).view(-1,1)

            log_joint  = log_qyGry + log_jump_cond + log_qrGy[:,-1].view(1,-1) 

            log_qrGy_t_unn = torch.logsumexp(log_joint, -1)
            norm_cst = torch.logsumexp(log_qrGy_t_unn,0)
            
            log_qrGy_t = log_qrGy_t_unn - norm_cst
            log_qrGy_t = log_qrGy_t.view(-1,1)

            log_qrGy   = torch.cat([log_qrGy, log_qrGy_t], -1)

            log_qrGry_t = log_joint - norm_cst - log_qrGy_t
            log_qrGry   = torch.cat([log_qrGry, log_qrGry_t.view(1,self.K,-1)], 0)

        return log_qrGy, log_qrGry[1:]


    def _conditionals_r(self):
        """
        T : number of observations

        returns: log_qrGy  : log qr_t|y_0:t, K x T - 1
               : log_qrGry : log q(r_t-2|r_t-1, y_0:t-1), K x K x (T-1)
        """
        #pdb.set_trace()
        """Here we compute q(r_0 | y_0) = q(y_0 | r_0)q(r_0) / q(y_0)"""
        #print(self.sigma_y)
        log_qrGy = MVN(self.cte_y, self.sigma_y).log_prob(self.observations[:,0]).view(-1,1) \
                       + torch.softmax(self.qr_0, 0).log()
        #print(torch.softmax(self.qr_0,0).log())
        log_qy_t = torch.logsumexp(log_qrGy,0).view(-1,1)
        #print(f'qy0: {log_qy_t.exp()}')
        log_qrGy = log_qrGy - log_qy_t

        #log_qrGry = torch.zeros(1,self.K,self.K)

        log_jump_cond = torch.softmax(self.qr_r_1,0).log()
        
        log_qy = log_qy_t
        log_qrGy = self.qr_0.flatten().log().unsqueeze(1)

        for t in range(1,self.T):
                  
            log_qyGry  = MVN(self.B_y @ self.observations[:,t-1] + self.cte_y,
                             self.sigma_y).log_prob(self.observations[:,t]).view(-1,1)

            log_qrGy_t = log_qyGry + log_jump_cond + \
                         log_qrGy[:,-1].view(1,-1)

            log_qrGy_t = torch.logsumexp(log_qrGy_t, -1).view(-1,1)
            log_qy_t   = torch.logsumexp(log_qrGy_t,0)

            log_qy = log_qy + log_qy_t

            log_qrGy_t = log_qrGy_t  - log_qy_t

            log_qrGy   = torch.cat([log_qrGy, log_qrGy_t], -1)

        log_qyGry  = MVN(self.B_y @ self.observations[:,t-1] + self.cte_y,
                    self.sigma_y).log_prob(self.observations[:,t]).view(-1,1)
        log_qrGy_T = log_qyGry + torch.softmax(self.qr_r_1,0).log() \
                    + log_qrGy[:,-1].view(1,-1)
        log_qrGy_T = torch.logsumexp(log_qrGy_T, -1).view(-1,1)
        log_qy   = torch.logsumexp(log_qrGy_T,0) + log_qy

        """regarding q(r_t-2|r_t-1, y_0:t-1), the first dimension is for r_t-1 and the second for 
        r_t-2"""
        
        return log_qrGy, log_qy

    
    def _joint_smoothing_jump(self, filtering_jump):
        """
        returns : r_t, r_t-1 | y_0:T, q(r1,r0 | y_0:T), ..., q(rT, rT-1 | y_0:T)
        """
        backward_yt = MVN(self.B_y @ self.observations[:,-2] + \
                          self.cte_y, self.sigma_y).log_prob(self.observations[:,-1]).reshape(-1,1)

        log_qjump_cond = torch.softmax(self.qr_r_1,0).log()
        
        #pdb.set_trace()
        log_joint_smoothing_t = backward_yt + log_qjump_cond + filtering_jump[:,-2].reshape(1,-1) 
        log_joint_smoothing_t = log_joint_smoothing_t - torch.logsumexp(log_joint_smoothing_t, (-1,-2))
        log_joint_smoothing = log_joint_smoothing_t.unsqueeze(0)

        log_joint_smoothing = torch.zeros(self.T, self.K, self.K)
        log_joint_smoothing[-1] = log_joint_smoothing_t

        for t in range(self.T-2,0,-1):

            backward_yt = backward_yt.reshape(-1,1) + \
                          MVN(self.B_y @ self.observations[:,t-1] + \
                              self.cte_y, self.sigma_y).log_prob(self.observations[:,t]).view(1,-1) + \
                          log_qjump_cond 

            backward_yt = torch.logsumexp(backward_yt, 0)

            log_joint_smoothing_t = backward_yt.unsqueeze(-1) + log_qjump_cond + filtering_jump[:,t - 1].unsqueeze(0)
            log_joint_smoothing_t = log_joint_smoothing_t - torch.logsumexp(log_joint_smoothing_t, (-1,-2))

            #log_joint_smoothing = torch.cat([log_joint_smoothing, log_joint_smoothing_t.unsqueeze(0)],0)
            log_joint_smoothing[t] = log_joint_smoothing_t
        #print('joint_smoothing',log_joint_smoothing[1:].exp())
        return log_joint_smoothing[1:] 

    def _filtering_moments(self, log_qrGy, log_qrGry):
        """
        returns : means m0,...,mT-1 : (T-1) x K x dim_x x 1
                : covs  P0,...,PT-1 : (T-1) x K x dim_x x dim_x 
        """
        #x_init = torch.tensor([200,0,0,0]).unsqueeze(0).double()
        #x_init = torch.tensor([0.]).unsqueeze(0).double()

        x_init = self.x_init 
        means = x_init.repeat(self.K,1).unsqueeze(-1)
        means = means[None,...]
        
        covs  = 50.*torch.eye(self.dim_x).unsqueeze(0).repeat(self.K,1,1).double().unsqueeze(0)

        for t in range(1,self.T):
            
            Dy = self.D_x @ self.observations[:,t].view(1,-1,1)+self.D_prime_x @ self.observations[:,t-1].view(1,-1,1)
            mean_t_1_ext = means[t-1].view(-1,1,self.dim_x,1)
            cov_t_1_ext  = covs[t-1].view(-1,1,self.dim_x,self.dim_x)
            #pdb.set_trace()
            mean_t = self.C_x @ mean_t_1_ext + Dy + self.cte_x.view(self.K,-1,1)
            mean_t = mean_t * (log_qrGry[t-1].T.view(self.K,-1,1,1).exp())
            mean_t = mean_t.sum(0).unsqueeze(0)

            means  = torch.cat([means, mean_t], 0)

            mean_t_ext = means[t][None,...]

            """cov_t  = self.sigma_x + self.C_x @ (cov_t_1_ext @ self.C_x.transpose(dim0 = 1, dim1 = 2) + \
                     mean_t_1_ext @ mean_t_1_ext.transpose(dim0 = 2, dim1 = 3) + \
                     mean_t_1_ext @ Dy.transpose(dim0 = 1, dim1 = 2) + \
                     mean_t_1_ext @ mean_t_ext.transpose(dim0 = 2, dim1 = 3)) + \
                     Dy @ (mean_t_1_ext.transpose(dim0 = 2, dim1 = 3) @ self.C_x.transpose(dim0 = 1, dim1 = 2) + \
                     Dy.transpose(dim0 = 1, dim1 = 2) + mean_t_ext.transpose(dim0 = 2, dim1 = 3)) - \
                     mean_t_ext @ mean_t_1_ext.transpose(dim0 = 2, dim1 = 3) @ self.C_x.transpose(dim0 = 1, dim1 = 2) - \
                     mean_t_ext @ Dy.transpose(dim0 = 1, dim1 = 2) + \
                     mean_t_ext @ mean_t_ext.transpose(dim0 = 2, dim1 = 3)"""

            mmT    = self.sigma_x + self.C_x @ cov_t_1_ext @ self.C_x.transpose(dim0 = -1, dim1 = -2) + \
                     self.C_x @ mean_t_1_ext @ mean_t_1_ext.transpose(dim0 = -2, dim1 = -1) @ self.C_x.transpose(dim0 = -2, dim1 = -1) + \
                     self.C_x @ mean_t_1_ext @ Dy.transpose(dim0 = -2, dim1 = -1) + \
                     (self.C_x @ mean_t_1_ext @ Dy.transpose(dim0 = -2, dim1 = -1)).transpose(dim0 = -1, dim1 = -2) + \
                     Dy @ Dy.transpose(dim0 = -1, dim1 = -2).unsqueeze(0)

            mmT    = mmT * (log_qrGry[t-1].exp().T.view(self.K, -1, 1, 1))
            mmT    = mmT.sum(0)

            cov_t  = mmT - mean_t.squeeze(0) @ mean_t.squeeze(0).transpose(-1,-2)
            covs   = torch.cat([covs, cov_t.unsqueeze(0)],0)
            
        return means, covs
    
    def _DKL_cont(self, filtering_mean, filtering_cov, joint_smoothing):
        
        dkl = 0
        q_r0_T = joint_smoothing[0].logsumexp(0).exp()
        #print(self.H)
        #print(self.Q)
        #print(self.R)
        K = self.Q @ self.H.transpose(-2,-1) @ torch.inverse(self.H @ self.Q @ self.H.transpose(dim0 = -1, dim1 = -2) + self.R)
        C_tilde = self.F - K @ self.H @ self.F
        D_tilde = K
        Sigma_tilde = (torch.eye(self.dim_x) - K @ self.H) @ self.Q
        H_tilde = self.H @ self.F
        Q_tilde = self.R + self.H @ self.Q @ self.H.transpose(dim0 = -1, dim1 = -2)
        trace   = ((torch.inverse(Sigma_tilde) @ self.sigma_x) * torch.eye(self.dim_x)).sum((-1,-2))
        log_det = torch.logdet(Sigma_tilde) - torch.logdet(self.sigma_x) - self.dim_x
        G = trace + log_det

        D = D_tilde - self.D_x
        D_prime= -self.D_prime_x
        A = C_tilde - self.C_x

        dkl0 = q_r0_T * MVN((self.H @ filtering_mean[0]).squeeze(-1),
                            self.R + self.H @ filtering_cov[0] @ self.H.transpose(dim0 = -1, dim1 = -2)).log_prob(self.observations[:,0])
        dkl0 = - dkl0.sum() #vérifié

        inv_sigma_tilde = torch.inverse(Sigma_tilde)
        inv_Q_tilde = torch.inverse(Q_tilde)
        #print(dkl0)

        for t in range(1,self.T):
            
            mean_ext = filtering_mean[t-1].unsqueeze(1)
            var_ext  = filtering_cov[t-1].unsqueeze(1)

            trace = (inv_sigma_tilde @ A @ var_ext @ A.transpose(dim0 = -1, dim1 = -2))
            trace = (trace * torch.eye(self.dim_x)).sum((-1,-2))

            alpha = (A @ mean_ext + D @ self.observations[:,t].reshape(-1,1)+D_prime@self.observations[:,t-1].reshape(-1,1)).transpose(-1,-2) @ \
                    inv_sigma_tilde @ (A @ mean_ext + D @ self.observations[:,t].reshape(-1,1)+D_prime@self.observations[:,t-1].reshape(-1,1))

            alpha = (G + trace + alpha.squeeze(-1).squeeze(-1))/2
            alpha = joint_smoothing[t-1].T.exp() * alpha
           
                     
            trace = (torch.inverse(Q_tilde) @ H_tilde @ var_ext @ H_tilde.transpose(-1,-2) * torch.eye(self.dim_y)).sum((-1,-2))
            beta  = (self.dim_y / 2) * torch.tensor(2*math.pi).log() + torch.logdet(Q_tilde)/2

            beta_norm = (H_tilde @ mean_ext - self.observations[:,t].reshape(-1,1)).transpose(-1,-2) @ inv_Q_tilde @ (H_tilde @ mean_ext - self.observations[:,t].reshape(-1,1))

            beta = beta + trace/2 + beta_norm.squeeze(-1).squeeze(-1)/2
            beta = joint_smoothing[t-1].T.exp() * beta

            dkl = dkl + beta.sum() + alpha.sum()

        return dkl + dkl0

    def _Dkl_discr(self, joint_smoothing, log_qrGry):

        log_qr0_T = joint_smoothing[0].logsumexp(0)
        #print(log_qr0_T.exp())
        dkl0 = torch.sum(log_qr0_T.exp().unsqueeze(-1) * (log_qr0_T.unsqueeze(-1) - self.pr_0.log()))
        dkl = dkl0 + (joint_smoothing.exp() * (log_qrGry - self.pr_r_1.softmax(0).log())).sum()
        
        return dkl 

    def KL(self):
        
        log_qrGy, log_qy = self._conditionals_r2()

        log_jump_joint_smooth = self._joint_smoothing_jump(log_qrGy)
        #smoothing : r_k, r_k-1 | y_0:T
        log_qrk_1Grky = log_jump_joint_smooth - log_jump_joint_smooth.logsumexp(-1).unsqueeze(-1)
        log_qrkGrk_1y = log_jump_joint_smooth - log_jump_joint_smooth.logsumexp(-2).unsqueeze(1)

        filtering_mean, filtering_cov = self._filtering_moments(log_qrGy, log_qrk_1Grky)
        mult = filtering_mean.squeeze(-1) * log_qrGy.exp().T.unsqueeze(-1)
        means = mult.sum(1)
        #print(means.size())
        MSE = (means.detach() - self.ground_truth).square().mean()

        dkl_cont  = self._DKL_cont(filtering_mean, filtering_cov, log_jump_joint_smooth)
        dkl_discr = self._Dkl_discr(log_jump_joint_smooth, log_qrkGrk_1y)
        #print('means')
        #plt.plot(means.detach(),color='blue')
        #plt.plot(self.ground_truth,color='red')
        #print(means.size())
#         plt.plot(means.detach()[:,0],means.detach()[:,2],color='blue')
#         plt.plot(self.ground_truth.transpose()[0,],self.ground_truth.transpose()[2,],color='red')
#         plt.figure()
#         plt.plot(means.detach()[:,1],color='blue')
#         plt.plot(self.ground_truth.transpose()[1,],color='red')
        
        #print('dkl_cont',dkl_cont)
        #print('dkl_discr',dkl_discr)
        return dkl_cont , dkl_discr, MSE
        
    def train(self, iterations, plot = False):

        
        #print(self.learning_rate)
        div_prec=10**10
        MSE_prec=10**10
        #scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.1)
        for i in tqdm(range(iterations)):
            #print(self.optimizer)
            self.optimizer.zero_grad()
            div_cont,div_discrete, MSE = self.KL()
            div=div_cont+div_discrete
            if div<div_prec:
                
                #model_state = {'model': self, 'optimizer': self.optimizer.state_dict()}
                #torch.save(model_state,"save_self")
                torch.save(self,"save_self")
                div_prec=div
                
            
            if (i>250):
                self.optimizer.param_groups[0]['lr']=1e-2
               
          
            
            if i%10==0:
                print(f'\ndiv_discrete: {div_discrete} div_continue: {div_cont} \ndivergence : {div}\nMSE: {MSE}')
               

            self.divergence_values.append(div.item())
            self.mean_squared_error.append(MSE.item())
            
            if MSE<MSE_prec:
                torch.save(self,"save_min_mse")
                MSE_prec=MSE
                
            
            if i<50:
                div_cont.backward(retain_graph = True)    
                #div.backward(retain_graph = True)
            else:
                div.backward(retain_graph = True)
                #self.optimizer.param_groups[0]['lr']=1e-1
            
            self.optimizer.step()
            #scheduler.step()
            triu_Q = self._vect_to_triu(self.tril_Q, self.dim_x)
            triu_R = self._vect_to_triu(self.tril_R.repeat(self.K,1), self.dim_y)
            triu_x = self._vect_to_triu(self.tril_sigma_x, self.dim_x) 
            triu_y = self._vect_to_triu(self.tril_sigma_y, self.dim_y) 
            self.Q  = (triu_Q @ triu_Q.transpose(-1,-2)).double()
            self.R  = (triu_R @ triu_R.transpose(-1,-2)).double()
            self.sigma_x = (triu_x @ triu_x.transpose(-1,-2)).double()
            self.sigma_y = (triu_y @ triu_y.transpose(-1,-2)).double()
            
        #model_state = {'model': self, 'optimizer': self.optimizer.state_dict()}
        #torch.save(model_state,"save_self_final")    
        torch.save(self,"save_self_final2")
        print(f'\ndivergence_finale : {div}\nMSE_finale: {MSE} \ndivmin :{div_prec}')
        
        if plot:

            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20,10))
            ax[0].plot(self.divergence_values)
            ax[0].set_xlabel('Iterations', fontsize = 20)
            ax[0].set_ylabel('$D_{\mathrm{KL}}$', fontsize = 20)

            ax[1].plot(self.mean_squared_error)
            ax[1].set_xlabel('Iterations', fontsize = 20)
            ax[1].set_ylabel('MSE', fontsize = 20)

            plt.show()


    
