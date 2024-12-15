import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.special import softmax
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_mutations(seq1, seq2, seq_offset):
    muts = []
    for i, (a1, a2) in enumerate(zip(seq1.upper(),seq2.upper())):
        if a1!=a2:
            muts.append(f'{a1}{i+seq_offset}{a2}')
    return muts

def normalize(vals, max_val=None):
    if max_val is None:
        max_val = vals.max()
    return (vals - vals.min()) / (max_val - vals.min())

class MultiAnnealer():
    '''
    '''

    def __init__(self,MultiScore,seq0=None,mutant_indices=None):
        self.Score = MultiScore
        self.target_seq = np.array(list(self.Score.target_seq))
        self.Score.predict_mutscan(self.target_seq, self.Score.uniprot_index)
        
        # if not specified, set all non-NaN indices as mutable
        if mutant_indices is None:
            self.model_coverage = ~np.any(np.isnan(self.Score.smm), axis=1)
            self.mutant_indices = self.Score.idx_i[self.model_coverage]
        else:
            self.mutant_indices = mutant_indices
        self.mutant_mask = np.array([(i in self.mutant_indices) for i in self.Score.idx_i])
        
        # if not specified, generate random seq equal L to MultiScore, non-mutant coords as WT
        if seq0 is None:
            self.seq0 = np.random.choice(self.Score.idx_aa, self.Score.L)
            self.seq0[~self.mutant_mask] = self.target_seq[~self.mutant_mask]
        else:
            self.seq0 = np.array(seq0)
        assert len(self.seq0) == self.Score.L
        
        # initialize
        self.seq = deepcopy(self.seq0)
        self.trajectories = []
        self.Probs = np.zeros(self.Score.smm.shape)
        

    def GibbsSample(self, T):
        '''draw a random mutation, based on single mut energies'''
        # compute energies
        self.Score.predict_mutscan(self.seq, self.Score.uniprot_index)
        self.Probs[self.mutant_mask,:] = softmax(self.Score.smm[self.mutant_mask,:]/T)
        #softmax(np.nan_to_num((self.Score.smm / T) + 0.0001, 0))
        p_flat = self.Probs.flatten()
        
        # draw mutant
        ij = np.random.choice(np.arange(p_flat.shape[0]), p=p_flat)
        
        #modify sequence
        i,j = np.unravel_index(ij, self.Probs.shape) 
        idx = self.Score.idx_i[i]
        aa = self.Score.idx_aa[j]
        mut = f'{self.seq[i]}{idx}{aa}'
        self.seq[i] = aa
        self.trajectories.append(
            {'mut':mut,'seq':''.join(self.seq),
            'P_mut':p_flat[ij], 'P_avg':np.mean(p_flat)} ) #,ignore_index=True)
        
    def Run(self, Tcycle, loud=False):
        n=0
        for T in tqdm(Tcycle):
            self.GibbsSample(T)
            if loud and n%100 == 0:
                print(''.join(self.seq))
            n+=1
            
            
    '''Tools for inspecting the model and outputs'''
    def get_design_summary(self, seq, seq_offset, target_seq=None):
        if target_seq is None:
            target_seq = self.Score.target_seq
        muts = get_mutations(target_seq, seq, seq_offset)
        self.Score.predict_mutscan(target_seq, seq_offset)
        smm = self.Score.mat_to_table()
        assert len(set(muts) - set(smm.mut)) == 0
        return smm[smm.mut.isin(muts)]

    def save_trajectory(self, plot=True):
        self.traj_df = pd.DataFrame(self.trajectories)
        self.target_scores = {}
        for col in self.Score.names:
            print(col)
            try:
                self.traj_df[col] = self.traj_df.seq.apply(
                    lambda x: self.Score.scorefxn[col].predict(
                        x, self.Score.uniprot_index))
                self.target_scores[col] = self.Score.scorefxn[col].predict(
                        self.Score.target_seq, self.Score.uniprot_index)
            except: # skip score_functions that lack single sequence predictor
                continue
                
                
        if plot:
            leg = []
            f,ax = plt.subplots(1,2, figsize=(10,5))
            for col in self.Score.names:
                if col in self.traj_df.columns:
                    ax[0].plot(normalize(self.traj_df[col]))
                    ax[1].plot(normalize(self.traj_df[col], self.target_scores[col] / (1+self.Score.L*(col=='dist'))))
                    leg.append(col)
            ax[1].axhline(1.0, c='k', ls='--', zorder=-1, lw=0.75)
            ax[0].set_title('Score / max value')
            ax[1].set_title('Score / target seq value')
            ax[1].legend(leg)
                    
        return self.traj_df

    def visualize_mats(self):
        self.Score.predict_mutscan(self.Score.target_seq, self.Score.uniprot_index)
        smm = self.Score.w[:,np.newaxis,np.newaxis] * self.Score.aligned_scores

        N = len(self.Score.names)
        f,ax=plt.subplots(N,1, figsize=(10,8*N/6))
        for i,n in enumerate(self.Score.names):
            cm = ax[i].imshow(smm[i].T, cmap='RdBu', interpolation='none');
            f.colorbar(cm, aspect=4, shrink=0.5,ax=ax[i]);
            ax[i].set_title(n)
        f.tight_layout()

    def visualize_design(self, seq, seq_offset, tepi_name=None, evh_name='evh', dist_name='dist'):
        self.design_seq = seq
        self.summary = self.get_design_summary(seq, seq_offset)
        
        cols = [k for k in self.Score.names
                if (k in self.summary.columns) and not (k in [evh_name, dist_name])]
        f,ax = plt.subplots(1,len(cols), figsize=(4*len(cols),4));
        
        for i,k in enumerate(cols):
            ax[i].scatter(self.summary[evh_name], self.summary[k])
            ax[i].set_title(k)
        
        if tepi_name is None:
            return
        
        f,ax = plt.subplots(figsize=(10,3))
        tepi = self.Score.scorefxn[tepi_name]
        
        tepi.predict(self.Score.target_seq, self.Score.uniprot_index)
        self.target_p_epitope = deepcopy(tepi.P_seq_epitopes[0])
        tepi.predict(seq, seq_offset)
        self.design_p_epitope = deepcopy(tepi.P_seq_epitopes[0])
        
        ax.plot(tepi.idx_cores, self.design_p_epitope)
        ax.plot(tepi.idx_cores, self.target_p_epitope, alpha=0.5)
