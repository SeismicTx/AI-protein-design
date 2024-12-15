import dill
import numpy as np
import pandas as pd
from copy import deepcopy


class MultiScore():

    def __init__(self, target_seq, uniprot_index, gaps_as_zero=True):
        self.target_seq = target_seq
        self.uniprot_index = uniprot_index
        self.L = len(target_seq)
        self.idx_aa = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
        self.idx_i = np.arange(uniprot_index, uniprot_index+self.L)
        self.AA  = len(self.idx_aa)
        self.aa_map = {aa: np.eye(self.AA)[:,i] for i,aa in enumerate(self.idx_aa)}
        if gaps_as_zero:
            self.aa_map['-'] = np.zeros((self.AA,self.AA))[:,0]
        self.x = self.seq_to_mat()
        
        self.scorefxn = {}
        self.scorefxn_weights = {}
        self.scorefxn_maps = {}
        self.scorefxn_xform = {}
        self.null_mat = np.zeros((self.L, self.AA))
        self.null_mat[:,:] = np.nan
        
        self.seq_df = pd.DataFrame({'i':self.idx_i, 'aa':list(target_seq)})

    def save(self, outfile):
        '''save object as compressed dill file'''
        dill.settings['recurse'] = True
        with open(outfile, 'wb') as f:
            dill.dump(self, f)
        
    def seq_to_mat(self, seq=None):
        '''turn sequence into 2D one-hot matrix'''
        if seq is None:
            seq = self.target_seq
        x = np.array([self.aa_map[aa] for aa in seq])        
        return x
    
    def add_scorefxn(self, score_object, score_name, weight=1,
        xform=lambda model,seq,i0: model.predict_mutscan(seq,i0)):
        '''Add additional term to the scoring function.
        Provide a score_object that contains parameters & functions needed to compute
        single mutation scores [LxAA], and has 'model.idx_i' to ensure mapping to target seq.
        By default 'model.predict_mutscan' is invoked, other lambda functions can be specified by xform'''
        assert all(score_object.idx_aa == self.idx_aa)
        self.scorefxn[score_name] = score_object
        self.scorefxn_weights[score_name] = weight
        self.scorefxn_xform[score_name] = xform
        
    def _precompute_mapping(self, weight_map=None, validate=True):
        '''For each scorefxn object, precompute the registration mapping model.idx_i to self.idx_i
        This avoids recomputing the index mapping everytime scores are generated.'''
        if weight_map is None:
            weight_map = self.scorefxn_weights
        
        self.N = len(weight_map)
        self.names = np.array(list(weight_map.keys()))
        self.w = np.array(list(weight_map.values()))
        self.idx_i_list = np.array([self.scorefxn[k].idx_i for k in self.names], dtype=object)
        self.send_list,self.land_list = get_index_maps(self.idx_i_list,self.idx_i)
        
        if validate:
            print(f'out of {len(self.seq_df)} target sequence positions')
            for k in self.names:
                self.seq_df[f'aa_wt_{k}'] = self.seq_df.i.map(self.scorefxn[k].seq_df.set_index('i').aa)
                vs = self.seq_df[~self.seq_df[f'aa_wt_{k}'].isna()]
                print(sum(vs[f'aa_wt_{k}']==vs['aa']),'/',len(vs),f' sequence match for model {k}')
        
    def predict_mutscan(self, seq, start_index, recompute_map=False):
        '''Iterate over scorefxn operations yielding mutation scores [model.LxAA],
        reindexes to our [LxAA], and computes the weighted sum'''
        if recompute_map:
            self._precompute_mapping()

        self.smm = deepcopy(self.null_mat)
        self.unaligned_scores = np.zeros(self.N, dtype=object)
        for n,k in enumerate(self.names):
            self.unaligned_scores[n] = self.scorefxn_xform[k](self.scorefxn[k], seq, start_index)

        self.aligned_scores = send_matrices_to_target(self.unaligned_scores, self.send_list, self.land_list, self.smm)
        self.smm = np.sum(self.w[:,np.newaxis,np.newaxis] * self.aligned_scores, axis=0)
        return self.smm
    
    def mat_to_table(self, aligned_scores=None, smm=None, seq=None):
        '''unpack predicted scores to human-readable dataframe'''
        if aligned_scores is None:
            aligned_scores = self.aligned_scores
        if smm is None:
            smm = self.smm
        if seq is None:
            seq = self.target_seq
        
        self.aligned_scores_df_i = {}
        self.aligned_scores_df = []
        for n,k in enumerate(self.names):
            df = pd.DataFrame(
                    aligned_scores[n],
                    index=pd.Series(self.idx_i, name='i'),
                    columns=pd.Series(self.idx_aa, name='aa_mut'))
            
            self.aligned_scores_df_i[k] = df
            self.aligned_scores_df.append(
                df.melt(ignore_index=False, value_name=k
                       ).set_index('aa_mut',append=True))
                
        self.smm_df = pd.DataFrame(
            smm,
            index=pd.Series(self.idx_i, name='i'),
            columns=pd.Series(self.idx_aa, name='aa_mut'))
        
        self.aligned_scores_df.append(
            self.smm_df.melt(ignore_index=False, value_name='multiscore'
                       ).set_index('aa_mut',append=True))
                
        self.aligned_scores_df = pd.concat(self.aligned_scores_df, axis=1).reset_index()
        self.aligned_scores_df['aa_wt'] = self.aligned_scores_df.i.map(dict(zip(self.idx_i, seq)))
        self.aligned_scores_df['mut'] = self.aligned_scores_df.apply(lambda x: f'{x.aa_wt}{x.i}{x.aa_mut}', axis=1)
        
        for n,k in enumerate(self.names):
            self.aligned_scores_df[f'{k}xW'] = self.w[n]*self.aligned_scores_df[k]
        
        col_order = ['i','aa_wt','aa_mut','mut','multiscore'] +  list(self.names) + [f'{k}xW' for k in self.names]
        self.aligned_scores_df = self.aligned_scores_df.loc[:,col_order]
        
        return self.aligned_scores_df
    
    
'''
scripts that stack mutation matrices reshapes to match target
coded so the mapping is computed once, and stacking will be fast when matrices are updated
'''

def send_matrix_to_target(matrix, idx_send, idx_land, target, axis=0):
    '''given precomputed send - land indices,
    sends values from matrix to land locations in target matrix'''
    target.swapaxes(axis,0)[idx_land] = matrix.swapaxes(axis,0)[idx_send]
    return target

def get_index_map(idx_matrix, idx_target):
    '''precomputes send - land indices for matrix indexings'''
    map_target = {i:n for n,i in enumerate(idx_target)}
    idx_send = [n for n,i in enumerate(idx_matrix) if i in idx_target]
    idx_land = [map_target[i] for i in idx_matrix if i in idx_target]
    return idx_send, idx_land

def get_index_maps(idx_matrices, idx_target):
    '''precomputes send - land indices for list of matrix indexings'''
    idx_sends, idx_lands = zip(*[get_index_map(idx_matrix,idx_target) for idx_matrix in idx_matrices])
    return idx_sends, idx_lands 

def send_matrices_to_target(matrices, idx_sends, idx_lands, target, axis=0):
    '''given precomputed send - land indices,
    sends values from matrices to land locations in copies of target matrix'''
    targets = np.stack([target]*len(matrices))
    for i, (matrix, idx_send, idx_land) in enumerate(zip(matrices, idx_sends, idx_lands)):
        targets[i] = send_matrix_to_target(matrix, idx_send, idx_land, targets[i], axis)
    return targets
