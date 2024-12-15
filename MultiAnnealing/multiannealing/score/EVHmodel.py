import dill
import pandas as pd
import numpy as np
from evcouplings.couplings import CouplingsModel
from evcouplings.couplings.model import _single_mutant_hamiltonians


class EVHmodel:
    '''Given a sequence, index offset, and evcouplings model file
    Create a scoring object that produces evolutionary hamiltonian scores for seq variants

    Note: Couplings models are typically indexed relative to full length uniprot seq.
    We inherit the indexing from the couplings model. Therefore seq_offset must translate
    the target sequence to fit into the full length uniprot indexing.
    
    Adhere to the rules of a score class:
    - self.predict(seq, seq_offset) [1]
    - self.predict_mutscan(seq, seq_offset) [LxAA]
    - self.mat_to_table(mat) [pd.DataFrame]
    self.idx_i [L], self.idx_aa [LxAA], self.seq_df (cols: i,aa)
    More: self.model (evcouplings.model object), self.conform_seq_to_model (strip to focus columns)
    '''

    def __init__(self, seq, seq_offset, couplings_model_file, quiet=False, set_modelfile_index=None):
        '''get necessary information from couplings.model file,
        if not 'quiet' report if offset sequence properly corresponds to couplings model'''
        if isinstance(couplings_model_file,str):
            self.couplings_model_file = couplings_model_file
            self.model = CouplingsModel(couplings_model_file)
        else:
            self.model = couplings_model_file

        if set_modelfile_index is not None:
            self.model.index_list = self.model.index_list - self.model.index_list[0] + set_modelfile_index

        self.idx_i = self.model.index_list
        self.idx_aa = self.model.alphabet
        self.model_seq_map = dict(zip(self.model.index_list, self.model.target_seq))
        self.target_seq = seq
        self.seq = seq
        self.seq_offset=seq_offset
        self.seq_df = pd.DataFrame({
            'i':seq_offset+np.arange(len(seq)), 'aa':list(seq)})
        self.seq_df['modeled'] = self.seq_df.i.isin(self.model.index_list)
        self.seq_df['aa_model_target'] = self.seq_df.i.map(self.model_seq_map)
        if not quiet:
            matches = self.seq_df.aa == self.seq_df.aa_model_target
            print(f'{matches.mean()*100:.0f}% match to EVHModel', f'{(~matches).sum()} mismatches')
            print(f'{sum(~self.seq_df.modeled)} residues missing in EVHModel')

    def save(self, outfile):
        '''save object as compressed dill file'''
        with open(outfile, 'wb') as f:
            dill.dump(self, f)
        
    def predict(self, seq=None, seq_offset=None):
        '''compute hamiltonian of a single sequence'''
        if seq is None:
            seq = self.seq
        seq = self.conform_seq_to_model(seq, seq_offset)
        pred = self.model.hamiltonians([seq])[0,0]
        return pred
    
    def predict_delta(self, seq, seq_ref=None, seq_offset=None):
        '''compute delta hamiltonian of single sequence versus reference'''
        if seq_ref is None:
            seq_ref = self.seq
        seq_ref = self.conform_seq_to_model(seq_ref, seq_offset)
        seq = self.conform_seq_to_model(seq)
        pred, pred_ref = self.model.hamiltonians([seq, seq_ref])[:,0]
        pred_delta = pred - pred_ref
        return pred_delta
    
    def predict_mutscan(self, seq_ref=None, seq_offset=None):
        '''compute delta hamiltonian of all single mutations [LxAA] relative to target'''
        if seq_ref is None:
            seq_ref = self.seq
        seq_ref = self.conform_seq_to_model(seq_ref, seq_offset)
        seq_ref_onehot = self.model.convert_sequences([seq_ref])[0]
        pred_mutscan = _single_mutant_hamiltonians(seq_ref_onehot, self.model.J_ij, self.model.h_i)        
        return pred_mutscan[:,:,0]

    def mat_to_table(self, X, flat=True):
        '''unpack predicted delta hamiltonian per mutation [LxAA] as human-readable table'''
        AAs = self.model.alphabet
        if X.shape == (len(self.seq), len(AAs)):
            X_df = pd.DataFrame(X, index=self.seq_df.i, columns=list(AAs))
        else:
            X_df = pd.DataFrame(np.zeros((len(self.seq), len(AAs))), index=self.seq_df.i, columns=AAs)
            X_df.loc[:,:] = np.nan
            included_indices = [(i in X_df.index) for i in self.model.index_list]
            print(len(included_indices), sum(included_indices), len(self.model.index_list[included_indices]))
            X_df.loc[self.model.index_list[included_indices],:] = X[included_indices,:]
        if flat:
            X_df = X_df.melt(ignore_index=False).reset_index()
            X_df.columns = ['i','aa_mut','delta_hamiltonian']
            X_df['aa_wt'] = X_df.i.map(self.seq_df.set_index('i').aa)
            X_df['mut'] = X_df['aa_wt'] + X_df['i'].astype(str) + X_df['aa_mut']
            X_df = X_df.loc[:,['i','aa_wt','aa_mut','mut','delta_hamiltonian']]
        return X_df
    
    def conform_seq_to_model(self, seq, seq_offset=None, validate=False):
        '''extract focus column positions from provided sequence,
        via full length uniprot indices in couplings model'''
        L = len(self.model.target_seq)
        if len(seq) == L:
            return seq
        if seq_offset is None:
            seq_offset = self.seq_offset
        if validate:
            assert(self.model.index_list[0]-seq_offset >= 0)
            assert(self.model.index_list[-1]-seq_offset < len(seq))
        
        seqx = ''
        for i in self.model.index_list:
            seqx += seq[i-seq_offset]
        return seqx
