import dill
import numpy as np
import pandas as pd


class TargetSeqEffect():
    '''Modifies scoring object to give mutations relative to target
    sequence, independent of input sequence.
    
    Useful when confidence in model is higher near specified sequence,
    than for design tragectories e.g. as a filter to discourage damaging mutations.'''
    
    def __init__(self, model, target_seq=None):
        self.target_seq = target_seq
        if target_seq is None:
            self.target_seq = model.target_seq
        self.smm = model.predict_mutscan(self.target_seq, model.seq_offset)
        self.seq_offset = model.seq_offset
        self.idx_i = model.idx_i
        self.idx_aa = model.idx_aa
        self.L = len(self.idx_i)
        self.AA = len(self.idx_aa)
        self.aa_map = {aa: np.eye(self.AA)[:,i] for i,aa in enumerate(self.idx_aa)}
        self.seq_df = model.seq_df

    def save(self, outfile):
        '''save object as compressed dill file'''
        with open(outfile, 'wb') as f:
            dill.dump(self, f)
                 
    def penalize_aa(self, aa, penalty):
        '''e.g. to avoid adding cysteines self.penalize_aa('C',-10)'''
        idx_myaa = np.where(self.idx_aa == aa)[0][0]
        self.smm[:,idx_myaa] = penalty
        self.smm[self.target_seq==aa,idx_myaa] = 0
        
    def penalize_site(self, i, penalty):
        '''avoid mutating site'''
        idx_myi = np.where(self.idx_i == i)[0][0]
        self.smm[idx_myi,:] = penalty
        self.smm[idx_myi,self.idx_aa==self.target_seq[idx_myi]] = 0
        
    def predict_mutscan(self,seq,seq_offset):
        return self.smm

    def seq_to_mat(self, seq, seq_offset=None):
        if isinstance(seq, np.ndarray) and len(seq.shape) == 2:
            return seq
        
        x = self.seq_to_mat_basic(seq)
        
        if seq_offset is not None and (seq_offset!=self.seq_offset or len(x) != self.L):
            assert seq_offset <= self.idx_i[0]
            x = x[self.idx_i-seq_offset]

        assert x.shape == (self.L, self.AA)

        return x

    def seq_to_mat_basic(self, seq=None):
        x = np.array([self.aa_map[aa] for aa in seq])        
        return x


class SeqDist():
    '''Hamming distance of mutations versus a target sequence'''
    
    def __init__(self, target_seq, uniprot_start):
        self.target_seq = target_seq
        self.uniprot_start = uniprot_start
        self.L  = len(self.target_seq)
        self.idx_aa = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
        self.AA = len(self.idx_aa)
        self.aa_map = {aa: np.eye(self.AA)[:,i] for i,aa in enumerate(self.idx_aa)}

        self.idx_i = np.arange(uniprot_start, uniprot_start+self.L)
        
        self.seq_df = pd.DataFrame({'i':self.idx_i,'aa':list(target_seq)})
    
    def save(self, outfile):
        '''save object as compressed dill file'''
        with open(outfile, 'wb') as f:
            dill.dump(self, f)
        
    def get_seq_coords(self, seq, seq_offset):
        seq = ''.join(seq).upper()
        if (seq_offset == self.uniprot_start) and (len(seq) == self.L):
            self.seq = seq
        else:
            self.seq = [seq[i-seq_offset] for i in self.idx_i]
        
    def predict(self, seq, seq_offset):
        self.get_seq_coords(seq, seq_offset)
        self.hamming = sum(a!=b for a,b in zip(self.target_seq, self.seq))
        return self.hamming
    
    def predict_mutscan(self, seq, seq_offset):
        self.predict(seq, seq_offset)
        self.smm_delta = np.array([1*(a==b)-self.aa_map[a] for a,b in zip(self.target_seq, self.seq)])
        self.smm_hamming = self.hamming + self.smm_delta
        return self.smm_hamming