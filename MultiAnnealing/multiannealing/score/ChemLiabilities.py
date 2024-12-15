import dill
import pandas as pd
import numpy as np
from copy import deepcopy
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from ..utils import py_pross

class ChemLiabilities:
    '''Given a sequence, index offset, pdbfile and chain
    Create a scoring object that tallies chemical liabilities in seq variants
    The liability predictions are based on a motif library, thresholds for SASA, and 2ndary structure
    

    Note: PDB files often have bespoke indexing. We recommend adjusting your PDB file itself to share
    full length uniprot indexing.
    Otherwise, specify 'pdb_index' during initialization to set the 1st index of the target chain.
    With PDB translated to full length uniprod indexing, ensure that seq_offset translates
    the target sequence to fit into the full length uniprot indexing.

    Adhere to the rules of a score class:
    - self.predict(seq, seq_offset) [1]
    - self.predict_mutscan(seq, seq_offset) [LxAA]
    - self.mat_to_table(mat) [pd.DataFrame]
    self.idx_i [L], self.idx_aa [LxAA], self.seq_df (cols: i,aa)
    More: self.idx_liabilities, self.sasa_vec [L]
    '''
    
    def __init__(self, seq, seq_offset, pdb_file, pdb_chain, pdb_index=None,pdb_name=None, quiet=False,
        AA_CHEMISTRY_DATA_FILE = 'data/AA_sasa.tsv'):
        '''prepares sequence & structure for liability predictions
        recommend using uniprot index for start of seq idx, and start of pdb idx'''
        
        # sequence info
        self.seq = seq
        self.seq_offset = seq_offset
        self.idx_i = np.arange(seq_offset, seq_offset+len(seq))
        self.idx_aa = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
        self.L = len(self.idx_i)
        self.AA = len(self.idx_aa)
        self.aa_map = {aa: np.eye(self.AA)[:,i] for i,aa in enumerate(self.idx_aa)}
        self.seq_df = pd.DataFrame({'aa': list(self.seq), 'i': self.idx_i})
        self.seq_to_mat(seq, update=True, compare=True)
        
        # amino acid and chemical liability info
        self.load_chemistry(AA_CHEMISTRY_DATA_FILE)
        self.idx_liabilities = np.array(list(self.liabilities_dict.keys()))
        self.idx_liabilities_map = {k:i for i,k in enumerate(self.idx_liabilities)}
        
        # structure info
        self.pdb_file = pdb_file
        self.pdb_chain = pdb_chain
        if pdb_name is None:
            pdb_name = pdb_file.split('/')[-1].strip('.pdb')
        self.pdb_name = pdb_name
        self.load_structure()
        if pdb_index is None:
            pdb_index = self.sasa_df.resi.min()
        self.pdb_index = pdb_index
        self.sasa_df['i'] = self.sasa_df.resi - self.sasa_df.resi.min() + pdb_index
        self.pdb_pross_df['i'] = self.pdb_pross_df.resi - self.pdb_pross_df.resi.min() + pdb_index
        
        # merge with target sequence (dataframe with row per AA in seq)
        self.seq_df['pdb_aa'] = self.seq_df.i.map(self.sasa_df.set_index('i').aa)
        self.seq_df['sasa'] = self.seq_df.i.map(self.sasa_df.set_index('i').sasa)
        self.seq_df['sasa_rel'] = self.seq_df.i.map(self.sasa_df.set_index('i').sasa_rel)
        self.seq_df['pdb_aa'] = self.seq_df.i.map(self.sasa_df.set_index('i').aa)
        self.seq_df['ss'] = self.seq_df.i.map(self.pdb_pross_df.set_index('i').ss)
        self.sasa_vec = np.array(self.seq_df.sasa)
        self.ss_vec = np.array(self.seq_df.ss)
        
        # validate
        matches = self.seq_df.aa == self.seq_df.pdb_aa
        if not quiet:
            print(f'{matches.mean()*100:.0f}% match to PDB', f'{(~matches).sum()} mismatches')
            print(f'{sum(self.seq_df.pdb_aa.isna())} residues missing in PDB')
            
        # predict chem liabilities
        self.predict()
        self.predict_mutscan()
        self.mat_to_table()
    
    def save(self, outfile):
        '''save object as compressed dill file'''
        with open(outfile, 'wb') as f:
            dill.dump(self, f)
    
    def predict(self, seq=None, seq_offset=None, sasa_criterion=True, ss_criterion=False, include_sites=None):
        '''tally liabilities for a single input sequence'''
        self.x = self.seq_to_mat(seq, seq_offset)
        self.x_pred_i = np.zeros((self.idx_liabilities.shape[0], self.idx_i.shape[0]))
        for n, (motif,sasa,in_loop) in enumerate(self.liability_matrices):
            N = motif.shape[0]
            
            for i in range(len(self.idx_i)-N-1):
                if include_sites is not None and (self.idx_i[i] not in include_sites[self.idx_liabilities[n]]):
                    continue
                if sasa_criterion and (self.sasa_vec[i] <= sasa):
                    continue
                if ss_criterion and self.ss_vec[i]:
                    continue

                S = np.sum(self.x[i:i+N] * motif)
                self.x_pred_i[n,i] = (S == N)

        self.x_pred_liab = np.sum(self.x_pred_i, axis=1)
        self.x_pred = np.sum(self.x_pred_liab)
        return self.x_pred
    
    def predict_mutscan(self, seq=None, seq_offset=None, sasa_criterion=True, ss_criterion=False, include_sites=None):
        '''tally change in liabilities for all single mutant of target sequence [LxAA]'''
        self.x = self.seq_to_mat(seq, seq_offset)
        self.x_pred_smm_liab = np.zeros((self.idx_liabilities.shape[0], self.idx_i.shape[0], self.idx_aa.shape[0]))
        for n, (motif,sasa,in_loop) in enumerate(self.liability_matrices):
            N = motif.shape[0]
            for i in range(self.x.shape[0]-N-1):
                if include_sites is not None and (self.idx_i[i] not in include_sites[self.idx_liabilities[n]]):
                    continue
                if sasa_criterion and (self.sasa_vec[i] <= sasa):
                    continue
                if ss_criterion and self.ss_vec[i]:
                    continue
                Sj = np.sum(self.x[i:i+N] * motif, axis=1)
                S = np.sum(Sj)
                if S == N: # hit, any site mutants can remove if mismatch (0 or -1)
                    self.x_pred_smm_liab[n,i:i+N,:] += motif - 1
                elif S == N-1: # 1aa away from hit, mismatch site mutants can add (0 or 1)
                    j = np.where(Sj==0)[0][0]
                    self.x_pred_smm_liab[n,i+j,:] += motif[j]
                    
        self.x_pred_smm = np.sum(self.x_pred_smm_liab, axis=0)
        return self.x_pred_smm
        
        
    def seq_to_mat(self, seq, seq_offset=None, update=False, compare=False):
        '''transform given sequence into [LxAA] one hot encoding,
        compare versus preloaded PDB, and save human-readable sequence table'''
        if seq is None:
            return self.x
        elif isinstance(seq, np.ndarray) and len(seq.shape) == 2:
            return seq
        
        x = self.seq_to_mat_basic(seq)
        
        if seq_offset is not None:
            assert seq_offset <= self.idx_i[0]
            x = x[self.idx_i-seq_offset]
            # idx_i is in uniprot coords, so this gives 0-index
            # given sequence must be superset of target region
        
        if update:
            self.x = x
            self.x_df = pd.DataFrame(x, index=self.idx_i, columns=self.idx_aa)
            
        if compare:
            seq_x = np.array(self.x_df.columns)[np.argmax(x, axis=1)]
            self.seq_df.loc[:,'aa_seq'] = seq_x
            print('given sequences matches model WT at:',sum(self.seq_df.aa_seq == self.seq_df.aa),'/', len(self.seq_df), 'sites')
            print('there are ',sum(self.seq_df.aa_seq.isna()),' sites covered by the model but missing in given seq')

        return x
    
    
    def seq_to_mat_basic(self, seq=None):
        '''transform given sequence into [LxAA] one hot encoding'''
        x = np.array([self.aa_map[aa] for aa in seq])        
        return x
        
 
    def load_chemistry(self, AA_CHEMISTRY_DATA_FILE):
        '''load max surface area of amino acids, and liability motifs'''
        self.AA_CHEMISTRY_DATA_FILE = AA_CHEMISTRY_DATA_FILE
        self.sasa = pd.read_csv(AA_CHEMISTRY_DATA_FILE,sep='\t')
        self.map_aa3 = self.sasa.set_index('A3').AA
        self.map_max_sasa = self.sasa.set_index('A3').Tien_measured
        # motif, minimum sasa, only if in loop (T/F)
        self.liabilities_dict = {
            'deamidation' : ['N[DNPTGSC]', 30, True],
            'DP_clipping' : ['DP', 0, False],
            'N-glycosylation' : ['N[^P][ST]', 40, False],
            'RGD' :         ['RGD', 0, False],
            'oxidation' :   ['[MW]', 40, False],
            'isomerization' : ['D[DCSAG]', 15, True]
        }
        self.liability_matrices = [(self.motif_to_matrix(motif), sasa, in_loop)
                                   for motif,sasa,in_loop in self.liabilities_dict.values()]
            
            
        
    def load_structure(self):
        '''load PDB, compute SASA per residue, and extract 2ndary structure'''
        p = PDBParser(QUIET=1)
        sr = ShrakeRupley()
        self.pdb = p.get_structure(self.pdb_name, self.pdb_file)
        
        # drop chains other than target
        chains = list(self.pdb[0].get_chains())
        for c in chains:
            if c.id != self.pdb_chain:
                self.pdb[0].detach_child(c.id)
        sr.compute(self.pdb, level='R')
        
        # compute SASA information
        self.sasa_df = pd.DataFrame(
            [(r.resname, r.id[1], r.sasa) for r in
             self.pdb[0][self.pdb_chain].get_residues()],
            columns=['aa3','resi','sasa'])
        self.sasa_df['sasa_rel'] = self.sasa_df.apply(
            lambda r: r.sasa/self.map_max_sasa[r.aa3] if (r.aa3 in self.map_max_sasa) else None, axis=1)
        self.sasa_df['aa'] = self.sasa_df.aa3.map(self.map_aa3)
        self.pdb_pross = py_pross.PDBFile(self.pdb_file)
        self.pdb_pross_chain = self.pdb_pross.read(as_protein=1).chains_with_name(self.pdb_chain)[0]
        _,_,_,self.pdb_pross_txt = py_pross.rc_ss(self.pdb_pross_chain)
        self.pdb_pross_df = pd.DataFrame({'resi':[int(r.idx) for r in self.pdb_pross_chain.elements],
                                          'ss': self.pdb_pross_txt})
        self.sasa_df['ss'] = self.sasa_df.resi.map(self.pdb_pross_df.set_index('resi').ss.apply(lambda x: x in 'HE'))
        
    def expand_motif(self, motif):
        '''expand regex motif into a list of acceptable amino acids per position, e.g.:
        'A[GT]C' -> ['A','GT','C']
        'A[^G]C' -> ['A','ATC','C']
        '''
        aa_list = []
        bracket_open = False
        negate = False

        for aa in motif:
            if not bracket_open:
                if aa == '[':
                    bracket_open = True
                    aa_list.append('')
                    continue
                else:
                    aa_list.append(aa)
            else:
                if aa == ']':
                    bracket_open = False
                    negate = False
                elif aa == '^':
                    aa_list[-1] = ''.join(self.idx_aa)
                    negate=True
                else:
                    if negate:
                        aa_list[-1] = ''.join([c for c in aa_list[-1] if c != aa])
                    else:
                        aa_list[-1] += aa
        return aa_list

    def motif_to_matrix(self, motif):
        '''convert regex motif into one-hot matrix representation'''
        motif_x = self.expand_motif(motif)
        mat = np.zeros((len(motif_x),len(self.idx_aa)))

        for i,aa_set in enumerate(motif_x):
            col = np.array([(aa in aa_set) for aa in self.idx_aa])
            mat[i,:] = col

        return mat
    
    def pred_to_table(self, x_pred_i=None):
        '''unpack predicted liabilities as human-readable table'''
        if x_pred_i is None:
            x_pred_i = self.x_pred_i
        motif_length = {n: m[0].shape[0] for n,m in enumerate(self.liability_matrices)}
        
        self.pred_dict = {
            (self.idx_i[i], self.idx_liabilities[n]): f'{self.idx_i[i]}{self.seq[i:i+motif_length[n]]}'
            for n,i in zip(*np.where(x_pred_i))
        }

        self.pred_df = deepcopy(self.seq_df)
        
        # sites of chemical modifications
        for liab in self.idx_liabilities:
            self.pred_df.loc[:,liab] = self.pred_df.i.apply(
                lambda i: self.pred_dict[(i,liab)] if (i,liab) in self.pred_dict else None)
        
        self.pred_df.loc[:,'liabilities'] = self.pred_df.apply(
            lambda x: ','.join([l for l in self.idx_liabilities if not x[l] is None]), axis=1)
        
        self.pred_df.loc[:,'motifs'] = self.pred_df.apply(
            lambda x: ','.join([x[l] for l in self.idx_liabilities if not x[l] is None]), axis=1)
        
        # residues influencing chemical modification
        self.overlap_dict = {j:[] for j in self.idx_i}
        for (i,n),v in self.pred_dict.items():
            for j in range(i,i+motif_length[self.idx_liabilities_map[n]]):
                self.overlap_dict[j].append(v)
        
        self.pred_df.loc[:,'overlaps_motif'] = self.pred_df.i.map(self.overlap_dict).apply(lambda x: ','.join(x))
        
        return self.pred_df
    
    def mat_to_table(self, x_pred_smm=None, x_pred_i=None):
        '''unpack predicted liabilities per mutation [LxAA] as human-readable table'''
        if x_pred_smm is None:
            x_pred_smm = self.x_pred_smm
        if x_pred_i is None:
            x_pred_i = self.x_pred_i
        pred_df = self.pred_to_table(x_pred_i)
        self.smm_df = pd.DataFrame(x_pred_smm, columns=self.idx_aa)
        self.smm_df['i'] = self.idx_i
        self.smm_df = pd.merge(pred_df, self.smm_df, on='i')
        return self.smm_df
