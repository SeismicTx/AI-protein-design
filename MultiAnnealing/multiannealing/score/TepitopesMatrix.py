import os
import dill
import pandas as pd
import numpy as np
from copy import deepcopy

def mutation(seq, mut, index_offset=1):
    if isinstance(mut, str):
        mut = (mut[0], int(mut[1:-1]), mut[-1])
    aa_wt, i, aa_mut = mut
    assert seq[i-index_offset] == aa_wt
    seq = seq[:i-index_offset] + aa_mut + seq[i-index_offset+1:]
    return seq

def mutations(seq, muts, index_offset):
    mut_seq = deepcopy(seq)
    for mut in muts:
        mut_seq = mutation(mut_seq, mut, index_offset)
    return mut_seq

def relu(x,x0=0):
    xc = x-x0
    return xc*(xc>0)

IdeS = '''
VTSVWTKGVTPPANFTQGEDVFHAPYVANQGWYDITKTFNGKDDLLCGAATAG
NMLHWWFDQNKDQIKRYLEEHPEKQKINFNGEQMFDVKEAIDTKNHQLDSKLF
EYFKEKAFPYLSTKHLGVFPDHVIDMFINGYRLSLTNHGPTPVKEGSKDPRGG
IFDAVFTRGDQSKLLTSRHDFKEKNLKEISDLIKKELTEGKALGLSHTYANVR
INHVINLWGADFDSNGNLKAIYVTDSDSNASIGMKKYFVGVNSAGKVAISAKE
IKEDNIGAQVLGLFTLSTGQDSWNQTN
'''.replace('\n','').upper()

class TepitopesMatrix:
    '''Given netMHCIIpan precomputed mutation scan, allele populations table, and the first index of seq scanned by netMHCIIpan
    Create a scoring object that produces MHC-II epitope display scores for seq variants
    
    Uses results of a netMHCIIpan scan (15mers across full length with 1aa-step, all 19x15 mutants per window) to build a
    pre-computed matrix of mutation effects on epitope probability. This is used to compute scores that *approximate*
    a complete netMHCIIpan prediction.
    
    Behavior:
    - For each mutant, the top prediction per MHC-II binding core is contrasted to the top prediction per binding core for wildtype
    - netMHCIIpan doesn't return a hit if the display probability is too low, so we treat those as Prob = 0.
    - The various predictors in netMHCIIpan have different false-positive rates. To compare them, we use the E-value (%Rank) and transform to linear scale.
    - Best approximation for sequences with high identity to the original scanned sequence.
    - To avoid redundant mutations that reduce the probability of an epitope that's already p=0, a ReLu is applied per epitope probability to ensure p>=0. 

    Note: 
    Regions of the sequence in precomputed scan that are outside of the input sequence are assumed to be identical to wildtype used in the precomputed scan.
    Ideally it is best to perform the netMHCIIpan scan using a sequence range identical to that you intend to produce as physical protein product.

    Adhere to the rules of a score class:
    - self.predict(seq, seq_offset) [1]
    - self.predict_mutscan(seq, seq_offset) [LxAA]
    - self.mat_to_table(mat) [pd.DataFrame]
    self.idx_i [L], self.idx_aa [LxAA], self.seq_df (cols: i,aa)
    More: self.model (evcouplings.model object), self.conform_seq_to_model (strip to focus columns) 
    '''
    def __init__(self, netmhciipan_file=None, allele_populations_file=None, delta_P_smm_file=None, P_wt_file=None, seq_offset=None,
                score_col='%Rank_EL'):
        self.score_col = score_col
        if netmhciipan_file is not None:
            self.netmhciipan_file = netmhciipan_file
            self.load_predicted_epitopes(netmhciipan_file)
            self.prepare_predicted_epitopes(seq_offset)
            if allele_populations_file is not None:
                self.allele_populations_file = allele_populations_file
                self.load_epitope_populations(allele_populations_file)
                self.precompute_mutscan(score_col=score_col)
                self.target_seq = IdeS
                self.seq_to_mat(
                    self.target_seq, self.seq_offset, update=True, compare=True)
                self.delta_P_smm_wt = np.load(delta_P_smm_file)
                self.P_wt = np.loadtxt(P_wt_file, delimiter=",")
                self.predict()
                self.predict_mutscan()

    def save(self, outfile, small=True):
        '''saves scorefxn object as compressed dill file'''
        if small:
            self_small = deepcopy(self)
            del self_small.epitope_scores
            del self_small.epitope_scores_dd
            with open(outfile,'wb') as f:
                dill.dump(self_small, f)
            del self_small
        else:
            with open(outfile,'wb') as f:
                dill.dump(self, f)

    def load_predicted_epitopes(self, epitope_file):
        '''Loads netMHCIIpan results file. Can take time.'''
        if isinstance(epitope_file, str):
            print(f'reading file {os.path.getsize(epitope_file)/(1024*1024)} MB')
            self.epitope_scores = pd.read_csv(epitope_file)
            print('reading complete')
        else:
            print('non-string found, attempting to copy pre-loaded dataframe')
            print('if making multiple matrices from the same scan, more efficient to initialize using the de-duplicate the 15mers')
            assert(isinstance(epitope_file, pd.DataFrame))
            self.epitope_scores = deepcopy(epitope_file)
        
    def prepare_predicted_epitopes(self, seq_offset, muts=True):
        '''Adjust to target indexing, relabel mutations, add same K-mer wildtype label to each mutant'''
        self.seq_offset  = seq_offset

        self.epitope_scores['MHC_gene'] = self.epitope_scores.MHC.apply(
            lambda x: [gene for gene in ['DRB1','DRB3','DRB4','DRB5','DP','DQ'] if gene in x][0])

        # adjust indexing
        self.epitope_scores['pep_start'] += self.seq_offset - 1
        self.epitope_scores['pep_end'] += self.seq_offset - 1
        self.epitope_scores['Core_Pos'] = self.epitope_scores['pep_start'] + self.epitope_scores['Of']

        if muts:
            print('steps to load mutants')
            mut_list = self.epitope_scores['mut'].dropna().drop_duplicates()
            mut_map = {m: f'{m[0]}{int(m[1:-1])+self.seq_offset-1}{m[-1]}' for m in mut_list if m!='wt'}
            self.epitope_scores['mut'] = self.epitope_scores['mut'].map(mut_map).fillna('wt')
            print('updated mutant information')
            self.epitope_scores['wt_name'] = self.epitope_scores.pep_start.apply(lambda x: f'wt/{x}-{x+14}')
            wt_rows = self.epitope_scores.mut.eq('wt')
            self.epitope_scores['wt'] = wt_rows
            self.epitope_scores.loc[wt_rows,'mut'] = self.epitope_scores[wt_rows].wt_name

    def precompute_mutscan(self, score_col='%Rank_EL', xform=lambda x: 2 / (1 + np.exp(x * 0.575))):
        '''create an [A,LxAA,E] mut matrix per allele [A], the effect of a mutant [LxAA] on an epitope [E]
        - important to work from the rank score, because it adjusts by the null distribution
        but requires we provide a xform to revert the E-value into an additive metric
        - in the original scan, there are redundant predictions (alternate 15mers scored per epitope),
        so we de-duplicate at the Allele x Mutant x Epitope level, keeping the highest scoring 15mer'''
        print('computing score')
        self.score_transform = xform
        self.epitope_scores['epitope_score'] = xform(self.epitope_scores[score_col])
        self.epitope_scores_dd = deepcopy(
            self.epitope_scores.sort_values('epitope_score', ascending=False
                                            ).drop_duplicates(['MHC', 'mut', 'Core_Pos']))


    def load_epitope_populations(self, populations_file, fill_value=None, normalize=False):
        '''load population data into allele->weight dictionary'''
        self.allele_pops = pd.read_csv(populations_file)
        
        if fill_value is not None:
            self.fill_rows = []
            for mhc in set(self.epitope_score.MHC) - set(self.allele_pops.MHC):
                self.fill_rows.append({'MHC':mhc, 'avg pop%':fill_value, 'fill':True})
            self.fill_rows = pd.DataFrame(self.fill_rows)
            self.allele_pops = pd.concat([self.allele_pops, self.fill_rows])
        
        self.allele_pops['MHC_gene'] = self.allele_pops.MHC.apply(
            lambda x: [gene for gene in ['DRB1','DRB3','DRB4','DRB5','DP','DQ'] if gene in x][0])
        
        if normalize:
            for gene in ['DRB1','DRB3','DRB4','DRB5','DP','DQ']:
                gene_rows = self.allele_pops.MHC_gene.eq(gene)
                self.allele_pops.loc[gene_rows,'avg pop% normed'] = \
                    self.allele_pops[gene_rows]['avg pop%'] / self.allele_pops[gene_rows]['avg pop%'].sum()
            self.weight_column = 'avg pop% normed'
        else:
            self.weight_column = 'avg pop%'
        
        self.weight_for_allele = dict(self.allele_pops.set_index('MHC')[self.weight_column])
        self.epitope_scores['P_allele'] = self.epitope_scores['MHC'].map(self.weight_for_allele)


    
    def seq_to_mat(self, seq, seq_offset=None, update=False, compare=False):
        if seq is None:
            return self.x
        elif isinstance(seq, np.ndarray) and len(seq.shape) == 2:
            return seq
        
        x = self.seq_to_mat_basic(seq)
        
        if seq_offset is not None and seq_offset!=self.seq_offset:
            assert seq_offset <= self.idx_i[0]
            x = x[self.idx_i-seq_offset]
            # idx_i is in uniprot coords, so this gives 0-index
            # given sequence must be superset of target region

        assert x.shape == (self.L, self.AA)
        
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
        x = np.array([self.aa_map[aa] for aa in seq])        
        return x

    def set_gene_weights(self, gene_weight_map='default'):
        if gene_weight_map == 'default':
            gene_weight_map = {'DRB1': 1.0, 'DP': 0.333, 'DQ': 0.333,
                               'DRB3': 0.333, 'DRB4': 0.333, 'DRB5': 0.333}
        self.gene_weight_map = gene_weight_map

        if self.gene_weight_map is None:
            self.gene_weights = np.array([1.0 for g in self.idx_mhcgenes])
        else:
            self.gene_weights = np.array([gene_weight_map[g] for g in self.idx_mhcgenes])

    def set_epitope_weights(self, epitope_weight_map=None, missing=0):
        self.epitope_weight_map = epitope_weight_map
        if self.epitope_weight_map is None:
            self.epitope_weights = np.ones(self.idx_cores.shape)
        elif isinstance(self.epitope_weight_map, dict):
            self.epitope_weights = np.array([epitope_weight_map.get(i, missing) for i in self.idx_cores])
        else:
            self.epitope_weights = np.array([epitope_weight_map(i) for i in self.idx_cores])

    def predict(self, seq=None, seq_offset=None, update=True, per_epitope=False, prob_floor=0):
        '''compute epitope display probs: P_seq_epitopes [genes,epitopes]
        & weighted scores: P_seq_genes [genes], P_seq [1]
        Return weighted score P_seq [1]'''
        x_seq = self.seq_to_mat(seq, seq_offset, update)
        # delta_P_smm_wt_x = MAT( delta_P_smm(wt)(i,aa) for i,aa in seq ) : [genes,epitopes,length]
        self.delta_P_smm_wt_x = np.sum(self.delta_P_smm_wt*x_seq, axis=-1)
        # epitope display probabilities
        self.P_seq_epitopes_raw = self.P_wt + np.sum(self.delta_P_smm_wt_x, axis=-1)
        self.P_seq_epitopes = relu(self.P_seq_epitopes_raw, prob_floor)
        # tally weighted scores
        self.P_seq_genes = np.sum(self.P_seq_epitopes * self.epitope_weights, axis=1)
        self.P_seq = np.sum(self.P_seq_genes * self.gene_weights)
        if per_epitope:
            return self.P_seq_epitopes
        return self.P_seq

    def predict_delta(self, seq, seq_ref=None, seq_offset=None, update=True, prob_floor=0):
        P_ref = self.predict(seq_ref, seq_offset, update=False, prob_floor=prob_floor)
        P_seq = self.predict(seq, seq_offset, update=update, prob_floor=prob_floor)
        return P_seq - P_ref

    def predict_mutscan(self, seq=None, seq_offset=None, update=True, prob_floor=0):
        '''compute delta_P_smm_epitopes [genes,epitopes,length,AAs]
        & weighted scores: delta_P_smm_genes [genes,length,AAs], delta_P_smm [length,AAs]
        return delta_P_smm [length,AAs]'''
        x_seq = self.seq_to_mat(seq, seq_offset, update=update)
        # compute per-epitope probability of given sequence ( Px = relu(Pwt + SUM(deltaPsmm_ia * x_ia)) )
        self.delta_P_smm_wt_x = np.sum(self.delta_P_smm_wt*x_seq, axis=-1) # [genes,cores,L]
        self.P_seq_epitopes_raw = self.P_wt + np.sum(self.delta_P_smm_wt_x, axis=-1)
        self.P_seq_epitopes = relu(self.P_seq_epitopes_raw, prob_floor) # [genes,cores]
        
        # compute per-epitope probability of single mutants of given sequence ( Pxsmm = relu((deltaPsmm - delptaPsmm*x) + (Px + SUM(deltaPsmm_ia * x_ia))) )
        self.P_smm_epitopes_raw = (self.delta_P_smm_wt - self.delta_P_smm_wt_x[:,:,:,np.newaxis]) + self.P_seq_epitopes_raw[:,:,np.newaxis,np.newaxis]
        self.P_smm_epitopes = relu(self.P_smm_epitopes_raw, prob_floor) # [genes,cores,L,aa]
        
        # compute delta ( deltaPxsmm = Pxsmm - Px )
        self.delta_P_smm_epitopes = self.P_smm_epitopes - self.P_seq_epitopes[:,:,np.newaxis,np.newaxis] # [genes,cores,L,aa]
        
        # weight epitopes & genes
        self.delta_P_smm_epitopes = self.P_smm_epitopes - self.P_seq_epitopes[:,:,np.newaxis,np.newaxis]  # [genes,cores,L,aa]
        self.delta_P_smm_genes = np.sum(self.delta_P_smm_epitopes * self.epitope_weights[np.newaxis,:,np.newaxis,np.newaxis], axis=1) # [genes,L,aa]
        self.delta_P_smm = np.sum(self.delta_P_smm_genes * self.gene_weights[:,np.newaxis,np.newaxis], axis=0) # [L,aa]

        return self.delta_P_smm