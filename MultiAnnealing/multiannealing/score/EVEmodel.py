import dill
import numpy as np
import os,sys
import json
import argparse
import pandas as pd
import torch

from EVE import VAE_model
from utils import data_utils


class EVEmodel:
    def __init__(self, seq, seq_offset, 
                 msa_list, protein_index, msa_data_folder, msa_weights_location,
                 threshold_sequence_frac_gaps, threshold_focus_cols_frac_gaps, 
                 model_name_suffix, 
                 model_parameters_location, vae_checkpoint_location,
                 num_samples_compute_evol_indices, batch_size,
                 quiet=False): # need to get rid of caps after i copy it elsewhere
        
        self.num_samples = num_samples_compute_evol_indices
        self.batch_size = batch_size
        self.mapping_file = pd.read_csv(msa_list)
        self.protein_name = self.mapping_file['protein_name'][protein_index]
        self.msa_location = msa_data_folder + os.sep + self.mapping_file['msa_location'][protein_index]
        print("Protein name: "+str(self.protein_name))
        print("MSA file: "+str(self.msa_location))
        
        self.theta = float(self.mapping_file['theta'][protein_index])
        print(self.theta)
        print("Theta MSA re-weighting: "+str(self.theta))
        
        self.data = data_utils.MSA_processing(
                MSA_location=self.msa_location,
                theta=self.theta,
                use_weights=True,
                weights_location=msa_weights_location + os.sep + self.protein_name + '_theta_' + str(self.theta) + '.npy',
                threshold_sequence_frac_gaps=threshold_sequence_frac_gaps,
                threshold_focus_cols_frac_gaps=threshold_focus_cols_frac_gaps
        )
        
        
        self.model_name = self.protein_name + "_" + model_name_suffix
        print("Model name: "+str(self.model_name))

        self.model_params = json.load(open(model_parameters_location))

        self.model = VAE_model.VAE_model(
                        model_name=self.model_name,
                        data=self.data,
                        encoder_parameters=self.model_params["encoder_parameters"],
                        decoder_parameters=self.model_params["decoder_parameters"],
                        random_seed=42
        )
        self.model = self.model.to(self.model.device)
        
        self.checkpoint_name = str(vae_checkpoint_location) + os.sep + self.model_name + "_final"
        checkpoint = torch.load(self.checkpoint_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Initialized VAE with checkpoint '{}' ".format(self.checkpoint_name))
        
        self.idx_i = self.data.focus_start_loc+np.array(self.data.focus_cols)
        self.idx_aa = np.array(list(self.data.alphabet))
        self.model_seq_map = {self.data.focus_start_loc+x:y for x, y in zip(self.data.focus_cols,self.data.focus_seq_trimmed)}
        self.target_seq = seq
        self.seq = seq
        self.seq_offset=seq_offset
        self.seq_df = pd.DataFrame({
            'i':seq_offset+np.arange(len(seq)), 'aa':list(seq)})
    
        self.seq_df['modeled'] = self.seq_df.i.isin(self.idx_i)
        self.seq_df['aa_model_target'] = self.seq_df.i.map(self.model_seq_map)
        
        if not quiet:
            matches = self.seq_df.aa == self.seq_df.aa_model_target
            print(f'{matches.mean()*100:.0f}% match to EVEModel', f'{(~matches).sum()} mismatches')
            print(f'{sum(~self.seq_df.modeled)} residues missing in EVEModel')
            
    def predict_mutscan(self, seq, seq_offset):
        d = self.create_all_singles(seq, seq_offset)
        list_mutations, mean_predictions, _ = self.compute_all_singles(d)
        df = {}
        df['mutations'] = list_mutations
        df['mean_predictions'] = mean_predictions
        df = pd.DataFrame(df)
        pred_mutscan = df.mean_predictions.values.reshape(-1,20)
        return pred_mutscan

    def predict(self, seq, seq_offset):
        mean_prediction, _ = self.compute_one_seq(seq, seq_offset)
        return mean_prediction

    def predict_delta(self, seq, seq_ref, seq_offset):
        new_mean_prediction, _ = self.compute_one_seq(seq, seq_offset)
        ref_mean_prediction, _ = self.compute_one_seq(seq_ref, seq_offset)
        delta = new_mean_prediction - ref_mean_prediction
        return delta

    def conform_seq_to_model(self, seq, seq_offset, validate=False): # later, use self.data instead
        '''extract focus column positions from provided sequence,
        via full length uniprot indices in couplings model'''
        start_idx = self.data.focus_start_loc
        L = len(self.data.focus_seq_trimmed)
        if len(seq) == L:
            return seq
        if seq_offset is None:
            seq_offset = self.seq_offset
        if validate:
            assert(self.data.focus_cols[0]-seq_offset >= 0)
            assert(self.data.focus_cols[-1]-seq_offset < len(seq))

        trunc_seq = seq[start_idx-seq_offset:]
        seqx = ''.join(np.array(list(trunc_seq))[self.data.focus_cols])
        return seqx

    def create_all_singles(self, seq, seq_offset): #seq_offset is uniprot index start of design sequence; here it is the same as the model
        start_idx = self.data.focus_start_loc

        # need to adjust with seq_offset and start_idx so my "seq" is now my starting sequence from which i built my alignment and model
        trunc_seq = seq[start_idx-seq_offset:]

        mutant_to_focuscolsseq = {}

        for i, letter in enumerate(trunc_seq):
            if i in self.data.focus_cols:
                for mut in self.data.alphabet:
                    pos = start_idx + i
                    mutant = letter+str(pos)+mut
                    mut_seq = list(trunc_seq).copy()
                    mut_seq[i] = mut 
                    # above is the mutated truncated version, now need to take just the focus columns
                    mut_focuscolsseq = ''.join(np.array(list(mut_seq))[self.data.focus_cols])
                    mutant_to_focuscolsseq[mutant] = mut_focuscolsseq
        return mutant_to_focuscolsseq

    def compute_all_singles(self, singles_dict):
        #One-hot encoding of mutated sequences
        mutated_sequences_one_hot = np.zeros((len(singles_dict),len(self.data.focus_cols),len(self.data.alphabet)))
        for i,mutation in enumerate(singles_dict.keys()):
            sequence = singles_dict[mutation]
            for j,letter in enumerate(sequence):
                if letter in self.data.aa_dict:
                    k = self.data.aa_dict[letter]
                    mutated_sequences_one_hot[i,j,k] = 1.0

        mutated_sequences_one_hot = torch.tensor(mutated_sequences_one_hot)
        dataloader = torch.utils.data.DataLoader(mutated_sequences_one_hot, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        prediction_matrix = torch.zeros((len(singles_dict),self.num_samples))

        with torch.no_grad():
#             for i, batch in enumerate(tqdm.tqdm(dataloader, 'Looping through mutation batches')):
            for i, batch in enumerate(dataloader):
                x = batch.type(self.model.dtype).to(self.model.device)
    #             for j in tqdm.tqdm(range(self.num_samples), 'Looping through number of samples for batch #: '+str(i+1)):
                for j in range(self.num_samples):
                    seq_predictions, _, _ = self.model.all_likelihood_components(x)
                    prediction_matrix[i*self.batch_size:i*self.batch_size+len(x),j] = seq_predictions
#                 tqdm.tqdm.write('\n')
            mean_predictions = prediction_matrix.mean(dim=1, keepdim=False)
            std_predictions = prediction_matrix.std(dim=1, keepdim=False)
    #         delta_elbos = mean_predictions - mean_predictions[0]
    #         evol_indices =  - delta_elbos.detach().cpu().numpy()

        return singles_dict.keys(), mean_predictions.detach().cpu().numpy(), std_predictions.detach().cpu().numpy()

    def compute_one_seq(self, seq, seq_offset):
        # this assumes you're getting the 
        #One-hot encoding of mutated sequences
        mutated_sequences_one_hot = np.zeros((1,len(self.data.focus_cols),len(self.data.alphabet)))

        sequence = self.conform_seq_to_model(seq, seq_offset=seq_offset)
        for j,letter in enumerate(sequence):
            if letter in self.data.aa_dict:
                k = self.data.aa_dict[letter]
                mutated_sequences_one_hot[0,j,k] = 1.0

        mutated_sequences_one_hot = torch.tensor(mutated_sequences_one_hot)
        dataloader = torch.utils.data.DataLoader(mutated_sequences_one_hot, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        prediction_matrix = torch.zeros((1,self.num_samples))

        with torch.no_grad():
#             for i, batch in enumerate(tqdm.tqdm(dataloader, 'Looping through mutation batches')):
            for i, batch in enumerate(dataloader):
                x = batch.type(self.model.dtype).to(self.model.device)
    #             for j in tqdm.tqdm(range(num_samples), 'Looping through number of samples for batch #: '+str(i+1)):
                for j in range(self.num_samples):
                    seq_predictions, _, _ = self.model.all_likelihood_components(x)
                    prediction_matrix[i*self.batch_size:i*self.batch_size+len(x),j] = seq_predictions
#                 tqdm.tqdm.write('\n')
            mean_predictions = prediction_matrix.mean()
            std_predictions = prediction_matrix.std()
    #         delta_elbos = mean_predictions - mean_predictions[0]
    #         evol_indices =  - delta_elbos.detach().cpu().numpy()

        return mean_predictions.detach().cpu().numpy(), std_predictions.detach().cpu().numpy()
