from multiannealing.score.ChemLiabilities import ChemLiabilities
from multiannealing.score.TepitopesMatrix import TepitopesMatrix
from multiannealing.score.EVHmodel import EVHmodel
from multiannealing.score.MultiScore import MultiScore
from glob import iglob
import dill
import yaml

####################################################################################
# prepare scorefxn objects #########################################################
####################################################################################
default_gene_weights = {'DRB1':1.0,'DP':0.333,'DQ':0.333,
                        'DRB3':0.333,'DRB4':0.333,'DRB5':0.333}


# RUN
with open('Work/input/job_data.yml','r') as f:
	job_data = yaml.safe_load(f)


for file in iglob('Work/evcouplings/Q*mcc*/couplings/*.model'):
	print('\nLOADING EVCOUPLINGS')
	print(f'\t{file}')
	try:
		evh = EVHmodel(
		    seq=job_data['target_seq'],
		    seq_offset=job_data['index1'],
		    couplings_model_file=file
		)
		print('success')
		evh.save(file.replace('.model','.dill'))
		del evh
	except:
		print('failed')
		pass