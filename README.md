# AI-protein-design
Code to reproduce the results in our manuscript "AI protein design enables rapid drug discovery and deimmunization". 

## Installation

First, download and set up the code base.

```
git clone https://github.com/SeismicTx/AI-protein-design.git
cd AI-protein-design
```

We highly recommend using conda to set up the environment correctly for reproducing the results of the paper. This will take about five minutes on a standard personal computer.

```
conda env create -f env.yml
conda activate multiannealing
pip install -r requirements.txt
cd MultiAnnealing
pip install .
cd ..
```
Next, download the model objects needed to reproduce the results of the paper.

```
wget https://seismictx-public.s3.us-east-2.amazonaws.com/AI-protein-design/netmhciipan/IdeS_48-391_demo_netMHCIIpan_BA.dill
wget https://seismictx-public.s3.us-east-2.amazonaws.com/AI-protein-design/netmhciipan/IdeS_48-391_demo_netMHCIIpan_EL.dill
wget https://seismictx-public.s3.us-east-2.amazonaws.com/AI-protein-design/evcouplings/Q9F1R7_STRPY_48-391_b0.60_msc70_mcc50/couplings/Q9F1R7_STRPY_48-391_b0.60_msc70_mcc50.dill

mkdir evcouplings

mv IdeS_48-391_demo_netMHCIIpan_BA.dill data/netmhciipan/
mv IdeS_48-391_demo_netMHCIIpan_EL.dill data/netmhciipan/
mv Q9F1R7_STRPY_48-391_b0.60_msc70_mcc50.dill data/evcouplings/

```

The full netMHCIIpan output described in the paper is also available at `https://seismictx-public.s3.us-east-2.amazonaws.com/AI-protein-design/netmhciipan/IdeS_48-391_demo_netMHCIIpan.csv`. 

To reproduce the results of the paper, follow the steps in the notebook `MultiAnnealing/notebooks/IdeS_simulated_annealing.ipynb`. This should take about ten minutes on a standard personal computer.