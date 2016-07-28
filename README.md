# vilds_interface

Code to interface between Matlab and the variational inference linear dynamical system (vilds) model [1,2] python implementation - https://github.com/earcher/vilds

### Usage example:

From Matlab:

```vildsi_export_spikes(outfile_for_spiking_data, y)```

- *outfile_for_spiking_data* - filename, will be written, store spiketrains in an HDF5 file 
- *y* -  binned spiketrains. 3-D Matlab array [nNeurons x nTimesteps x nTrials]

From the command line:

```python run_vilds.py outfile_for_spiking_data outfile_for_vilds_model outfile_for_vilds_estimates n_latents ffn_width```

- *outfile_for_spiking_data* - filename, output from previous step
- *outfile_for_vilds_model* - filename, will be written, stores the results of vilds model fit
- *outfile_for_vilds_estimates* - filename, will be written, stores estimates computed using the vilds model (e.g., posterior means)
- *n_latents* - dimensionality of the LDS to fit
- *ffn_width* - width of the feed-forward network that maps latents onto observed firing rates


Back in Matlab:

```results = vildsi_import_results(outfile_for_vilds_estimates)```

- *outfile_for_vilds_estimates* - filename, output from previous step


## Components
### Matlab code: 
`vildsi_export_spikes.m` - outputs spikes to an hdf5 file for easy import into Python

`vildsi_import_results.m` - takes the resulting output and makes it easily parse-able in Matlab

### Python code:
`run_vilds.py` - reads in data, calls vilds, saves results to file

NOTE: You must edit run_vilds.py to specify the path to the vilds codepack


> 1. E Archer, IM Park, L Buesing, J Cunningham, L Paninski (2015). [Black box variational inference for state space models](http://arxiv.org/abs/1511.07367)
> 2. Y Gao, E Archer, L Paninski, J Cunningham (2016). [Linear dynamical neural population models through nonlinear embeddings](http://arxiv.org/abs/1605.08454)

