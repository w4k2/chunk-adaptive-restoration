# Chunk Adaptive Restoration

Following repositoy contains code for paper *Employing chunk size adaptation to overcomeconcept drift*.

This repo is organised as follows:

* detectors folder contains code for fhdsdm drift detector
* evaluator directory contains implementation of metrics
* results folder contains .npy files with metrics values collected during experiments
* streams folder contains code with wrappers for stream-learn data streams, insects and usnet. It also stores .arff files with datastreams for experiements
* config.py contains dictionary with streams and parameters used for experiments
* run.py is scirpt that runs all experiments and saves results.

## Results

In paper following results were reported:

|algorithm| SR(0.9)<br/> pvales| SR(0.8)<br/> pvales| SR(0.7)<br/> pvales|
|---------|------|-------|-------|
|  WAE    | 0.005 | 0.101 | 0.031| 
|  AUE    | 0.001 | 0.000 | 0.000 | 
|  AWE    | 0.014 | 0.012 | 0.018 | 
|  SEA    | 0.000 | 4.573e-05 | 0.000 | 

We encourage to reproduces this results with procedure described in next section.

## Reproduction

This section contains steps necesseary for code reproduciton.

### Dependencies

Before runing any python code please create and source conda environment:

```
conda env create -f environment.yml
conda activate chunk-adaptive-restoration
```

### Data download

Plese download USP data stream repository from link: https://sites.google.com/view/uspdsrepository
Unpack .zip file and move upacked repository folder into streams in this repo.

### Run experiments

To execute experiments simply run:
```
python run.py
```

Evaluation metrics and logs are printed on the console.
Additionaly script will generate .npy files in results folder with 2 variants for each learning algorithm: baseline and ours.
Also plots will be generated in plots directory.
This files can be than used to genrated tex tables with folowing scritps:

```
python save_tex_table.py
python stats.py
```

This will generate .tex files in tables folder. This can be maunaly verified or copy-pasted into overleaf.