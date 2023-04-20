# AliceJetAI

## Introduction
ML tools are used to gain insight into the modification of parton showers of QGP. The Long Short-Term Memory is a neural network that is able to handle variable-length data sequences. Using the LSTM and OC-SVM to perform Unsupervised Anomaly Detection of Quenched Jets. Combining LSTM and OC-SVM was proposed by Tolga Ergen in his paper 'Unsupervised Anomaly Detection With LSTM Neural Networks' found at https://ieeexplore.ieee.org/document/8836638. LSTM is given jet data (or other variable length sequenced data), its output is the hidden states of the LSTM and are pooled to a fixed length data sequence (mean, first and last pooling), which can then be given to the OC-SVM for rating/classification. Training sequences have been proposed by Tolga Ergen for the combined LSTM and OC-SVM model, of which the Quadratic Programming-Based Training Sequence has been implemented in this project.

## Data
Jet data is obtained using JetToyHI from Marta Verweij: https://github.com/mverwe/JetToyHI.
For this project we have used the branch 'strong2020' and modified versions of the script `runSoftDrop.cc`. These are 'runSoftDropSkinny.cc' which removes most parts that are not required from the `runSoftDrop.cc` script. It includes Particle ID values required to separate quark and gluon jets, to be used for quark-gluon jet discrimination tests. Mixtures of quark and gluon jets can be made during any training (regular or hyper), or by running `analysis/mix_quark_gluon_samples.py`.
The second script is `runSoftDropTiny.cc`, similar to `runSoftDropSkinny.cc` but omitted the Particle ID part. This script can be used on Jewel datasets (Jewel data does not have particle IDs so `runSoftDropSkinny.cc` will give errors), for which we have used the HEPMC data from bhofman.
Scripts could (should?) be included with this repository (or JetToyHI) at a later point (I currently don't have access to them) and are currently found in `/project/alice/users/wesselr/.../Gijs/JetToyHI/` (not 100% sure about this exact location). 
Alternatively, create modified versions of `runSoftDrop.cc` storing Particle ID data of jet initiators, using CA, and including: `sigJetRecur_jetpt`, `sigJetRecur_z`, and `sigJetRecur_dr12`. Ask bhofman for help in using HEPMC files with JetToyHI.

Additionally, to test the model performance, Pen-Based Recognition of Handwritten Digits dataset has been used, found at https://archive.ics.uci.edu/ml/datasets/pen-based+recognition+of+handwritten+digits. 

Test sample datasets have been included in `samples/`:
- JetToyHIResultSoftDropSkinny.root
- JetToyHIResultSoftDropTiny_simple-1.root
- pendigits

## Installation
Make sure the correct versions of Python packages are installed, most importantly Pytorch.
Check `requirements.txt` file for these. Additionally, refer to `setup.py` and run `pip install -e .` from the terminal to install packages.

## How to use?
*. With any training session, cost and cost condition plots are generated, use these to judge training performance. These can also be created with `analysis/make_cost_condition_plots.py`. Output of plotting is found in `output/`.
1. Hypertraining is used to find parameters resulting in low cost scores. This can be performed with `analysis/do_hyper_training.py`. You can set datasets/ranges/nr.oftrials in the script. Output gives best parameters, but further analysis on parameter values can be done using Violin plots: run 'analysis/make_violin_plots.py'. Hypertraining can be run multicore and use GPU, this can be set in the script (make sure you are able to run multicore processes), and this works well on the Nikhef Stoomboot by submitting the scripts `submit_job_hyper_training.sh` and `submit_job_multc_hyper.sh` both creating jobs of `job_hyper_training.sh`. Logfiles will be stored in `logfiles/` containing the printouts from the script. Running multicore will create batches of trials, running parallel on individual cores, cost evaluation and subsequent paramater selection will be performed after each batch. Trained models and run info is stored in `storing_results/`.
2. A normal training session can be performed with the `analysis/do_regular_training.py` script. Use best found parameters, set in the hyperspace given in the script. Use GPU for speed increase, multicore not implemented here. Runs on stoomboot with `job_regular_training.sh` and `submit_job_reg.sh`. Trained models and run info is stored in `storing_results/`.
3. Anomaly testing of trained models can be performed with `analysis/get_anomalies.py`. Here, ROC curves and single variable distributions can be made with anomalous distribution stacked on top of the normal (one-class) distribution. 
4. Testing of proposed LSTM+OC-SVM model can be done with the Digits samples. Run `testing/do_regular_training_digits.py` and use the results in `testing/get_anomalies_digits.py`. This outputs ROC curves and 2D plots of the distribution of pooled hidden states, showing (in)correct anomalous/normal data.
5. Other testing tools can be found under `testing/`, to make Lund planes, for instance. This folder is a bit messy, but I'll leave it there for anyone interested in messing more with the model.

## Questions?
Feel free to contact me at wesselrijk@gmail.com
