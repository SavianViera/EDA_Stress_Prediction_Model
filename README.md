# Readme
## Instruction
Download WESAD data @ "https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html" and place underneath "Data" folder
## Code
In "Python_code" folder
- [`load_eda.ipynb`](Python_code/load_eda.ipynb) produces `wesad_subject_list.pkl` which holds a list of dataframes of each participant
- [`Signal_Cleaning.ipynb`](Python_code/Signal_Cleaning.ipynb) applies a Butterworth filter onto each EDA signal, per subject normalization is then conducted, lastly the signals and their respective labels are concatenated into a final dataframe with NAs removed.
- [`Decomposition_of_Signals.ipynb`](Python_code/Decomposition_of_Signals.ipynb) decomposes the signals from `filtered_eda_signals.csv` into phasic and tonic matrices. Produces `phasic_signal.csv` and `tonic_signal.csv`


## Data
- WESAD data should be placed here
### wesad_eda
- [`wesad_subject_list`](Data/wesad_eda/wesad_subject_list) is a list of dataframes for each participant's EDA signal and labels
- [`filtered_eda_signals.csv`](Data/wesad_eda/filtered_eda_signals.csv) is the concatenated filtered and normalized lists of EDA signals
- [`phasic_signal.csv`](Data/wesad_eda/phasic_signal.csv) is phasic decomposition of EDA signals
- [`tonic_signal.csv`](Data/wesad_eda/tonic_signal.csv) is the tonic decomposition of EDA signals

