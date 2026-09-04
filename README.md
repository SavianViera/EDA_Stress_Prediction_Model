# Readme
## Instruction
Download WESAD data @ "https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html" and place underneath "Data" folder
## Code
In "python_code" folder
- `load_eda.ipynb` produces `wesad_subject_list.pkl` which holds a list of dataframes of each participant
- `Signal_Cleaning.ipynb` applies a Buttersworth filter onto each EDA signal, per subject normalization is then conducted, lastly the signals and their respective labels are concatenated into a final dataframe `` with NAs removed.

## Data
- WESAD data should be placed here
  # wesad_eda
  - `wesad_subject_list` is a list of dataframes for each participant's EDA signal and labels
