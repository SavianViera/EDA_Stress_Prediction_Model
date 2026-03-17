# EDA Exploration

- `Data`
  - `raw`:
      - `participants_combined.csv`: raw data from original EDA
      - `eda_5m32s_23ptcp_shift.mat`: raw data updated to correct start point
  - `processed`: 
      - `Window_10_No_Overlap.csv`: made by `Function_For_Window_And_Overlap_Dataframe.R` 10 seconds of data per patient with no overlap on conditions
      - `MatLab_EDA_For_R.csv` made by `Converting_MatLabFile_To_R` to make the `eda_5m32s_23ptcp_shift.mat` file usable in R

- `Scripts`
  - `Python_Scripts`:
    - `Machine_Learning_EDA_Stimulated_or_Not_Stimulated.ipynb`: test run exploring feature importance and baseline prediction without tuning boosting
  - `R_Scripts`:
        -`Scripts_for_EDA`:
          - `Converting_MatLabFile_To_R`: processes `eda_5m32s_23ptcp_shift.mat` into a data-frame readable in R
          - `Function_For_Window_And_Overlap_Dataframe.R` takes the `eda_5m32s_23ptcp_shift.mat` data-frame and transforms it into a long data-frame with definable window of time per patient and user definable sensitivity on overlapping emotional states. 
          - `Feature_Engineering_10_Window_Dataset.R` examines descriptive features of the signals for the 10 window no overlap dataset
## Notes from Chen
- fpp2 package for R may be useful
- Add commentary of the ACF plots
- Add commentary for the standardized EDA by conditions plot
- Explore means to describe signal with statistics to gain power to relate a signal to a description (considered fourier transform, however, we lose the time point when converting. This defeats the point of a time analysis. If we fourier transform each condition can we compare those?)
