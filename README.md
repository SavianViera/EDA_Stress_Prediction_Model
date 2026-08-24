# Project Journal
- 8/5/26: the model was learning to classify the trials, not the emotions?
- 8/7/26: recreated the encoder/decoder and loss function from Birkenmaier (testing accuracy with 5 fold CV currently at 84%)
  - Implemented feature design and optimized the loss function with a parameter grid search
  - Initial accuracy on binary valence is 84%
  - Increased window size to 60s and increased to 87.2%
  - tasks to do (unordered):
    - ~~artifact smoothing~~ Second: Implemented accuracy increase to 87.8%, when removing S17 accuracy is 91%, unsure of issue with participant, will analyze
    - cvxeda parameter tuning
    - ~~switch from k-fold to loocv~~ First: Completed, accuracy stayed stable at approximately 87% accuracy, though notably struggled with S17
    - updated lambda parameter search for 60s windows
    - machine learning parameter search 
-

# EDA Exploration

- `Data`
  - `raw`:
      - `participants_combined.csv`: raw data from original EDA
      - `eda_5m32s_23ptcp_shift.mat`: raw data updated to correct start point
  - `processed`: 
      - `Window_10_No_Overlap.csv`: made by `Function_For_Window_And_Overlap_Dataframe.R` 10 seconds of data per patient with no overlap on conditions
  - 
      - `MatLab_EDA_For_R.csv` made by `Converting_MatLabFile_To_R` to make the `eda_5m32s_23ptcp_shift.mat` file usable in R
      - `CVX_Phasic.csv` is the raw phasic signal for each patient using the cvxEDA package in Python
      - `CVX_Tonic.csv` is the raw tonic signal for each patient using the cvxEDA package in Python
      - `Phasic_Reshaped.csv` is the window converted phasic signal for each patient using the cvxEDA package in Python
      - `Tonic_Rehaped.csv` is the window converted tonic signal for each patient using the cvxEDA package in Python
      - `Final_Result_All_Signals_Combined.csv` is a concatenation of all the signals with phasic, tonic, and raw for the specified window

- `Scripts`
  - `Python_Scripts`:
    - `Machine_Learning_EDA_Stimulated_or_Not_Stimulated.ipynb`: test run exploring feature importance and baseline prediction without tuning boosting
    - `01cvxEDA.ipnyb` is a jupyter notebook which contains the cvxEDA package applied and phasic and tonic datasets for cvxEDA creation
    - `cvxEDA_Machine_Learning_EDA_Stimulated_or_Not_Stimulated.ipynb` is a test run to explore feature important and baseline prediction without tuning with the tonic and phasic signals generated from cvxEDA
    - eda_artifact_detection.py is to remove noise from the data by categorizing true SCR signals with MAD (Median Absolute Deviation) and linear interpolation for continuity, I also implemented as high-pass filter here as well.
    - eda_supervised_autoencoder.py is the supervised autoencoder for a CNN, we use the encoder and decoder to predict the ability to recreate the signal and use the weights from this model as features for the full model, has a custom loss function which makes it "supervised" that weights the loss function with a classification for higher accuracy
    - eda_statistical_features.py creates the tabular data for added accuracy
    - eda_hybrid_pipeline.py serves to do grid search optimization of models with a k-fold pipeline
    - eda_loso_pipeline.py is the final step in the script pipeline and is loso cross validation for most stable accuracy results and also informs which participants are difficult to classify
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
