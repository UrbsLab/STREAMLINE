# Explaining STREAMLINE Output
This section describes the files in the experiment directory with reference to a 
sample run on the [demonstration dataset](sample.md#demonstration-data)

If this is a demo run the saved experiment folder that was output by the run, called `demo`. 
If you changed the `experiment_name` parameter or if this is a custom run it may be different and be named the same as
`experiment_name` your parameter.

Within this folder you should find the following:

* `<experiment_name>_ML_Pipeline_Report.pdf` [File]: This is an automatically formatted PDF summarizing key findings during the model training and evaluation. A great place to start!
* `metadata.csv` [File]: Another way to view the STREAMLINE parameters used by the pipeline.  These are also organized on the first page of the PDF report.
* `metadata.pickle` [File]: A binary 'pickle' file of the metadata for easy loading by the 11 pipeline phases. (For more experienced users)
* `algInfo.pickle` [File]: A binary 'pickle' file including a dictionary indicating which ML algorithms were used, along with abbreviations of names for figures/filenames, and colors to use for each algorithm in figures. (For more experienced users)
* `DatasetComparisons` [Folder]: Containing figures and statistical significance comparisons between the two datasets that were analyzed with STREAMLINE. (This folder only appears if more than one dataset was included in the user specified data folder, i.e. `data_path`, and phase 7 of STREAMLINE was run). Within the PDF summary, each dataset is assigned an abbreviated designation of 'D#' (e.g. D1, D2, etc) based on the alphabetical order of each dataset name. These designations are used in some of the files included within this folder.
* [Folders] - A folder for each of the two datasets analyzed (in this demo there were two: `demodata` and `hcc-data_example_no_covariates`). These folders include all results and models respective to each dataset. We summarize the contents of each folder below (feel free to skip this for now and revisit it as needed)...
    * `exploratory` [Folder]: Includes all exploratory analysis summaries and figures.
    * `CVDatasets` [Folder]: Includes all individual training and testing datasets (as .csv files) generated.
        * Note: These are the datasets passed to modeling so if imputation and scaling was conducted, these datasets will have been partitioned, imputed, and scaled.
    * `scale_impute` [Folder]: Includes all pickled files preserving how scaling and/or imputation was conducted based on respective training datasets.
    * `feature_selection` [Folder]: Includes feature importance and selection summaries and figures.
    * `models` [Folder]: Includes the ML algorithm hyperparameters selected by Optuna for each CV partition and modeling algorithm, as well as pickled files storing all models for future use.
    * `model_evaluation` [Folder]: Includes all model evaluation results, summaries, figures, and statistical comparisons.
    * `applymodel` [Folder]: Includes all model evaluation results when applied to a hold out replication datasets. This includes a new PDF summary of models when applied to this further hold-out dataset.
        * Note: In the demonstration analysis we only created and applied a replication dataset for `hcc-data_example`. Therefore, this folder only appears in output folder for `hcc-data_example`.
    * `runtimes.csv` [File]: Summary file giving the total runtimes spent on each phase or ML modeling algorithm in STREAMLINE.
