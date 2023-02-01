![alttext](https://github.com/UrbsLab/STREAMLINE/blob/main/Pictures/STREAMLINE_LOGO.jpg?raw=true)
# Overview
STREAMLINE is an end-to-end automated machine learning (AutoML) pipeline that empowers anyone 
to easily run, interpret, and apply a rigorous and customizable analysis for data mining 
or predictive modeling. Notably, this tool is currently limited to supervised learning on 
tabular, binary classification data but will be expanded as our development continues. The 
development of this pipeline focused on 
1) overall automation
2) avoiding and detecting sources of bias 
3) optimizing modeling performance
4) ensuring complete reproducibility (under certain STREAMLINE parameter settings)
5) capturing complex associations in data (e.g. feature interactions)
6) enhancing interpretability of output. 

Overall, the goal of this pipeline is to provide a transparent framework to learn from data as well as identify 
the strengths and weaknesses of ML modeling algorithms or other AutoML algorithms.

A preprint introducing and applying STREAMLINE is now available 
[here](https://arxiv.org/abs/2206.12002?fbclid=IwAR1toW5AtDJQcna0_9Sj73T9kJvuB-x-swnQETBGQ8lSwBB0z2N1TByEwlw).

See [CITATIONS.md](./CITATIONS.md) for how to cite this preprint and/or the codebase prior to the availability of the final 
peer-reviewed publication.

## STREAMLINE Schematic

This schematic breaks the overall pipeline down into 4 basic components: 
(1) preprocessing and feature transformation, 
(2) feature importance evaluation and selection, 
(3) modeling, and 
(4) postprocessing.

![alttext](https://github.com/UrbsLab/STREAMLINE/blob/main/Pictures/ML_pipe_schematic.png?raw=true)

***
### Implementation
STREAMLINE is coded in Python 3 relying heavily on pandas and scikit-learn as well as a variety of other python packages.

***
### Disclaimer
We make no claim that this is the best or only viable way to assemble an ML analysis pipeline for a given 
classification problem, nor that the included ML modeling algorithms will yield the best performance possible. 
We intend many expansions/improvements to this pipeline in the future to make it easier to use and hopefully more effective in application.  We welcome feedback, suggestions, and contributions for improvement.

#### General Guidelines for STREAMLINE Use
* SVM and ANN modeling should only be applied when data scaling is applied by the pipeline.
* Logistic Regression' baseline model feature importance estimation is determined by the exponential of the feature's coefficient. This should only be used if data scaling is applied by the pipeline.  Otherwise `use_uniform_FI` should be True.
* While the STREAMLINE includes `impute_data` as an option that can be turned off in `DataPreprocessing`, most algorithm implementations (all those standard in scikit-learn) cannot handle missing data values with the exception of eLCS, XCS, and ExSTraCS. 
* In general, STREAMLINE is expected to fail with an errors if run on data with missing values, while `impute_data` is set to 'False'.

### More Information
More information about the nature of STREAMLINE can be found in [INFO.md](./INFO.md).

***
## Installation and Use

*TODO*

```
pip install streamline
```

***

## Demonstration 
Included with this pipeline is a folder named `DemoData` including two small datasets used as a demonstration of pipeline efficacy. 

New users can easily run the included jupyter notebook 'as-is', and it will be run automatically on these datasets. 
The first dataset `hcc-data_example.csv` is the Hepatocellular Carcinoma (HCC) dataset taken from the UCI Machine Learning repository. 
It includes 165 instances, 49 fetaures, and a binary class label. 
It also includes a mix of categorical and numeric features, about 10% missing values, and class imbalance, i.e. 63 deceased (class = 1), and 102 surived (class 0).  

To illustrate how STREAMLINE can be applied to more than one 
dataset at once, we created a second dataset from this HCC dataset called `hcc-data_example_no_covariates.csv`, which is the same as the first but we have removed two covariates, i.e. `Age at Diagnosis`, and `Gender`.

***
# Acknowledgements
STREAMLINE is the result of 3 years of on-and-off development gaining feedback from multiple biomedical research collaborators at the University of Pennsylvania, 
Fox Chase Cancer Center, Cedars Sinai Medical Center, and the University of Kansas Medical Center. 
The bulk of the coding was completed by Ryan Urbanowicz, Robert Zhang and Harsh Bandhey. Special thanks to 
Yuhan Cui, Pranshu Suri, Patryk Orzechowski, Trang Le, Sy Hwang, Richard Zhang, Wilson Zhang, 
and Pedro Ribeiro for their code contributions and feedback.  

We also thank the following collaborators for their feedback on application 
of the pipeline during development: Shannon Lynch, Rachael Stolzenberg-Solomon, 
Ulysses Magalang, Allan Pack, Brendan Keenan, Danielle Mowery, Jason Moore, and Diego Mazzotti.