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

See [below](#citations) for how to cite this preprint and/or the codebase prior to the availability of the final 
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


***
## Installation and Use

*In near future*

```
pip install streamline
```

***

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

***
# Citations
To cite the STREAMLINE preprint on arXiv, please use:
```
@article{urbanowicz2022streamline,
  title={STREAMLINE: A Simple, Transparent, End-To-End Automated Machine Learning Pipeline Facilitating Data Analysis and Algorithm Comparison},
  author={Urbanowicz, Ryan J and Zhang, Robert and Cui, Yuhan and Suri, Pranshu},
  journal={arXiv preprint arXiv:2206.12002v1},
  year={2022}
}
```
If you wish to cite the STREAMLINE codebase, please use:
```
@misc{streamline2022,
  author = {Urbanowicz, Ryan and Zhang, Robert},
  title = {STREAMLINE: A Simple, Transparent, End-To-End Automated Machine Learning Pipeline},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/UrbsLab/STREAMLINE/} }
}
```
## URBS-lab related research
In developing STREAMLINE we integrated a number of methods and lessons learned from our lab's previous research. We briefly summarize and provide citations for each.

### A rigorous ML pipeline for binary classification
A preprint describing an early version of what would become STREAMLINE applied to pancreatic cancer.
```
@article{urbanowicz2020rigorous,
  title={A Rigorous Machine Learning Analysis Pipeline for Biomedical Binary Classification: Application in Pancreatic Cancer Nested Case-control Studies with Implications for Bias Assessments},
  author={Urbanowicz, Ryan J and Suri, Pranshu and Cui, Yuhan and Moore, Jason H and Ruth, Karen and Stolzenberg-Solomon, Rachael and Lynch, Shannon M},
  journal={arXiv preprint arXiv:2008.12829v2},
  year={2020}
}
```

### Relief-based feature importance estimation
One of the two feature importance algorithms used by STREAMLINE is MultiSURF, a Relief-based filter feature importance algorithm that can prioritize features involved in either univariate or multivariate feature interactions associated with outcome. We believe that it is important to have at least one 'interaction-sensitive' feature importance algorithm involved in feature selection prior such that relevant features involved in complex associations are not filtered out prior to modeling. The paper below is an introduction and review of Relief-based algorithms.  
```
@article{urbanowicz2018relief,
  title={Relief-based feature selection: Introduction and review},
  author={Urbanowicz, Ryan J and Meeker, Melissa and La Cava, William and Olson, Randal S and Moore, Jason H},
  journal={Journal of biomedical informatics},
  volume={85},
  pages={189--203},
  year={2018},
  publisher={Elsevier}
}
```
This next published research paper compared a number of Relief-based algorithms and demonstrated best overall performance with MultiSURF out of all evaluated. This second paper also introduced 'ReBATE', a scikit-learn package of Releif-based feature importance/selection algorithms (used by STREAMLINE).
```
@article{urbanowicz2018benchmarking,
  title={Benchmarking relief-based feature selection methods for bioinformatics data mining},
  author={Urbanowicz, Ryan J and Olson, Randal S and Schmitt, Peter and Meeker, Melissa and Moore, Jason H},
  journal={Journal of biomedical informatics},
  volume={85},
  pages={168--188},
  year={2018},
  publisher={Elsevier}
}
```

### Collective feature selection
Following feature importance estimation, STREAMLINE adopts an ensemble approach to determining which features to select. The utility of this kind of 'collective' feature selection, was introduced in the next publication.
```
@article{verma2018collective,
  title={Collective feature selection to identify crucial epistatic variants},
  author={Verma, Shefali S and Lucas, Anastasia and Zhang, Xinyuan and Veturi, Yogasudha and Dudek, Scott and Li, Binglan and Li, Ruowang and Urbanowicz, Ryan and Moore, Jason H and Kim, Dokyoon and others},
  journal={BioData mining},
  volume={11},
  number={1},
  pages={1--22},
  year={2018},
  publisher={Springer}
}
```

### Learning classifier systems
STREAMLINE currently incorporates 15 ML classification modeling algorithms that can be run. Our own research has closely followed a subfield of evolutionary algorithms that discover a set of rules that collectively constitute a trained model. The appeal of such 'rule-based machine learning algorithms' (e.g. learning classifier systems) is that they can model complex associations while also offering human interpretable models. In the first paper below we introduced 'ExSTraCS', a learning classifier system geared towards bioinformatics data analysis. ExSTraCS was the first ML algorithm demonstrated to be able to tackle the long-standing 135-bit multiplexer problem directly, largely due to it's ability to use prior feature importance estimates from a Relief algorithm to guide the evolutionary rule search.
```
@article{urbanowicz2015exstracs,
  title={ExSTraCS 2.0: description and evaluation of a scalable learning classifier system},
  author={Urbanowicz, Ryan J and Moore, Jason H},
  journal={Evolutionary intelligence},
  volume={8},
  number={2},
  pages={89--116},
  year={2015},
  publisher={Springer}
}
```
In the next published pre-print we introduced a scikit-learn implementation of ExSTraCS (used by STREAMLINE) as well as a pipeline (LCS-DIVE) to take ExSTraCS output and characterize different patterns association between features and outcome. Future work will demonstrate how STREAMLINE can be linked with LCS-DIVE to better understand the relationship between features and outcome captured by rule-based modeling.
```
@article{zhang2021lcs,
  title={LCS-DIVE: An Automated Rule-based Machine Learning Visualization Pipeline for Characterizing Complex Associations in Classification},
  author={Zhang, Robert and Stolzenberg-Solomon, Rachael and Lynch, Shannon M and Urbanowicz, Ryan J},
  journal={arXiv preprint arXiv:2104.12844},
  year={2021}
}
```
In the next publication we introduced the first scikit-learn compatible implementation of an LCS algorithm. Specifically this paper implemented eLCS, an educational learning classifier system. This eLCS algorithm is a direct descendant of the UCS algorithm.
```
@inproceedings{zhang2020scikit,
  title={A scikit-learn compatible learning classifier system},
  author={Zhang, Robert F and Urbanowicz, Ryan J},
  booktitle={Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion},
  pages={1816--1823},
  year={2020}
}
```
eLCS was originally developed as a very simple supervised learning LCS implementation primarily as an educational resource pairing with the following published textbook.
```
@book{urbanowicz2017introduction,
  title={Introduction to learning classifier systems},
  author={Urbanowicz, Ryan J and Browne, Will N},
  year={2017},
  publisher={Springer}
}
```