# Citing STREAMLINE

If you use STREAMLINE in a scientific publication, please consider citing the following paper as well as noting the *release* applied within the manuscript (i.e. the Beta 0.2.4 release was applied in the publication below):

[Urbanowicz, Ryan, et al. "STREAMLINE: A Simple, Transparent, End-To-End Automated Machine Learning Pipeline Facilitating Data Analysis and Algorithm Comparison." Genetic Programming Theory and Practice XIX. Singapore: Springer Nature Singapore, 2023. 201-231.](https://link.springer.com/chapter/10.1007/978-981-19-8460-0_9)

BibTeX Citation:
```
@incollection{urbanowicz2023streamline,
  title={STREAMLINE: A Simple, Transparent, End-To-End Automated Machine Learning Pipeline Facilitating Data Analysis and Algorithm Comparison},
  author={Urbanowicz, Ryan and Zhang, Robert and Cui, Yuhan and Suri, Pranshu},
  booktitle={Genetic Programming Theory and Practice XIX},
  pages={201--231},
  year={2023},
  publisher={Springer}
}
```

If you wish to cite the STREAMLINE codebase instead, please use the following (indicating the release used in the link, for example, v0.2.5-beta):
```
@misc{streamline2022,
  author = {Urbanowicz, Ryan and Zhang, Robert},
  title = {STREAMLINE: A Simple, Transparent, End-To-End Automated Machine Learning Pipeline},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.2.5-beta} }
}
```
## STREAMLINE Applications
This section provides citations to publications applying STREAMLINE in recent research.

* [Exploring Automated Machine Learning for Cognitive Outcome Prediction from Multimodal Brain Imaging using STREAMLINE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10283099/)
```
@article{wang2023exploring,
  title={Exploring Automated Machine Learning for Cognitive Outcome Prediction from Multimodal Brain Imaging using STREAMLINE},
  author={Wang, Xinkai and Feng, Yanbo and Tong, Boning and Bao, Jingxuan and Ritchie, Marylyn D and Saykin, Andrew J and Moore, Jason H and Urbanowicz, Ryan and Shen, Li},
  journal={AMIA Summits on Translational Science Proceedings},
  volume={2023},
  pages={544},
  year={2023},
  publisher={American Medical Informatics Association}
}
```

* [Comparing Amyloid Imaging Normalization Strategies for Alzheimer’s Disease Classification using an Automated Machine Learning Pipeline](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10283108/)
```
@article{tong2023comparing,
  title={Comparing Amyloid Imaging Normalization Strategies for Alzheimer’s Disease Classification using an Automated Machine Learning Pipeline},
  author={Tong, Boning and Risacher, Shannon L and Bao, Jingxuan and Feng, Yanbo and Wang, Xinkai and Ritchie, Marylyn D and Moore, Jason H and Urbanowicz, Ryan and Saykin, Andrew J and Shen, Li},
  journal={AMIA Summits on Translational Science Proceedings},
  volume={2023},
  pages={525},
  year={2023},
  publisher={American Medical Informatics Association}
}
```

* [Toward Predicting 30-Day Readmission Among Oncology Patients: Identifying Timely and Actionable Risk Factors](https://ascopubs.org/doi/abs/10.1200/CCI.22.00097)
```
@article{hwang2023toward,
  title={Toward Predicting 30-Day Readmission Among Oncology Patients: Identifying Timely and Actionable Risk Factors},
  author={Hwang, Sy and Urbanowicz, Ryan and Lynch, Selah and Vernon, Tawnya and Bresz, Kellie and Giraldo, Carolina and Kennedy, Erin and Leabhart, Max and Bleacher, Troy and Ripchinski, Michael R and others},
  journal={JCO Clinical Cancer Informatics},
  volume={7},
  pages={e2200097},
  year={2023},
  publisher={Wolters Kluwer Health}
}
```

*[A Data-Driven Analysis of Ward Capacity Strain Metrics That Predict Clinical Outcomes Among Survivors of Acute Respiratory Failure](https://link.springer.com/article/10.1007/s10916-023-01978-5)

Kohn, R., Harhay, M.O., Weissman, G.E. et al. A Data-Driven Analysis of Ward Capacity Strain Metrics That Predict Clinical Outcomes Among Survivors of Acute Respiratory Failure. J Med Syst 47, 83 (2023).

* [Identifying Barriers to Post-Acute Care Referral and Characterizing Negative Patient Preferences Among Hospitalized Older Adults Using Natural Language Processing](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10148308/)
```
@inproceedings{kennedy2022identifying,
  title={Identifying Barriers to Post-Acute Care Referral and Characterizing Negative Patient Preferences Among Hospitalized Older Adults Using Natural Language Processing},
  author={Kennedy, Erin E and Davoudi, Anahita and Hwang, Sy and Freda, Philip J and Urbanowicz, Ryan and Bowles, Kathryn H and Mowery, Danielle L},
  booktitle={AMIA Annual Symposium Proceedings},
  volume={2022},
  pages={606},
  year={2022},
  organization={American Medical Informatics Association}
}
```

## Other STREAMLINE Related Research
In developing STREAMLINE we integrated a number of methods and lessons learned from our lab's previous research. We briefly summarize and provide citations for each.

### A rigorous ML pipeline for binary classification
A [preprint](https://arxiv.org/abs/2008.12829) describing an early version of what would become STREAMLINE applied to pancreatic cancer.

```
@article{urbanowicz2020rigorous,
  title={A Rigorous Machine Learning Analysis Pipeline for Biomedical Binary Classification: Application in Pancreatic Cancer Nested Case-control Studies with Implications for Bias Assessments},
  author={Urbanowicz, Ryan J and Suri, Pranshu and Cui, Yuhan and Moore, Jason H and Ruth, Karen and Stolzenberg-Solomon, Rachael and Lynch, Shannon M},
  journal={arXiv preprint arXiv:2008.12829v2},
  year={2020}
}
```

The STREAMLINE [preprint](https://arxiv.org/abs/2206.12002).
```
@article{urbanowicz2022streamline,
  title={STREAMLINE: A Simple, Transparent, End-To-End Automated Machine Learning Pipeline Facilitating Data Analysis and Algorithm Comparison},
  author={Urbanowicz, Ryan J and Zhang, Robert and Cui, Yuhan and Suri, Pranshu},
  journal={arXiv preprint arXiv:2206.12002v1},
  year={2022}
}
```

### Relief-based feature importance estimation
One of the two feature importance algorithms used by STREAMLINE is MultiSURF, a Relief-based filter feature importance algorithm that can prioritize features involved in either univariate or multivariate feature interactions associated with outcome. We believe that it is important to have at least one 'interaction-sensitive' feature importance algorithm involved in feature selection prior such that relevant features involved in complex associations are not filtered out prior to modeling. The [paper below](https://www.sciencedirect.com/science/article/pii/S1532046418301400) is an introduction and review of Relief-based algorithms.  
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
This [next published research paper](https://www.sciencedirect.com/science/article/pii/S1532046418301412) compared a number of Relief-based algorithms and demonstrated best overall performance with MultiSURF out of all evaluated. This second paper also introduced 'ReBATE', a scikit-learn package of Releif-based feature importance/selection algorithms (used by STREAMLINE).
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
Following feature importance estimation, STREAMLINE adopts an ensemble approach to determining which features to select. The utility of this kind of 'collective' feature selection, was introduced in the [next publication](https://link.springer.com/article/10.1186/s13040-018-0168-6).
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
STREAMLINE currently incorporates 15 ML classification modeling algorithms that can be run. Our own research has closely followed a subfield of evolutionary algorithms that discover a set of rules that collectively constitute a trained model. The appeal of such 'rule-based machine learning algorithms' (e.g. learning classifier systems) is that they can model complex associations while also offering human interpretable models. In the [first paper below](https://link.springer.com/article/10.1007/s12065-015-0128-8) we introduced 'ExSTraCS', a learning classifier system geared towards bioinformatics data analysis. ExSTraCS was the first ML algorithm demonstrated to be able to tackle the long-standing 135-bit multiplexer problem directly, largely due to it's ability to use prior feature importance estimates from a Relief algorithm to guide the evolutionary rule search.
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

In the [next published pre-print](https://arxiv.org/abs/2104.12844) we introduced a scikit-learn implementation of ExSTraCS (used by STREAMLINE) as well as a pipeline (LCS-DIVE) to take ExSTraCS output and characterize different patterns association between features and outcome. Future work will demonstrate how STREAMLINE can be linked with LCS-DIVE to better understand the relationship between features and outcome captured by rule-based modeling.
```
@article{zhang2021lcs,
  title={LCS-DIVE: An Automated Rule-based Machine Learning Visualization Pipeline for Characterizing Complex Associations in Classification},
  author={Zhang, Robert and Stolzenberg-Solomon, Rachael and Lynch, Shannon M and Urbanowicz, Ryan J},
  journal={arXiv preprint arXiv:2104.12844},
  year={2021}
}
```

In the [next publication](https://dl.acm.org/doi/abs/10.1145/3377929.3398097) we introduced the first scikit-learn compatible implementation of an LCS algorithm. Specifically this paper implemented eLCS, an educational learning classifier system. This eLCS algorithm is a direct descendant of the UCS algorithm.
```
@inproceedings{zhang2020scikit,
  title={A scikit-learn compatible learning classifier system},
  author={Zhang, Robert F and Urbanowicz, Ryan J},
  booktitle={Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion},
  pages={1816--1823},
  year={2020}
}
```

eLCS was originally developed as a very simple supervised learning LCS implementation primarily as an educational resource pairing with the following [published textbook](https://books.google.com/books?hl=en&lr=&id=C6QxDwAAQBAJ&oi=fnd&pg=PR5&dq=Introduction+to+learning+classifier+systems&ots=pTcnuuYQPE&sig=wNgZmWkcne9m3LQgDzuBu30uQ1Y#v=onepage&q=Introduction%20to%20learning%20classifier%20systems&f=false).
```
@book{urbanowicz2017introduction,
  title={Introduction to learning classifier systems},
  author={Urbanowicz, Ryan J and Browne, Will N},
  year={2017},
  publisher={Springer}
}
```
