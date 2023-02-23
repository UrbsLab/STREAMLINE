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