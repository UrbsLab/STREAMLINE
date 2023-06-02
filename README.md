![alttext](https://github.com/UrbsLab/STREAMLINE/blob/main/info/Pictures/STREAMLINE_Logo_Full.png?raw=true)
# Overview

NOTE: We are currently updating the main branch of STREAMLINE with a major update. Please download and use the most recent release rather than the main branch until these updates have been completed and a new release has been created.


STREAMLINE is an end-to-end automated machine learning (AutoML) pipeline
that empowers anyone to easily train, interpret, and apply predictive models as
part of a rigorous and customizable data mining analysis. It is programmed in
Python 3 using many common libraries including [Pandas](https://pandas.pydata.org/)
and [scikit-learn](https://scikit-learn.org/stable/).

The schematic below describes and breaks the STREAMLINE analysis pipeline 
down into it's 9 phases.

![alttext](https://github.com/UrbsLab/STREAMLINE/blob/main/info/Pictures/ML_pipe_schematic.png?raw=true)

Detailed documentation of STREAMLINE is available [here](https://urbslab.github.io/STREAMLINE/index.html).

## Pipeline Design
The goal of STREAMLINE is to provide an easy and transparent framework
to learn predictive associations from tabular data. The design of this pipeline
is meant to not only pick a best performing algorithm/model for a given dataset,
but to leverage the different algorithm perspectives (i.e. biases, strengths,
and weaknesses) to gain a broader understanding of the associations in that data.

The overall development of this pipeline focused on:
   1. Automation and ease of use
   2. Avoiding and detecting common sources of bias
   3. Optimizing modeling performance
   4. Ensuring reproducibility (see STREAMLINE parameter settings)
   5. Capturing complex associations in data (e.g. feature interactions)
   6. Enhancing interpretability of output
   7. Run mode flexibility
   8. More advanced users can easily add their own scikit-learn compatible modeling algorithms to STREAMLINE

We recommend reviewing [./info/about.md](info/about.md) to gain a deeper understanding
of STREAMLINE with respect to it's overall design, what it includes, what it
can be used for, and implementation highlights that differentiate it from other
AutoML tools.

## Current Limitations
At present, STREAMLINE is limited to supervised learning on tabular,
binary classification data. We are currently expanding STREAMLINE to multi-class
and regression outcome data as well.

As STREAMLINE is currently in its 'beta' release, we recommend users first check that they have downloaded the
most recent release of STREAMLINE before use. We are actively updating this software as feedback is received.

## STREAMLINE Publication
The first publication detailing STREAMLINE (release Beta 0.2.4) and applying it to
simulated benchmark data can be found [here](https://link.springer.com/chapter/10.1007/978-981-19-8460-0_9).

This paper is also available as a preprint on arxiv, [here](https://arxiv.org/abs/2206.12002?fbclid=IwAR1toW5AtDJQcna0_9Sj73T9kJvuB-x-swnQETBGQ8lSwBB0z2N1TByEwlw).

See [./info/citations.md](info/citations.md) for how to cite STREAMLINE.

***
# Installation and Use
STREAMLINE can be run using a variety of modes balancing ease of use and efficiency.
* Serially on Google Cloud in a Google Colab Notebook (best for beginners)
* Serially/locally in a Jupyter Notebook
* Serially/locally from the command line
* In parallel on an HPC (best for efficiency)

See [documentation](https://urbslab.github.io/STREAMLINE/index.html) for installation, requirements, and use details for each.

Basic installation instructions for use on Google Colab, and local runs are given below.

## Google Colab
There is no local installation or additional steps required to run
STREAMLINE on Google Colab.

Just have a Google Account and open this Colab link to run the demo now (takes ~ 6-7 min):
[https://colab.research.google.com/drive/18uU1KEs7SgFpJyFmot7LBEc85B6vbGU9?usp=sharing](https://colab.research.google.com/drive/18uU1KEs7SgFpJyFmot7LBEc85B6vbGU9?usp=sharing)


## Local
Install STREAMLINE for local use with the following command line commands:

```
git clone --single-branch https://github.com/UrbsLab/STREAMLINE
cd STREAMLINE
pip install -r requirements.txt
```

Now your STREAMLINE package is ready to use from the `STREAMLINE` folder either
from the included [Jupyter Notebook](https://github.com/UrbsLab/STREAMLINE/blob/dev/STREAMLINE-Notebook.ipynb) file or the command line.

***
# Other Information
## Demonstration Data
Included with this pipeline is a folder named `DemoData` including two small datasets used as a demonstration of
pipeline efficacy. New users can easily test/run STREAMLINE in all run modes set up to run automatically on these datasets.

Detailed info about advanced features and command line implementation can be found in  [./info/demo.md](info/demo.md)

***
## Disclaimer
We make no claim that this is the best or only viable way to assemble an ML analysis pipeline for a given
classification problem, nor that the included ML modeling algorithms will yield the best performance possible.
We intend many expansions/improvements to this pipeline in the future to make it easier to use and hopefully more effective in application.  We welcome feedback, suggestions, and contributions for improvement.

***
# Acknowledgements
STREAMLINE is the result of 3 years of on-and-off development gaining feedback from multiple biomedical research collaborators at the University of Pennsylvania, Fox Chase Cancer Center, Cedars Sinai Medical Center, and the University of Kansas Medical Center.
The bulk of the coding was completed by Ryan Urbanowicz, Robert Zhang and Harsh Bandhey. Special thanks to
Yuhan Cui, Pranshu Suri, Patryk Orzechowski, Trang Le, Sy Hwang, Richard Zhang, Wilson Zhang,
and Pedro Ribeiro for their code contributions and feedback.  

We also thank the following collaborators for their feedback on application
of the pipeline during development: Shannon Lynch, Rachael Stolzenberg-Solomon,
Ulysses Magalang, Allan Pack, Brendan Keenan, Danielle Mowery, Jason Moore, and Diego Mazzotti.
