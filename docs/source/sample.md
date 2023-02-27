# Sample Outputs

## Demonstration Data

Included with this pipeline is a folder named DemoData including two small datasets used as a 
demonstration of pipeline efficacy. New users can easily run the included Colab or jupyter notebook 
'as-is', and it will be run automatically on these datasets. The first dataset 
hcc-data_example.csv is the Hepatocellular Carcinoma (HCC) dataset taken from the UCI 
Machine Learning repository. It includes 165 instances, 49 fetaures, and a binary class label. 
It also includes a mix of categorical and numeric features, about 10% missing values, and class imbalance, 
i.e. 63 deceased (class = 1), and 102 surived (class 0). 

To illustrate how STREAMLINE can be applied to more than one dataset at once, we created a second dataset 
from this HCC dataset called hcc-data_example_no_covariates.csv, which is the same as the first but we have 
removed two covariates, i.e. Age at Diagnosis, and Gender.

Furthermore, to demonstrate how STREAMLINE-trained models may be applied to 
new data in the future through the Phase 9 (Replication Phase) we have simply 
added a copy of hcc-data_example.csv, renamed as hcc-data_example_rep.csv to the folder DemoRepData. 
While this is not a true replication dataset (as none was available for this example) it does illustrate 
the functionality of ApplyModel. Since the cross validation (CV)-trained models are being applied to all of the 
original target data, the ApplyModel.py results in this demonstration are predictably overfit. 

When applying trained models to a true replication dataset model prediction performance is generally 
expected to be as good or less well performing than the individual testing evaluations completed for each CV model.


## Run Pipeline and Output Folders
* To quickly pre-view the pipeline (pre-run on included [demonstration datasets](#demonstration-data) without any installation whatsoever, open the following link:

[https://github.com/UrbsLab/STREAMLINE/blob/dev/STREAMLINE-Notebook.ipynb](https://github.com/UrbsLab/STREAMLINE/blob/dev/STREAMLINE-Notebook.ipynb)

Note, that with this link, you can only view the pre-run STREAMLINE Jupyter Notebook and will not be able to run or permanently edit the code. This is an easy way to get a feel for what the pipeline is and does.

* To quickly pre-view the folder of output files generated when running STREAMLINE on the [demonstration datasets](#demonstration-data), open the following link:

[https://drive.google.com/drive/folders/15qIaE4ZxRuoYlm-Y8HU-bkwwMs7JHxmL?usp=sharing](https://drive.google.com/drive/folders/15qIaE4ZxRuoYlm-Y8HU-bkwwMs7JHxmL?usp=sharing
)


## Figures Summary
![alttext](pictures/STREAMLINE_Figures.png)
