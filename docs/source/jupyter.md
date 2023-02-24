# Running on Jupyter Notebook

## Running on Demo Dataset
Here we detail how to run STREAMLINE within the provided Jupyter Notebook named `STREAMLINE-Notebook.ipypnb`. 
This included notebook is set up to run on the included [demonstration datasets](sample.md#demonstration-data).

1. First, ensure all local installation is done as per the [guide](install.md#jupyter-notebook) in your environment 
   and dataset assumptions are satisfied.

2. Open Jupyter Notebook (info about Jupyter Notebooks here, https://jupyter.readthedocs.io/en/latest/running.html) by 
   going to the STREAMLINE Root folder, typing `jupyter notebook` and then opening the `STREAMLINE-Notebook.ipypnb` that
   shows up in the new page open on your web browser.

3. Scroll down to the second code block of the notebook below the header 'Mandatory Parameters to Update' and update the following run parameters to reflect paths on your PC.
    * `data_path`: Change the path so it reflects the location of the `DemoData` folder (within the STREAMLINE folder) on your PC, e.g. `C:/Users/ryanu/Documents/GitHub/STREAMLINE/DemoData`.
    * `output_path`: Change the path to specify the desired location for the STREAMLINE output experiment folder.

4. Click `Kernel` on the Jupyter notebook GUI, and select `Restart & Run All` to run the script.  

5. Running the included [demonstration datasets](sample.md#demonstration-data) with the pre-specified notebook run parameters, 
   should only take a 3-5 minutes depending on your PC hardware.
    * Note: It may take several hours or more to run this notebook in other contexts. Parameters that impact runtimes are discussed in [this section](tips.md#reducing-runtime) above. We recommend users with larger analyses to use a computing cluster if possible.

## Running on Your Own Datasets
Follow the same steps as in the [Colab Notebook](colab.md#running-on-your-own-datasets-tbd)
