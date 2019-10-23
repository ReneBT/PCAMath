# PCAMath
Visual derivation of PCA using spectra

Code for generating simulated spectra, performing PCA then visually displaying spectral plots illustrating all the mathematical equations during derivation.


## Getting Started

The example code is designed to run in a Jupyter notebook. We recomend creating a virtual environment, then installing the requirements. For more information, see the Installation section below.



## Installation

Jupyter notebooks can be installed via Anaconda (https://www.anaconda.com/distribution/), which is freely available with installers for Windows, Mac, and Linux.

To install a virtual environment, follow the instructions at https://anbasile.github.io/programming/2017/06/25/jupyter-venv/.


## Windows: Using Jupyter Notebooks with a virtual environment (using Anaconda)

Open the Anaconda Prompt from the start menu (in the Anaconda folder). Start by updating Anaconda and ensuring that it should work properly under 64-bit windows. There are a couple of issues that can be avoided by upgrading the python/jupyter environment before starting (such as pywin32 version 225 causing errors).

	python -m pip install --upgrade pip
	pip install --upgrade pywin32==224
	pip install ipykernel

Run the following commands (you should see "(base)"  prefixing your command prompt:

	cd "project download location"
	python -m venv .venv_pca
	".venv_pca/Scripts/activate.bat"

You should now see "(venv_pca) (base)"  prefixing your command prompt:

	pip install -r requirements.txt
	ipython kernel install --user --name=.venv_pca

Now run Jupyter Notebook from the start menu (in the Anaconda folder). In the browser window, navigate to the location to which you downloaded and extracted the PCAMath project folder. Click on the file "PCApaper.ipynb". This will pop up the "PCApaper" Notebook.

In the Notebook, click Kernel -> Change Kernel -> .venv_pca

Now your notebook is running in a virtual environment, with the specific requirements needed by this script.
