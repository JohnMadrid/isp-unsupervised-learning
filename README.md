# isp-unsupervised-learning
Code on supervised vs. unsupervised learning, created for Neuromatch's Impact Scholar Program 2025.

## Environment Setup

Use the provided Conda environment file to create a reproducible environment.

```bash
# go to the project's root directory
cd /Users/someting/isp-unsupervised-learning
# create the environment
conda env create -f environment.yml

# activate it
conda activate isp-unsupervised-learning

# register the kernel for Jupyter
python -m ipykernel install --user \
  --name isp-unsupervised-learning \
  --display-name "Python (isp-unsupervised-learning)"
```

Then open `80k_neurons_pca.ipynb` and select the "Python (isp-unsupervised-learning)" kernel.

### Updating/Refining the environment
If you make changes or want to capture the exact packages from your current environment (e.g., `nma-compneuro`), export and commit an updated file:

```bash
# from your current env (replace name if needed)
conda activate nma-compneuro

# export only explicitly installed packages (cleaner)
conda env export --from-history --no-builds > environment.yml

# OR export full spec including transitive deps (more exact)
conda env export --no-builds > environment.yml
```

Apply updates locally with:

```bash
conda env update -f environment.yml --prune
```

Optional: also generate a `requirements.txt` if you prefer pip:

```bash
pip freeze > requirements.txt
```
