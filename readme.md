# Repository for the draft `A FAST MULTIPLICATIVE UPDATES ALGORITHM FOR NON-NEGATIVE MATRIX FACTORIZATION'

This is a temporary repository to share the codes for the submitted draft on fastMU. The code will be improved upon publication.
## Organisation

- The following scripts contain the algorithms
  - NLS_Frobenius
  - NLS_KL
  - NMF_Frobenius
  - NMF_KL
- The experiments are in the scripts
  - "synthetic_comparisons_Frobenius_nls.py",
  - "synthetic_comparisons_Frobenius.py",
  - "synthetic_comparisons_KL_nls_sparse.py",
  - "synthetic_comparisons_KL_sparse.py",
  - "synthetic_Frobenius_delta_choice.py",
  - "synthetic_KL_delta_choice.py",
  - "test_audio_nls.py",
  - "test_audio_nmf.py",
  - "test_hyperspectral_nls.py",
  - "test_hyperspectral.py"
- Data is stored in data_and_scripts/
- All results including figures are stored in Results/

## Installation

The following libraries are required to run the algorithms:
- numpy
- scipy
- matplotlib
Additionally, to run the experiments, install the following packages:
- plotly
- pandas
- soundfile (audio experiment only)
Two custom packages are also required:
- nnfac (local version): install by 
  ```
  pip install .
  ```
    in the nnfac_perso directory
- shootout: install from [this github repository](https://github.com/cohenjer/shootout),
    should work with commit 511974955b4e458608ef6ebd3c0a5db2e12cdcf5

## Running experiments

Simply run the `runscript.py` script. It take as input parameter the number of random samples for the comparisons:
``` 
python runscript.py 0
``` 
uses local stored result to produce figures
``` 
python runscript.py xx
``` 
with xx some natural number runs all the experiments with xx samples, stores the results and produces figures.