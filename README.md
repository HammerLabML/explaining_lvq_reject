# Explaining Reject Options of Learning Vector Quantization Classifiers

This repository contains the implementation of the methods proposed in the paper [Explaining Reject Options of Learning Vector Quantization Classifiers](paper.pdf) by André Artelt, Johannes Brinkrolf, Roel Visser and Barbara Hammer.

The experiments as described in the paper are implemented in the folder [Implementation](Implementation/).

## Abstract
While machine learning models are usually assumed to always
output a prediction, there also exist extensions in the form of reject options which allow the model to reject inputs where only a prediction with an unacceptably low certainty would be possible. With the ongoing rise of eXplainable AI, a lot of methods for explaining model predictions have been developed. However, understanding why a given input was rejected, instead of being classified by the model, is also of interest. Surprisingly, explanations of rejects have not been considered so far. We propose to use counterfactual explanations for explaining rejects and investigate how to efficiently compute counterfactual explanations of different reject options for an important class of models, namely  prototype-based classifiers such as learning vector quantization models.


## Details
### Implementaton of experiments
The shell script `run_experiments.sh` runs all experiments -- note that the two folders `Results/` and `Results2/` must be created before running this script.

#### Algorithmic properties
The script `experiments1.py` runs experiments for evaluating algorithmic properties. The script expects three arguments:
1. Name of the used data set
2. Name of the LVQ model
3. Name of the reject option

Note that a list of all valid and supported names can be found in `utils.py`.

#### Goodness of explanations
The script `experiments2.py` runs experiments for evaluating the goodness of the computed counterfactual explanations. The script expects the same agruments as `experiments1.py` does -- see above.

### Other (important) stuff
#### `counterfactual.py`
Implementation of our proposed algorithms for computing counterfactual explanations of different reject options for LVQ models.

#### `lvq.py`
Implementation of a wrapper for LVQ models.

#### `reject_option.py`
Implementation of the reject options discussed in the paper.

## Data

Note that we did not publish all data sets due to unclear copyrights. Please contact us if you are interested in the medical data sets.

## Requirements

- Python3.6
- Packages as listed in `Implementation/REQUIREMENTS.txt`

## License

MIT license - See [LICENSE.md](LICENSE.md)

## How to cite

You can cite the version on [arXiv](https://arxiv.org/abs/2202.07244)
