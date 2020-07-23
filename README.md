# DMI

Class that implements the DMI imputation algorithm for imputing missing values in a dataset. DMI splits the dataset into horizontal segments using a C4.5 (J48) decision tree in order to increase the correlation between attributes for EMI. EMI is performed to impute missing numerical attribute values and mean/mode (within a leaf) imputation is used to perform missing categorical attribute values. Uses Amri Napolitano's EMI implementation for Weka.

DMI specification from:

*Rahman, M. G., and Islam, M. Z. (2013): Missing Value Imputation Using Decision Trees and Decision Forests by Splitting and Merging Records: Two Novel Techniques, Knowledge-Based Systems, Vol. 53, pp. 51 - 65, ISSN 0950-7051*, DOI information: 10.1016/j.knosys.2013.08.023, Available at [http://www.sciencedirect.com/science/article/pii/S0950705113002591](http://www.sciencedirect.com/science/article/pii/S0950705113002591)

For more information, please see Associate Professor Zahid Islam's website [here](http://csusap.csu.edu.au/~zislam/)

## BibTeX
```
@article{rahman2013imputation,
  title={Missing Value Imputation Using Decision Trees and Decision Forests by Splitting and Merging Records: Two Novel Techniques},
  author={Rahman, Md. Geaur and Islam, Md Zahidul},
  journal={Knowledge-Based Systems},
  volume={53},
  pages={51--65},
  year={2013},
  publisher={Elsevier}
}
```

## Installation
Either download DMI from the Weka package manager, or download the latest release from the "**Releases**" section on the sidebar of Github. A video on the installation and use of the package can be found [here](https://www.youtube.com/watch?v=mS_2im6XCD8&t=1s).

## Compilation / Development
Set up a project in your IDE of choice, including weka.jar and EMImputation.jar as compile-time libraries. EMImputation.jar is available in the Weka package manager.

## Changes:
- Leaves that are too small to run EMI are replaced by the nearest node above them in a tree. We call this process "merging".
- Records that cannot be assigned to any leaves in a tree for imputing an attribute will be assigned to the leaf with the closest centroid.
- If there are too few records in the whole dataset to ever run EMI (i.e. number of records < (number of numeric attributes in dataset + 2)) no merging takes place.

## Valid options are:
`-D`
minCategoriesForDiscretization - Minimum number of categories for discretization

`-N`
j48MinRecordsInLeaf - Minimum number of records in a J48 tree leaf. A negative value will default to (number of numeric attributes in dataset + 2).

`-F`
j48ConfidenceFactor - Confidence factor for J48

`-E`
minRecordsForEMI - Minimum records in a leaf for EMI to be able to run. A negative value will default to (number of numeric attributes in dataset + 2)

`-I`
emiNumIterations - Iterations for EMI. A negative value will be set to Integer.MAX_VALUE

`-L`
emiLogLikelihoodThreshold - Log likelihood threshold for terminating EMI.
