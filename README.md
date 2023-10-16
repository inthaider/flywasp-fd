# Fly-Wasp Backing Prediction Project

This project aims to predict fly backing events based on tabular data generated from recorded footage of fly-wasp interactions. This research is conducted in collaboration with Faizan Dogar (FD), who is affiliated with a Postdoc Researcher at Columbia University.

## Table of Contents

1. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
2. [Data Description](#data-description)
    - [DataFrame Structure](#dataframe-structure)
3. [Usage Instructions](#usage-instructions)
    - [Loading Data](#loading-data)
    - [Data Preprocessing](#data-preprocessing)
4. [Previous Models and Approaches](#previous-models-and-approaches)
    - [Models Used by FD](#models-used-by-fd)
    - [Assumptions](#assumptions)
    - [Data Split](#data-split)
5. [Potential Future Directions](#potential-future-directions)
    - [Using RNNs](#using-rnns)
6. [Contributors](#contributors)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

---

## Getting Started

### Prerequisites

- Python 3.x
- Pandas
- Scikit-learn
- (Optional) Deep Learning Libraries for RNN

### Installation

1. Clone the repository.
2. Install the required packages via pip or conda.

---

## Data Description

The data has been processed using DeepLabCut software and is based on interactions between fly-wasp pairs.

**Specifics:**

- **Frame Rate**: 50Hz (50 frames per second)
- **Frame Duration**: 0.025 seconds
- **Total Frames for Each Pair**: 72,000 (equivalent to 30 minutes)
- **Unique Fly-Wasp Pairs**: 136

### DataFrame Structure

The data is stored in a DataFrame with several columns:

- `start_walk`: Binary variable indicating whether the fly started walking backward (Target variable)
- `ANTdis_1`: TBD
- `ANTdis_2`: TBD
- ... (More columns to be explored)

---

## Usage Instructions

### Loading Data

The dataset will be shared as a pickle object. To read this into a pandas DataFrame, you can use the following generic code snippet:

```python
import pandas as pd

pickle_path = "path/to/your/pickle/file.pkl"
df = pd.read_pickle(pickle_path)
```

### Data Preprocessing

After loading the DataFrame, you will need to perform some preprocessing steps:

1. Remove unnecessary columns.
2. Rearrange columns for ease of use.
3. Calculate aggregate features.

For example:

```python
# Drop irrelevant columns
df = df.drop('plot', axis=1)

# Rearrange columns
cols = df.columns.tolist()
cols.insert(cols.index('F2Wdis') + 1, cols.pop(cols.index('F2Wdis_rate')))
df = df[cols]

# Compute the mean of certain features
df['ANTdis'] = df[['ANTdis_1', 'ANTdis_2']].mean(axis=1)

# Add the label for walking backward
df['start_walk'] = ((df['walk_backwards'] == 1) & (df['walk_backwards'].shift(1) == 0)).astype(int)
```

---

## Previous Models and Approaches

### Models Used by FD

Faizan Dogar initially used two modeling frameworks:

1. Random Forest
2. Logistic Regression

### Assumptions

- The data is independent and identically distributed (i.i.d.).

### Data Split

- Training: 1/3 of the data
- Validation: 1/3 of the data
- Testing: 1/3 of the data

### Feature Engineering:

- Lagged variables for each feature (20 lags)
  
### Target Variable:

- `start_walk`

---

## Potential Future Directions

### Using RNNs

For Recurrent Neural Networks (RNNs):

- FD suggested/is considering not using lagged variables.
- Preprocessing steps for RNNs such as scaling and normalization are yet to be decided.
- The specific architecture and setup for the RNN model are to be defined.
- ... (More ideas can be included here)

---

## Contributors

- Faizan Dogar
- Jibran Haider

---

## License

TBD

---

## Acknowledgments

Special thanks to the Postdoc Researcher at Columbia University for their invaluable guidance and resources.

---