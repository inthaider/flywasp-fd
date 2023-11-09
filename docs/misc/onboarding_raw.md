# Fly-Wasp Backing Prediction Project -- Faizan Dogar (FD) working for/with a PostDoc Researcher at Columbia University

## WeTransfer download link for the data (ff-mw.pkl):
*https://wetransfer.com/downloads/72619a02a0250f28c7d507d98a513b3520231014201152/898b33b5b0f0be38cb6328f4271c5ac920231014201209/c1f7a2*

## Onboarding email from FD to Jibran (Jb) with instructions
Hello Jibranay, 

I have shared with you a link to download the data (stored as 'ff-mw.pkl'). The data is for the wild type female fly - male wasp interaction. You will notice that it is a pickle object - just look up pandas reading in a pickle object as a dataframe. See attached the file with variable descriptions. 

Once you download the pickle object, run the following piece of code: 

## Load in the dataframe
pickle_path = r"D:\Fly-Wasp\Data\Pickle Objects\Mutants\\"

# And you can read it back into memory like this:
df = pd.read_pickle(pickle_path + 'Kt_f_X_LH_f_Z_18.pkl')
df = df.drop('plot', axis = 1)

cols = df.columns.tolist()

# Rearrange the column names
cols.insert(cols.index('F2Wdis') + 1, cols.pop(cols.index('F2Wdis_rate')))

# Reindex the DataFrame with the new column order
df = df[cols]

# Calculating the mean of the 'ANTdis_1', 'ANTdis_2 vars
df['ANTdis'] = df[['ANTdis_1', 'ANTdis_2']].mean(axis=1)

# Adding the label
# create new variable 'start_walk'
df['start_walk'] = ((df['walk_backwards'] == 1) & (df['walk_backwards'].shift(1) == 0)).astype(int)

Note that start_walk is the variable we are trying to predict. Before you start building any models, I suggest just get familiar with the data - the best way to do this is to visualize the distributions of different variables grouped by start_walk. 

Let us find a time to speak about this more once you have downloaded the dataset and are able to load it as a pandas dataframe. 

Best,
Faizan

## Notes from meeting with FD on 2023-10-16 11am-12pm.
### Data description:
- The postdoc researcher FD is working with used DeepLabCut software to track/interpret/analyze the raw frame-by-frame images, which can generate tabular data from the visual data.
- Frame rate is 50Hz (50 frames per second)
- Each frame represents 0.025 seconds
- For each fly-wasp pair, there is data for 72,000 frames (i.e. 30 minutes in total)
- We have 136 unique fly-wasp pairs
- Explanation of dataframe columns/features
    - start_walk: binary variable indicating whether the fly started walking backwards -- this is the variable that we are trying to predict
    - ANTdis_1: ???TBD!!!
    - ANTdis_2: ???TBD!!!

## Previous training paradigm(s) used by Faizan:
- Used two modelling frameworks:
    1. Random Forest
    2. Logistic Regression
- FD started with a preprocessed dataframe of data from DeepLabCut and generated lagged variables for his models (20 lags for each variable):
    - i.e. each observation included stimulus history for 20 frames prior to the frame in question
    - e.g. f2wdis_lag_1, f2wdis_lag_2, ..., f2wdis_lag_20 (similarly, there are other variables each with 20 lag values) -- these are just 20 of the variables for each observation (row) representing feature values of the corresponding lagged frame
    - Note that i in x_lag_i is counting backwards from the current frame
- An important assumption for both models was:
    - The data is independent and identically distributed (i.i.d.)
        - i.e. each set of values for each variable is independent of the other variables and identically distributed
- The task is to predict if the current frame is a backing onset event or not given the parameter history of the previous 20 frames
    - i.e. the target variable is start_walk
- Test-train split used by FD (136 fly-wasp pairs):
    - 1/3rds of the data was used for training
    - 1/3rds of the data was used for validation
    - 1/3rds of the data was used for testing

## Potential application of a recurrent neural network (RNN) model to this problem
- For RNN's, FD thinks we maybe shouldn't/won't use lagged variables
- Potential setup/initialization ideas:
    1. Perhaps use frame f - 1 and f - 2 as inputs to predict the start_walk label at frame f?
    2. ???TBD!!!
- Using traditional ML models viz. random forest and logistic regression, we only needed 1 observation (row) to predict the start_walk label; what do we do/need now for RNN?
- Need to look into preprocessing steps for RNNs
    - e.g. scaling, normalization, etc.
- Basically, need to figure out how to set up both the data and the RNN model (architecture etc.) for this problem




# variables in the ff-mw.pkl dataframe not in the original data before preprocessing:
Frame:
F2Wdis_rate:
file:
ANTdis:
walk_backwards:
start_walk: