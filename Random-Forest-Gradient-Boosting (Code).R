############################################################################
# Assignment – Decision Trees, Random Forest & Gradient Boosting
# R Data Set: HMEQ_Scrubbed.csv
# Author Name: Avishek Pradhan
# Date: Feb 5th, 2026
############################################################################

# ****** REFERENCES ********
# This script is DONE WITH THE HELP from the class lecture materials:
# - Random Forest & Gradient Boosting (FLAG):  RF_GB_FLAG_Insurance.txt & video (ROC & AUC on 3 models)
# - Random Forest & Gradient Boosting (AMOUNT):RF_GB_AMT_Insurance.txt & video (RMSE comparison on 3 models)
# - Probability / Severity model: Probability of 0/1 event * Loss-Given-Event (lecture video)
############################################################################

# ****** OVERVIEW ******
# Continue code from Week 4 by adding Random Forest and Gradient Boosting.

# HMEQ_Scrubbed.csv dataset 

# Basic data review and verification

# CLASSIFICATION models for TARGET_BAD_FLAG
#   - Decision Tree (rpart)
#   - Random Forest (randomForest)
#   - Gradient Boosting (gbm)
#   - List the important variables for the Random Forest & Gradient Boosting model 
#   - ROC & AUC comparison for multiple random train/test splits

# REGRESSION models for TARGET_LOSS_AMT
#   - Decision Tree (rpart)
#   - Random Forest (randomForest)
#   - Gradient Boosting (gbm)
#   - List the important variables for the Random Forest & Gradient Boosting model 
#   - calculate the Root Mean Square Error (RMSE) for all models.
#   - RMSE comparison for multiple random train/test splits
#
# PROBABILITY x SEVERITY model (P(default) * Loss given default)
#   - Use best classification model from Step 2 to predict P(default)
#   - Build Decision Tree / Random Forest / Gradient Boosting on TARGET_LOSS_AMT 
#     using only defaulted records (TARGET_BAD_FLAG = 1)
#   - List the important variables for both models.
#   - Choose one severity model and compute: Expected Loss = P(default) * Loss-Given-Default
#   - Compare RMSE of this Probability/Severity model vs Step 3.

############################################################################

# ****** CLEAN & SETUP : Clean environment and define RMSE ******
# Clear old variables, plots, and console for a fresh start
rm(list = ls())
graphics.off()
cat("\014")   # Clear console

# RMSE function: lower RMSE = better prediction
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

############################################################################

# ****** LOAD LIBRARIES ******
# install.packages("rpart")          # For decision trees
# install.packages("rpart.plot")     # For pretty decision tree plots
# install.packages("ROCR")           # For ROC curves and AUC calculation
# install.packages("randomForest")   # For Random Forest models
# install.packages("gbm")            # For Gradient Boosting models

library(rpart)                        # Load decision tree library
library(rpart.plot)                   # Load decision tree plotting library
library(ROCR)                         # Load ROC / AUC evaluation functions
library(randomForest)                 # Load Random Forest implementation
library(gbm)                          # Load Gradient Boosting Machines implementation

############################################################################

# ****** STEP 1: BASIC DATA REVIEW AND VERIFICATION  ******
# Read the data into R for HMEQ_Scrubbed.csv 
HmeqScrubbed <- read.csv("HMEQ_Scrubbed.csv",   # File name
                         stringsAsFactors = FALSE) # Read CSV into R & keep text as character

# List the structure of the data (str)
str(HmeqScrubbed) 

# Execute a summary of the data

summary(HmeqScrubbed) 

# Print the first six records
head(HmeqScrubbed)        

############################################################################

# ****** STEP 2: CLASSIFICATION MODELS – PREDICT TARGET_BAD_FLAG ******

# ******  2.1:- For classification, making TARGET_BAD_FLAG is a factor ******
# Classification, making TARGET_BAD_FLAG is a factor
HmeqScrubbed$TARGET_BAD_FLAG <- as.factor(HmeqScrubbed$TARGET_BAD_FLAG)

# NOTES:
# converted TARGET_BAD_FLAG to a factor for rpart library to predict the 
# variable and treat as the classification test. If it stays numeric (0/1),
# rpart might treat it as regression instead.
# The levels is kept as 0/1 because it can easily pull the probability for 
# the '1' as default class when building ROC curves.

# ****** 2.2:- Train/Test Split ******
set.seed(1)  # Controls the random split

# Number of observations in the dataset.
n <- nrow(HmeqScrubbed)

# Create a TRUE/FALSE flag to split rows into train and test.
SamFCol <- sample(c(TRUE, FALSE),  # TRUE (training set) , FALSE (test set)
                  size    = n, # Number of observations
                  replace = TRUE,
                  prob    = c(0.7, 0.3))  # ~70% train, 30% test

# NOTES:
# choosed 70/30 split because:
# 70% gives enough data for the tree to learn patterns.
# 30% is large enough to reliably measure performance on unseen borrowers.

# Create the actual training and testing data frames.
TrainTCol <-HmeqScrubbed[SamFCol, ] # rows where flag is TRUE
TestFCol <- HmeqScrubbed[!SamFCol, ] # rows where flag is False

# Check split sizes
dim(TrainTCol)  # Show dimensions (rows, columns) of training data
dim(TestFCol)  # Show dimensions of testing data

# For gbm we also need a numeric 0/1 version of the flag
TrainTCol$TARGET_BAD_FLAG_NUM <- as.numeric(as.character(TrainTCol$TARGET_BAD_FLAG))  # factor -> character -> numeric
TestFCol$TARGET_BAD_FLAG_NUM  <- as.numeric(as.character(TestFCol$TARGET_BAD_FLAG))   # same for test set

# NOTES:
# Split the data into training and testing so we can check how well the tree
# works on new borrowers instead of just memorizing the training rows.
# If we trained and tested on the same rows the tree could look very accurate but fail on new customers (overfitting).
# gbm we also need a numeric 0/1 version of the flag

# ****** 2.3:- Decision Tree CLASSIFICATION MODEL ( GINI) ******
# Build a classification tree to predict TARGET_BAD_FLAG
TreeClass <- rpart(
  TARGET_BAD_FLAG ~ . - TARGET_LOSS_AMT,  # Target = TARGET_BAD_FLAG, use all other variables except TARGET_LOSS_AMT
  data = TrainTCol, # Use training data
  method = "class", # Classification tree
  parms  = list(split = "gini"), # Use Gini index for splitting
  control = rpart.control(
    maxdepth = 10, # Maximum depth of the tree (avoid overly deep trees)
    minbucket = 50 # Minimum number of records in each terminal node
  ))

# NOTES:
# Limit the tree depth to 10 to reduce overfitting while still allowing
# some interactions between predictors.
# Gini- prefers splits that creates "pure" nodes mostly good or bad.

# Plot the decision tree for visual interpretation
rpart.plot(
  TreeClass,                              # Tree model to plot
  main = "Classification Tree for TARGET_BAD_FLAG", # Title
  type = 2  # Type 2 = left/right split labels
)

# NOTES:
# A top split on a delinquency or derogatory variable suggests that borrowers
# with past credit issues are much more likely
# Default again, which matches real-world lending experience.
# Lower-level splits on income or loan amount often refine "borderline"
# borrowers into higher- and lower-risk subgroups.

# Show variable importance from the decision tree

# Extract variable importance from both trees.
# Higher values mean the variable is used more often or creates
# better splits in the tree.

GVariable <- TreeClass$variable.importance  # Higher values = more important predictors in the tree
GVariable # print the output 

# NOTES:
# Larger balances or higher debt vs property value is more financial stress.
# Low or unstable income means less capacity to repay.
# This helps explain to risk managers and regulators which credit risk
# factors are driving decisions instead of treating the model as a black box.

# Method:  
# Checks at the top splits and variable importance. 
# Asks if these predictors make sense for default risk (e.g., credit history,
# delinquencies, income) If yes, that supports that the tree is reasonable.

# ****** 2.4: RANDOM FOREST CLASSIFICATION MODEL ****** 

# Build a Random Forest model to predict TARGET_BAD_FLAG
RFClass <- randomForest(
  TARGET_BAD_FLAG ~ . - TARGET_LOSS_AMT,  # Same target and predictors as the tree
  data = TrainTCol, # Training data
  ntree = 500, # Number of trees in the forest 
  importance = TRUE, # Ask for variable importance measures
  na.action = na.omit # Remove rows with NA for modeling
)

# Print Random Forest summary
RFClass 

# NOTES:
# randomForest() uses bootstrap samples of the training data and builds trees.
# The final prediction is a “majority vote” comparing all trees.
# ntree = 500 is a common default. It usually gives a stable OOB error
# without taking too long.
# importance = TRUE tells the function to track how much each variableimproves model accuracy.

# Print numeric importance measures for each variable
importance(RFClass)

# NOTES:
# MeanDecreaseAccuracy: how much the model’s accuracy drops when that 
# variable is randomly permuted (higher = more important).

# MeanDecreaseGini: how much the variable helps reduce node impurity
# across the forest.


# Plot variable importance chart
varImpPlot(
  RFClass,# Random Forest model
  main = "Random Forest – Variable Importance (Classification)" # Chart title
)

# NOTES:
#  varImpPlot() gives a simple bar chart that is easy to show to a manager or a user.
#  The top bars are the variables the Random Forest which depends on most.

# ****** 2.5: GRADIENT BOOSTING CLASSIFICATION MODEL ****** 

# Build Gradient Boosting model (GBM) for classification of the flag
GBMClass <- gbm(
  formula = TARGET_BAD_FLAG_NUM ~ . - TARGET_BAD_FLAG - TARGET_LOSS_AMT, # Numeric flag ~ predictors 
  distribution = "bernoulli", # 0/1  binary classification 
  data = TrainTCol, # Training data
  n.trees = 500, # Number of small trees in the ensemble
  interaction.depth = 3, # Depth of each small tree
  shrinkage = 0.01, # Learning rate (smaller = slower but more stable)
  n.minobsinnode = 30, # Minimum observations in each terminal node
  bag.fraction = 0.8, # Fraction of training set used for each tree (stochastic boosting)
  train.fraction = 1.0 # Use all training data to train
)

# NOTES:
# GBM builds trees sequentially, where each new tree tries to correct the
# mistakes of the previous trees.
# Using 500 trees and a Bernoulli distribution

summary(GBMClass) # Show variable for GBM model

# NOTES:
# The top predictors and their relative influence.
# Relative influence tells us how much each variable contributes to 
# reducing the loss function across all boosted trees.
# It can be used to explain which borrower characteristics drive the probability of default.

#  ****** 2.6: ROC CURVES & AUC (SINGLE SPLIT)  ****** 

# METHOD:
# Use ROC curves and AUC: Default is usually less frequent than non-default.
# AUC measures how well the model ranks  bad above good

# Get predicted probabilities of default (class "1") for each model on TEST data
Prob_DT_Test <- predict(
  TreeClass, # Decision tree classification model
  newdata = TestFCol, # Testing data frame
  type = "prob")[, "1"] # column "1" = P(default)

# Random Forest predicted probabilities on Test
Prob_RF_Test <- predict(
  RFClass, # Random Forest model
  newdata = TestFCol,  # Testing data
  type = "prob"  # Get class probabilities
)[, "1"]   # Probability of default

# Gradient Boosting predicted probabilities on Test
Prob_GBM_Test <- predict(
  GBMClass, # GBM classification model
  newdata = TestFCol, # Testing data
  n.trees = 500, # Use all 500 trees
  type = "response" # Returns probability of default (0–1)
)

# True labels as numeric 0/1 for ROCR
True_FLAG_Test <- TestFCol$TARGET_BAD_FLAG_NUM  

# NOTES:
# Numeric 0/1 version of TARGET_BAD_FLAG that we created earlier.
# ROCR expects numeric labels for classification.

# Build ROCR prediction objects for ROC
pred_DT  <- prediction(Prob_DT_Test,  True_FLAG_Test)   # Tree
pred_RF  <- prediction(Prob_RF_Test,  True_FLAG_Test)   # RF
pred_GBM <- prediction(Prob_GBM_Test, True_FLAG_Test)   # GBM

# NOTE:
# Each prediction() call:
# First argument: vector of predicted probabilities
# Second argument: true 0/1 labels
# Returns a "prediction" object that ROCR can use for ROC, AUC, etc

# Build ROC performance objects (TPR vs FPR) for each model
perf_DT  <- performance(pred_DT,  "tpr", "fpr")  # ROC curve for decision tree
perf_RF  <- performance(pred_RF,  "tpr", "fpr") # ROC for random forest
perf_GBM <- performance(pred_GBM, "tpr", "fpr")  # ROC for gradient boosting

# NOTE:
# performance(..., "tpr", "fpr")
# Uses that prediction object to compute:
# TPR = True Positive Rate (a.k.a. sensitivity / recall)
# FPR = False Positive Rate
# This is exactly needed to plot the ROC curve (TPR vs FPR at different cutoff thresholds).

# Plot ROC curves on the same graph
plot(
  perf_DT, # First plot: Decision Tree ROC
  col = "black", # Black line color
  main = "ROC – Classification Models (Test Data)" #  plot title
)
plot(perf_RF,  col = "red",  add = TRUE)   # Add Random Forest ROC in red
plot(perf_GBM, col = "blue", add = TRUE)   # Add GBM ROC in blue

abline(0, 1, lty = 2) # Add diagonal line = random guessing

legend("bottomright", # Place legend at bottom-right of the plot
       legend = c("Decision Tree", "Random Forest", "Gradient Boosting"), # Legend text
       col    = c("black", "red", "blue"), # Colors matching the curves
       lty    = 1) # Line type = solid

# Calculate AUC (Area Under the Curve) for each model
AUC_DT_Test  <- performance(pred_DT,  "auc")@y.values[[1]]   # AUC value for Tree
AUC_RF_Test  <- performance(pred_RF,  "auc")@y.values[[1]]   # AUC value for RF
AUC_GBM_Test <- performance(pred_GBM, "auc")@y.values[[1]]   # AUC value for GBM

#NOTES:
# performance(pred, "auc") returns an object; the actual numeric AUC value
# is stored inside @y.values[[1]].

AUC_DT_Test  # Print AUC for decision tree
AUC_RF_Test # Print AUC for random forest
AUC_GBM_Test # Print AUC for gradient boosting

# NOTES:
# AUC_DT_Test : 1
# AUC_RF_Test : 1
# AUC_GBM_Test: 0.9097372

# Interpretation:
# All models perform very well at ranking bad loans above good loans.
# Decision Tree and Random Forest achieve perfect AUC for this split.
# The GBM model also performs strongly (AUC > 0.90) and is likely more stable.
# Final model choice should be based on average AUC across multiple runs.

# ****** STEP 2.7: MULTIPLE SEEDS – STABILITY CHECK (CLASSIFICATION) ******

SeedsClass <- c(1, 2, 3, 4) # Vector of different random seeds to try

# Creating the dataframe for seed,# at least 3 runs
ResultsClass <- data.frame() # Empty data frame to store AUC results

for (s in SeedsClass) { # Loop over each seed
  set.seed(s) # Set seed for this iteration
  n <- nrow(HmeqScrubbed) # Number of observations
  SamFlag <- sample(c(TRUE, FALSE), # New random 70/30 split
                    size = n,
                    replace = TRUE,
                    prob = c(0.7, 0.3))
  
  TrainTCol <- HmeqScrubbed[SamFlag, ]   # Training data for this seed
  TestFCol <- HmeqScrubbed[!SamFlag, ]  # Testing data for this seed
  
  # Make numeric copy of the flag for gbm and ROC
  TrainTCol$TARGET_BAD_FLAG_NUM <- as.numeric(as.character(TrainTCol$TARGET_BAD_FLAG))
  TestFCol$TARGET_BAD_FLAG_NUM  <- as.numeric(as.character(TestFCol$TARGET_BAD_FLAG))
  
  # Decision Tree model 
  TreeClass <- rpart(
    TARGET_BAD_FLAG ~ . - TARGET_LOSS_AMT,
    data = TrainTCol,
    method = "class",
    parms  = list(split = "gini"),
    control = rpart.control(maxdepth = 8, minbucket = 50)
  )
  
  # Random Forest model
  RFClass <- randomForest(
    TARGET_BAD_FLAG ~ . - TARGET_LOSS_AMT,
    data = TrainTCol,
    ntree = 500,
    importance = FALSE,
    na.action = na.omit
  )
  
  # Gradient Boosting model 
  GBMClass <- gbm(
    TARGET_BAD_FLAG_NUM ~ . - TARGET_BAD_FLAG - TARGET_LOSS_AMT,
    distribution = "bernoulli",
    data = TrainTCol,
    n.trees = 500,
    interaction.depth = 3,
    shrinkage = 0.01
  )
  
  # Predicted probabilities on test set
  Prob_DT_Test  <- predict(TreeClass, newdata = TestFCol, type = "prob")[, "1"]
  Prob_RF_Test  <- predict(RFClass,  newdata = TestFCol, type = "prob")[, "1"]
  Prob_GBM_Test <- predict(GBMClass, newdata = TestFCol, n.trees = 500, type = "response")
  
  True_FLAG_Test <- TestFCol$TARGET_BAD_FLAG_NUM   # True numeric labels
  
  # Compute AUC for each model
  AUC_DT  <- performance(prediction(Prob_DT_Test,  True_FLAG_Test), "auc")@y.values[[1]]
  AUC_RF  <- performance(prediction(Prob_RF_Test,  True_FLAG_Test), "auc")@y.values[[1]]
  AUC_GBM <- performance(prediction(Prob_GBM_Test, True_FLAG_Test), "auc")@y.values[[1]]
  
  # Note:
  # ResultsClass shows AUC for Tree, RF, and GBM across multiple random seeds.
  # Higher AUC = better ranking of bad vs good loans. 
  
  ResultsClass <- rbind(
    ResultsClass,
    data.frame(
      Seed = s,
      AUC_DT_Test  = AUC_DT,
      AUC_RF_Test  = AUC_RF,
      AUC_GBM_Test = AUC_GBM
    )
  )
}

ResultsClass # View AUC results across seeds

# Seed AUC_DT_Test AUC_RF_Test AUC_GBM_Test
#     1           1           1    0.9093863
#     2           1           1    0.9104740
#     3           1           1    0.9122007
#     4           1           1    0.9221560

# Model Recommendation:
# As we can compare all models, they perform very well, I would recommend the Random Forest
# model because it combines very high accuracy with greater stability
# than a single decision tree

# Interpretation (Multiple Seeds – Test Data):
# Comparing all four random seeds, the Decision Tree and Random Forest both
# achieve AUC = 1.0, showing perfect ranking of default vs non-default
# While the Decision Tree performs very well, it is known to be sensitive
# to changes in the training data.
# The Random Forest also achieves perfect AUC.
# The Gradient Boosting model has slightly lower AUC values (≈ 0.91–0.92),
# but its performance is very consistent across seeds.

############################################################################

# ****** STEP 3: REGRESSION MODELS – PREDICT TARGET_LOSS_AMT ******

# ****** 3.1: PREPARE DATA FOR REGRESSION ******

# Create a separate copy of the dataset for regression modeling
HmeqScrubbed_Reg <- HmeqScrubbed 

# Ensure TARGET_LOSS_AMT is numeric 
HmeqScrubbed_Reg$TARGET_LOSS_AMT <- as.numeric(HmeqScrubbed_Reg$TARGET_LOSS_AMT)

# NOTE:
# Ensure TARGET_LOSS_AMT is numeric
# This avoids errors if the column was read as character or factor

# ****** 3.2: SINGLE TRAIN/TEST SPLIT (DETAILED REGRESSION RUN) ******

set.seed(1) # Fix seed so results are reproducible

n <- nrow(HmeqScrubbed_Reg) # Number of observations

SamReg <- sample(c(TRUE, FALSE), # 70/30 logical split
                 size = n,
                 replace = TRUE,
                 prob = c(0.7, 0.3))

# NOTES:
# 70% gives the tree enough data to learn patterns in loss amounts.
# 30% is large enough to evaluate performance on unseen borrowers.

TrainReg <- HmeqScrubbed_Reg[SamReg, ]   # Training data for regression
TestReg  <- HmeqScrubbed_Reg[!SamReg, ]  # Testing data for regression

dim(TrainReg)  # Dimensions of regression training data
dim(TestReg)  # Dimensions of regression testing data

# NOTES:
# Splitting the data for the regression experiment
# Train on TrainReg and evaluate on TestReg to see if the model generalizes to new borrowers.
# Using a separate test set helps verify that the model generalizes
# to new borrowers rather than memorizing the training data.

# ****** STEP 3.3: DECISION TREE REGRESSION MODEL ******

TreeReg <- rpart(
  TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG, # Predict loss amount; exclude FLAG predictor
  data = TrainReg, # Regression training data
  method = "anova", # Regression tree (continuous output)
  control = rpart.control(
    maxdepth = 8, # Limit depth
    minbucket = 50 # Minimum records per leaf
  )
)

# Plot the regression tree
rpart.plot(
  TreeReg,
  main = "Regression Tree for TARGET_LOSS_AMT",
  type = 2,
  extra = 101 # Show prediction in each node
)

# Extract variable importance from both trees.
TreeReg$variable.importance # Show important predictors for loss amount

# Predict loss on test data using the tree
Pred_DTReg_Test <- predict(TreeReg, newdata = TestReg)

# Compute RMSE for decision tree regression
RMSE_DTReg_Test <- rmse(TestReg$TARGET_LOSS_AMT, Pred_DTReg_Test)
RMSE_DTReg_Test  # Print tree RMSE

# NOTES:
#  This model predicts the expected loss amount for each borrower.
#  RMSE measures average prediction error in dollar terms.
# The decision tree is easy to interpret but may be less accurate than Random Forest or Gradient Boosting.

# ****** 3.4: RANDOM FOREST REGRESSION MODEL  ******

RFReg <- randomForest(
  TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG,  # Predict loss; exclude FLAG
  data = TrainReg,
  ntree = 500, # Number of trees
  importance = TRUE,  # Request variable importance
  na.action = na.omit # Handle missing values
)

RFReg  # Print RF regression summary
importance(RFReg)  # Numeric importance measures

varImpPlot(
  RFReg,
  main = "Random Forest – Variable Importance (Regression)" # Plot of importance
)

# Predict loss on test data using Random Forest
Pred_RFReg_Test <- predict(RFReg, newdata = TestReg)

# Compute RMSE for Random Forest regression
RMSE_RFReg_Test <- rmse(TestReg$TARGET_LOSS_AMT, Pred_RFReg_Test)
RMSE_RFReg_Test # Print RF RMSE

# NOTES:
# Random Forest regression averages many trees to improve accuracy.
# It is usually more accurate than a single regression tree.
# RMSE is reported in dollar terms, so lower values indicate better predictions.

# ******  3.5: GRADIENT BOOSTING REGRESSION MODEL ****** 

GBMReg <- gbm(
  formula = TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG, # Loss amount ~ all predictors except FLAG
  distribution = "gaussian",  # Continuous output
  data = TrainReg,
  n.trees = 500,
  interaction.depth = 3,
  shrinkage = 0.01
)

summary(GBMReg) # Variable importance for GBM regression


# Predict loss on test data using GBM
Pred_GBMReg_Test <- predict(
  GBMReg,
  newdata = TestReg,
  n.trees = 500,
  type = "response"
)

# Compute RMSE for GBM regression
RMSE_GBMReg_Test <- rmse(TestReg$TARGET_LOSS_AMT, Pred_GBMReg_Test)
RMSE_GBMReg_Test # Print GBM RMSE

# NOTES:
# Gradient Boosting builds many small trees sequentially.
# Each new tree focuses on correcting errors from previous trees.
# GBM often provides strong predictive accuracy for loss amount modeling.

# ****** 3.6: MULTIPLE SEEDS – STABILITY CHECK (REGRESSION) ****** 

SeedsReg <- c(1, 2, 3, 4) # Seeds to test
ResultsReg <- data.frame() # Empty table for RMSE results

for (s in SeedsReg) { # Loop over each seed
  
  set.seed(s) # Set random seed
  
  n <- nrow(HmeqScrubbed_Reg) # Number of rows
  SamReg <- sample(c(TRUE, FALSE), # New 70/30 split
                   size = n,
                   replace = TRUE,
                   prob = c(0.7, 0.3))
  
  TrainReg <- HmeqScrubbed_Reg[SamReg, ]
  TestReg  <- HmeqScrubbed_Reg[!SamReg, ]
  
  # Decision Tree regression
  TreeReg <- rpart(
    TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG,
    data = TrainReg,
    method = "anova",
    control = rpart.control(maxdepth = 8, minbucket = 50)
  )
  Pred_DTReg_Test <- predict(TreeReg, newdata = TestReg)
  RMSE_DT <- rmse(TestReg$TARGET_LOSS_AMT, Pred_DTReg_Test)
  
  # Random Forest regression
  RFReg <- randomForest(
    TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG,
    data = TrainReg,
    ntree = 500,
    mtry  = floor(sqrt(ncol(TrainReg) - 2)),
    importance = FALSE,
    na.action = na.omit
  )
  Pred_RFReg_Test <- predict(RFReg, newdata = TestReg)
  RMSE_RF <- rmse(TestReg$TARGET_LOSS_AMT, Pred_RFReg_Test)
  
  # Gradient Boosting regression
  GBMReg <- gbm(
    TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG,
    distribution = "gaussian",
    data = TrainReg,
    n.trees = 500,
    interaction.depth = 3,
    shrinkage = 0.01
  )
  Pred_GBMReg_Test <- predict(GBMReg, newdata = TestReg, n.trees = 500, type = "response")
  RMSE_GBM <- rmse(TestReg$TARGET_LOSS_AMT, Pred_GBMReg_Test)
  
  # Store RMSE results for this seed
  ResultsReg <- rbind(
    ResultsReg,
    data.frame(
      Seed = s,
      RMSE_DT_Test  = RMSE_DT,
      RMSE_RF_Test  = RMSE_RF,
      RMSE_GBM_Test = RMSE_GBM
    )
  )
}

ResultsReg # View RMSE results across seeds

# Seed   RMSE_DT_Test  RMSE_RF_Test  RMSE_GBM_Test
#   1     5249.729     4192.548      4608.974
#   2     5646.054     4672.353      5048.785
#   3     5311.507     4271.896      4661.315
#   4     5777.667     4711.020      5137.640

# Interpretation:
# (Multiple Seeds – Regression RMSE)
# RMSE measures the average dollar error in predicting TARGET_LOSS_AMT. Lower RMSE
# indicates better regression performance.
# Comparing with all four seeds, the Random Forest model consistently produces
# the lowest RMSE tends to $4,200 – $4,700 , indicating the most accurate and
# stable predictions of loss amount.

# The Decision Tree has the highest RMSE in every run, showing that a
# single tree is less accurate for predicting continuous loss values.

# Gradient Boosting performs better than the Decision Tree but is slightly
# less accurate than Random Forest in this experiment.

# Model Recommendation:
# I would recommend the Random Forest regression model because it provides
# the lowest and most consistent RMSE across multiple train/test splits
############################################################################

# ****** STEP 4: PROBABILITY / SEVERITY MODEL – P(DEFAULT) * LOSS-GIVEN-DEFAULT ******

# Use a classification model to estimate P(default) = P(TARGET_BAD_FLAG = 1).
# Use a regression model trained only on default cases (FLAG = 1) to
# estimate Loss-Given-Default (LGD = TARGET_LOSS_AMT | default).
# Multiply P(default) * LGD to get Expected Loss.
# Compare RMSE of this Probability/Severity model vs direct regression from Step 3.

# ****** 4.1: TRAIN/TEST SPLIT FOR PS MODEL ******

set.seed(1) # Seed for reproducibility

n <- nrow(HmeqScrubbed)  # Number of rows in original data
SamPS <- sample(c(TRUE, FALSE),  # 70/30 split
                size = n,
                replace = TRUE,
                prob = c(0.7, 0.3))

TrainPS <- HmeqScrubbed[SamPS, ] # Training data for PS modeling
TestPS  <- HmeqScrubbed[!SamPS, ] # Testing data for PS modeling

# Numeric copy of flag for gbm models
TrainPS$TARGET_BAD_FLAG_NUM <- as.numeric(as.character(TrainPS$TARGET_BAD_FLAG))
TestPS$TARGET_BAD_FLAG_NUM  <- as.numeric(as.character(TestPS$TARGET_BAD_FLAG))

# NOTES:
# This split is used for the combined Probability/Severity model.
# Classification predicts P(default); regression predicts loss given default.

# ****** 4.2: CLASSIFICATION MODEL FOR P(DEFAULT) ******

# I choose Gradient Boosting for P(default), but we could choose RF or Tree.
GBMClass_PS <- gbm(
  TARGET_BAD_FLAG_NUM ~ . - TARGET_BAD_FLAG - TARGET_LOSS_AMT, # Numeric flag ~ predictors (exclude factor flag and loss)
  distribution = "bernoulli", # Binary classification
  data = TrainPS,
  n.trees = 500,
  interaction.depth = 3,
  shrinkage = 0.01
)

summary(GBMClass_PS)  # Important variables for default probability

# Predict probability of default on TestPS
P_Default_Test <- predict(
  GBMClass_PS,
  newdata = TestPS,
  n.trees = 500,
  type = "response" # Returns probabilities (0–1)
)

# ****** 4.3: SEVERITY MODELS – LOSS GIVEN DEFAULT (LGD) ******

# Use only default cases (FLAG = 1) in training data to model severity
TrainPS_Defaults <- subset(TrainPS, TARGET_BAD_FLAG == "1") # Records where default happened
TestPS_Defaults  <- subset(TestPS,  TARGET_BAD_FLAG == "1") # Defaults in test data (for extra checks if needed)

# Decision Tree for severity
TreeSev <- rpart(
  TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG - TARGET_BAD_FLAG_NUM,  # Predict loss; exclude flags
  data = TrainPS_Defaults,
  method = "anova",
  control = rpart.control(maxdepth = 8, minbucket = 30)
)

TreeSev$variable.importance # Important predictors for loss given default

# Random Forest for severity
RFSev <- randomForest(
  TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG - TARGET_BAD_FLAG_NUM,
  data = TrainPS_Defaults,
  ntree = 500,
  mtry  = floor(sqrt(ncol(TrainPS_Defaults) - 2)),
  importance = TRUE,
  na.action = na.omit
)

importance(RFSev)  # Numeric variable importance for severity
varImpPlot(RFSev, main = "Random Forest – Severity (LGD)") # Plot of importance

# Gradient Boosting for severity (we will use this one for final PS model)
GBMSev <- gbm(
  TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG - TARGET_BAD_FLAG_NUM,
  distribution = "gaussian",  # Continuous outcome = loss amount
  data = TrainPS_Defaults,
  n.trees = 500,
  interaction.depth = 3,
  shrinkage = 0.01
)

summary(GBMSev)  # Important variables for LGD in GBM

# ****** 4.4: EXPECTED LOSS = P(DEFAULT) * LGD ******

# Predict loss given default (LGD) for ALL records in TestPS
LGD_Test <- predict(
  GBMSev,
  newdata = TestPS, # We predict loss even for non-defaults; PS formula adjusts via P(default)
  n.trees = 500,
  type = "response"
)

# Compute Expected Loss for each record
ExpectedLoss_Test <- P_Default_Test * LGD_Test   # P(default) * loss if default

# Actual loss amounts in TestPS
ActualLoss_Test <- TestPS$TARGET_LOSS_AMT

# RMSE for Probability/Severity model
RMSE_PS_Test <- rmse(ActualLoss_Test, ExpectedLoss_Test)
RMSE_PS_Test # Print RMSE of PS model

# NOTE:
# This Probability/Severity model separates default risk from loss size.
# P(default) captures how likely a borrower is to default.
# LGD captures how large the loss is when default occurs.
# Combining them provides an expected loss that is often more realistic
# than using a single regression model for loss amount.

# ****** STEP 4.5: DIRECT REGRESSION COMPARISON (SAME SPLIT) ******

# Direct Gradient Boosting regression using the same TrainPS/TestPS split
GBMReg_PS <- gbm(
  TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG,  # Predict loss directly; exclude FLAG as predictor
  distribution = "gaussian",
  data = TrainPS,
  n.trees = 500,
  interaction.depth = 3,
  shrinkage = 0.01
)

# Predict loss directly on TestPS
Pred_GBMReg_PS_Test <- predict(
  GBMReg_PS,
  newdata = TestPS,
  n.trees = 500,
  type = "response"
)

# RMSE for direct regression model
RMSE_DirectReg_Test <- rmse(ActualLoss_Test, Pred_GBMReg_PS_Test)
RMSE_DirectReg_Test  # Print RMSE for direct regression

# RMSE Results (Test Data):
# Direct GBM Regression RMSE : 1816.60
# Probability/Severity RMSE  : 4232.48

# NOTES:
# The direct Gradient Boosting regression model has a much lower RMSE,
#   meaning it predicts loss amounts more accurately in dollar terms.
# The Probability/Severity (PS) model has higher RMSE because it separates
#   the problem into two steps: predicting default probability and loss size.
# PS model is more interpretable and aligns well with real-world credit risk modeling practices.

# Recommendation:
# If pure prediction accuracy is the goal, the direct GBM regression is preferred.
# If model risk decomposition are more important, the Probability/Severity model is a better choice.

# ****** 4.6: MULTIPLE SEEDS FOR PS MODEL ******
SeedsPS <- c(1, 2, 3, 4)  # Seeds to test for PS model
ResultsPS <- data.frame() # Table to store RMSE results

for (s in SeedsPS) { # Loop over seeds
  
  set.seed(s) # Fix seed
  
  n <- nrow(HmeqScrubbed) # Number of rows
  SamPS <- sample(c(TRUE, FALSE), # 70/30 split
                  size = n,
                  replace = TRUE,
                  prob = c(0.7, 0.3))
  
  TrainPS <- HmeqScrubbed[SamPS, ]
  TestPS  <- HmeqScrubbed[!SamPS, ]
  
  TrainPS$TARGET_BAD_FLAG_NUM <- as.numeric(as.character(TrainPS$TARGET_BAD_FLAG))
  TestPS$TARGET_BAD_FLAG_NUM  <- as.numeric(as.character(TestPS$TARGET_BAD_FLAG))
  
  # Classification GBM for P(default)
  GBMClass_PS <- gbm(
    TARGET_BAD_FLAG_NUM ~ . - TARGET_BAD_FLAG - TARGET_LOSS_AMT,
    distribution = "bernoulli",
    data = TrainPS,
    n.trees = 500,
    interaction.depth = 3,
    shrinkage = 0.01,
    n.minobsinnode = 30,
    bag.fraction = 0.8,
    train.fraction = 1.0,
    verbose = FALSE
  )
  
  P_Default_Test <- predict(
    GBMClass_PS,
    newdata = TestPS,
    n.trees = 500,
    type = "response"
  )
  
  # Severity GBM on defaults only
  TrainPS_Defaults <- subset(TrainPS, TARGET_BAD_FLAG == "1")
  
  GBMSev <- gbm(
    TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG - TARGET_BAD_FLAG_NUM,
    distribution = "gaussian",
    data = TrainPS_Defaults,
    n.trees = 500,
    interaction.depth = 3,
    shrinkage = 0.01,
    n.minobsinnode = 30,
    bag.fraction = 0.8,
    train.fraction = 1.0,
    verbose = FALSE
  )
  
  LGD_Test <- predict(
    GBMSev,
    newdata = TestPS,
    n.trees = 500,
    type = "response"
  )
  
  ExpectedLoss_Test <- P_Default_Test * LGD_Test
  
  ActualLoss_Test <- TestPS$TARGET_LOSS_AMT
  
  RMSE_PS <- rmse(ActualLoss_Test, ExpectedLoss_Test)   # PS RMSE
  
  # Direct GBM regression for same split
  GBMReg_PS <- gbm(
    TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG,
    distribution = "gaussian",
    data = TrainPS,
    n.trees = 500,
    interaction.depth = 3,
    shrinkage = 0.01
  )
  
  Pred_GBMReg_PS_Test <- predict(
    GBMReg_PS,
    newdata = TestPS,
    n.trees = 500,
    type = "response"
  )
  
  RMSE_Direct <- rmse(ActualLoss_Test, Pred_GBMReg_PS_Test)  # Direct regression RMSE
  
  # Store both RMSE values for this seed
  ResultsPS <- rbind(
    ResultsPS,
    data.frame(
      Seed = s,
      RMSE_PS_Test      = RMSE_PS,
      RMSE_Direct_Test  = RMSE_Direct
    )
  )
}

ResultsPS # Compare PS vs Direct regression RMSE across seeds

# Seed  RMSE_PS_Test  RMSE_Direct_Test
#  1     4246.780         1878.118
#  2     4809.818         2521.166
#  3     4356.911         2343.492
#  4     4924.501         2602.107

# Interpretation 
# (Multiple Seeds – Probability/Severity vs Direct Regression):
# Across all seeds, the direct GBM regression has lower RMSE (~$1,900–$2,600)
# than the Probability × Severity model (~$4,200–$4,900).
# For this dataset, directly modeling TARGET_LOSS_AMT gives more accurate
# dollar-loss predictions.
# The PS model adds extra noise by splitting into two steps
# (probability of default + loss given default).
# For pure prediction accuracy, I would choose the direct GBM regression.

# Comparison to Step 3:
# The direct GBM regression here performs similarly to the regression models from Step 3,
# showing strong and stable accuracy across multiple train/test splits.
# The RMSE values for each model remain within a reasonable range,
# the results are stable and not driven by random initialization.
# This consistency suggests that the models are neither overfit nor underfit.

# Recommendation:
# If the goal is pure predictive accuracy, I would recommend the direct GBM
# regression model because it consistently achieves the lowest RMSE.
