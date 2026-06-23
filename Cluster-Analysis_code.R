###############################################################################
# Week 8 Assignment -Cluster Analysis
# Data Set: HMEQ_Scrubbed.csv
# Student: Avishek Pradhan
# Date: Feb 24th, 2026
############################################################################

# ****** Overview ******
# This script continues from Week 6 and Week 7.
# Here I selected: 
#  few continuous "risk" variables,
#  run PCA on them,
#  use the PC scores in KMeans clustering,
#  convert the clusters to flexclust so I can score all records,
#  and then use a decision tree to explain what each cluster looks like.
###############################################################################

# ****** REFERENCES ******
#  Lecture on segmentation vs clustering.
#  Lecture on KMeans algorithm and flexclust usage.
#  R example code for segmentation.
#  R example code for basic KMeans clustering.
#  R example of KMeans with iris dataset.
#  PCA + clustering example.
#  Themed clustering example.
###############################################################################

# ****** CLEAN & SETUP : Clean environment and define RMSE ******
# Clear old variables, plots, and console for a fresh start
rm(list = ls())
graphics.off()
cat("\014")   # Clear console

############################################################################

# ****** LOAD LIBRARIES ******
# These are my Week 6/7 libraries. In this Week 8 script I mainly use:
#  flexclust for clustering and prediction
#  rpart / rpart.plot for decision trees

# The other libraries are still listed because I started from my Week 7 code,
# but they are not used in this assignment.

# install.packages("ROCR")           # For ROC curves and AUC calculation
# install.packages("randomForest")   # For Random Forest models
# install.packages("gbm")            # For Gradient Boosting models
# install.packages("MASS")           # stepAIC for variable selection
# install(Rtsne)                     # tSNE implementation in R
# install.packages(ggplot2)          # sketch the graph

# USED LIBARIES
# install.packages("flexclust")
# install.packages("rpart")          # For decision trees
# install.packages("rpart.plot")     # For pretty decision tree plots


library(ROCR)                         # Load ROC / AUC evaluation functions
library(randomForest)                 # Load Random Forest implementation
library(gbm)                          # Load Gradient Boosting Machines implementation
library(MASS)                         # For stepAIC() - forward/backward/stepwise selection
library(Rtsne)                        # tSNE implementation in R
library (ggplot2)                     # sketch the graph
library(dplyr)                        # using Pipe operators

library(flexclust)
library(rpart)                        # Load decision tree library
library(rpart.plot)                   # Load decision tree plotting library
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


TargetFlag <- "TARGET_BAD_FLAG"
TargetAmt  <- "TARGET_LOSS_AMT"

# For ROC/AUC we want a numeric 0/1 version of the target flag
HmeqScrubbed$TARGET_BAD_FLAG_NUM <- as.numeric(as.character(HmeqScrubbed[[TargetFlag]]))

###############################################################################
# ****** STEP 2: PCA ANALYSIS (CONTINUOUS INPUT VARS, RISK THEME) ******

# Use only input variables (no targets).
# Use only continuous variables (no flag variables).
# Select at least 4 continuous variables with a common theme.
# Run PCA (center and scale).
# Show a Scree plot, decide #PCs.
# Print weights (loadings) and tell a story.
# Scatter plot of PC1 vs PC2 with black points.

#  ****** 2.1 Identify predictor sets ******

AllPredictors <- setdiff(names(HmeqScrubbed),
                         c(TargetFlag, TargetAmt, "TARGET_BAD_FLAG_NUM"))

# Numeric predictors only
NumMask <- sapply(HmeqScrubbed[, AllPredictors, drop = FALSE], is.numeric)
NumericPredictors <- AllPredictors[NumMask]

# Continuous = numeric with >2 unique values (rough filter for flags)
ContinuousVars <- NumericPredictors[
  sapply(NumericPredictors, function(v) length(unique(HmeqScrubbed[[v]])) > 2)
]

# Remaining predictors are flags / categoricals
Flag_or_CatVars <- setdiff(AllPredictors, ContinuousVars)

ContinuousVars
Flag_or_CatVars

# ****** 2.2 Choose a risk theme subset for PCA  ******

# I selected  a set of variables that all relate to credit risk:
#   IMP_DEROG  : number of prior serious derogatory reports
#   IMP_DELINQ : number of recent delinquencies
#   IMP_NINQ   : number of recent credit inquiries
#   IMP_CLAGE  : age of oldest credit line
#   IMP_DEBTINC: debt-to-income ratio

# These all fit a "risk behavior" theme, so I use them in PCA together instead
# of throwing every numeric column into the analysis.

RiskThemeVars <- c("IMP_DEROG",
                   "IMP_DELINQ",
                   "IMP_NINQ",
                   "IMP_CLAGE",
                   "IMP_DEBTINC")

RiskThemeVars <- intersect(RiskThemeVars, ContinuousVars)
RiskThemeVars   # final list used

HmeqPCA <- HmeqScrubbed[, RiskThemeVars, drop = FALSE]
summary(HmeqPCA)


#  ****** 2.3 PCA with centering and scaling ******

# I center and scale here so all the risk variables are on the same scale
# before PCA. This is the same setup we used in the insurance PCA lecture.

PcaModel <- prcomp(HmeqPCA,
                   center = TRUE,
                   scale. = TRUE)

# Eigenvalues (variance of each PC)
PcaEigen <- PcaModel$sdev^2
PcaEigen

# Proportion of variance explained
ExplainedVar <- PcaEigen / sum(PcaEigen)
ExplainedVar

# Scree plot to visualize how many PCs are useful
plot(1:length(PcaEigen),
     PcaEigen,
     type = "b",
     xlab = "Principal Component",
     ylab = "Eigenvalue",
     main = "Scree Plot – PCA on Risk Theme Variables")

#  ****** 2.4 Decide number of PCs to keep  ******

# I use the common rule of keeping PCs with eigenvalue > 1.
# In this case, two PCs meet that rule.
# The scree plot also shows a clear drop after PC2, so using 2 PCs
# seems reasonable for clustering.

Num_PCs_Eig <- sum(PcaEigen > 1)
Num_PCs_Use <- max(2, Num_PCs_Eig)

cat("PCs with eigenvalue > 1:", Num_PCs_Eig, "\n")
cat("PCs chosen for clustering:", Num_PCs_Use, "\n")

# PCs with eigenvalue > 1: 2 
# PCs chosen for clustering: 2 

# Based on both the eigenvalue rule and the scree plot shape,
# I will use the first two principal components for clustering.


#  ****** 2.5 Print loadings (weights) and interpret by hand  ******

PCA_Loadings <- PcaModel$rotation
PCA_Loadings[, 1:Num_PCs_Use]

# After looking at the loadings:

# PC1 has strong positive weights on IMP_DEROG, IMP_DELINQ, and IMP_NINQ.
# It also has a smaller positive weight on IMP_DEBTINC and a negative weight on IMP_CLAGE.
# This suggests PC1 is capturing overall credit risk behavior
# higher values mean more derogatory marks, more delinquencies,
# more recent inquiries, and slightly higher debt load.
# Since IMP_CLAGE is negative, shorter credit history also pushes PC1 higher.
# So I interpret PC1 as a general "high recent credit risk activity" component.

# PC2 looks different. It has strong negative weights on IMP_DELINQ and IMP_CLAGE,
# but positive weights on IMP_DEBTINC and IMP_NINQ.
# This suggests PC2 separates borrowers with high debt-to-income
# and more inquiries from those with longer credit history and fewer delinquencies.
# I see PC2 as distinguishing "debt pressure vs credit stability."

#  ****** 2.6 Add PC scores to data and plot PC1 vs PC2  ******

PCA_Scores <- predict(PcaModel, newdata = HmeqPCA)

HmeqScrubbed$PC1 <- PCA_Scores[, 1]
HmeqScrubbed$PC2 <- PCA_Scores[, 2]

plot(HmeqScrubbed$PC1,
     HmeqScrubbed$PC2,
     xlab = "PC1",
     ylab = "PC2",
     main = "PCA – PC1 vs PC2 (Risk Theme)",
     col  = "black",
     pch  = 16)

###############################################################################
# ****** STEP 3:  Cluster Analysis - Find the Number of Clusters ******

# Use the principal components from Step 2.
# Run KMeans for N = 1..10 (or more).
# Plot WSS (Scree) to find the elbow.
# Choose the "best" K and justify in words.

PC_for_Clustering <- as.data.frame(PCA_Scores[, 1:Num_PCs_Use])

MAX_K <- 10
WSS   <- numeric(MAX_K)

set.seed(123)  # keep random start consistent

# I run KMeans for K = 1 to 10.
# For each K, I store the total within-cluster sum of squares (WSS).
# This lets me create the "elbow" plot to see where adding more clusters
# stops improving the model significantly.

for (k in 1:MAX_K) {
  km_temp <- kmeans(PC_for_Clustering,
                    centers = k,
                    nstart  = 20)  # use multiple random starts so KMeans
                                    # does not get stuck in a poor local solution
  WSS[k] <- km_temp$tot.withinss
}

plot(1:MAX_K,
     WSS,
     type = "b",
     xlab = "Number of Clusters (K)",
     ylab = "Total Within-Cluster SS",
     main = "KMeans Scree Plot (PC-based Clusters)")

# From the scree plot, there is a sharp drop from K = 1 to K = 3,
# and then the curve starts to flatten.
# This suggests that 3 clusters capture most of the structure
# without overcomplicating the model.

K_Use <- 3  # Based on the elbow pattern in the WSS scree plot,
            # I choose K = 3 clusters for the final model.
cat("Number of clusters chosen (K_Use):", K_Use, "\n")

###############################################################################
# ****** STEP 4: Cluster Analysis ******

# Run KMeans with chosen K.
# Show cluster sizes and centers.
# Convert to flexclust (as.kcca).
# Barplot of cluster sizes.
# Score data (cluster membership).
# PC1 vs PC2 colored by cluster + legend.
# Check if clusters predict loan default.

# ***** 4.1 Final KMeans model ****** 

set.seed(123)
km_final <- kmeans(PC_for_Clustering,
                   centers = K_Use,
                   nstart  = 20)

km_final$size
km_final$centers

# Interpretation of cluster centers (in PC space):

# Cluster 1 (PC1 = 0.36, PC2 = 0.73):
# This group has moderately positive PC1 and clearly positive PC2.
# Since PC1 represents overall recent credit risk activity,
# and PC2 reflects debt pressure and inquiries,
# this cluster likely represents borrowers with moderate risk
# and relatively higher debt/inquiry activity.

# Cluster 2 (PC1 = 2.77, PC2 = -1.63):
# This cluster has very high PC1, meaning high derogatory marks,
# delinquencies, and inquiries. It also has strongly negative PC2,
# which suggests lower credit stability.
# This appears to be the highest-risk group.
# It is also the smallest cluster (393 records),
# which makes sense for an extreme-risk segment.

# Cluster 3 (PC1 = -0.67, PC2 = -0.42):
# This cluster has negative PC1 and slightly negative PC2.
# Since lower PC1 means fewer delinquencies and fewer derogatory marks,
# this group likely represents lower-risk borrowers
# with more stable credit profiles.

# ******  4.2 Convert to flexclust and barplot ****** 

# flexclust is used so that we can "predict" cluster membership
# later using the kcca object.

kf <- as.kcca(km_final, data = PC_for_Clustering)

barplot(kf,
        main = "Cluster Size Barplot (flexclust)",
        xlab = "Cluster",
        ylab = "Number of Observations")

# ****** 4.3 Score the data with the flexclust object ******  

Cluster_Assign <- predict(kf, newdata = PC_for_Clustering)
HmeqScrubbed$CLUSTER <- factor(Cluster_Assign)

table(HmeqScrubbed$CLUSTER)

# ****** 4.4 PC1 vs PC2 colored by cluster ****** 

plot(HmeqScrubbed$PC1,
     HmeqScrubbed$PC2,
     col  = HmeqScrubbed$CLUSTER,
     pch  = 16,
     xlab = "PC1",
     ylab = "PC2",
     main = "PC1 vs PC2 Colored by Cluster")

legend("topright",
       legend = levels(HmeqScrubbed$CLUSTER),
       col    = 1:length(levels(HmeqScrubbed$CLUSTER)),
       pch    = 16,
       title  = "Cluster")


# ******  4.5 Check relation between clusters and loan default ****** 

# Here I compare bad-loan rates across clusters.
# If some clusters have a much higher TARGET_BAD_FLAG = 1 rate, that means
# the clustering is picking up real credit risk differences.

tab_cluster_default <- table(HmeqScrubbed$CLUSTER,
                             HmeqScrubbed[[TargetFlag]])

tab_cluster_default

prop_cluster_default <- prop.table(tab_cluster_default, margin = 1)
prop_cluster_default

# Interpretation of clusters based on PC means (barplot):

# Cluster 1 (43% of data):
# PC1 is slightly positive and PC2 is clearly positive.
# This suggests moderate levels of credit risk behavior
# and moderate debt/inquiry activity.
# The default rate (~25%) supports that this is a medium-risk group.

# Cluster 2 (7% of data):
# PC1 is very high and PC2 is strongly negative.
# High PC1 indicates high delinquencies, derogatory marks,
# and more recent credit problems.
# This cluster has the highest default rate (~67%),
# which confirms this is the high-risk borrower segment.

# Cluster 3 (50% of data):
# PC1 is negative and PC2 is slightly negative.
# Lower PC1 suggests fewer delinquencies and better credit history.
# This cluster has the lowest default rate (~9%),
# which indicates this is the safest borrower segment.

# Overall, the clusters appear to separate borrowers into
# low-risk, medium-risk, and high-risk groups.
# The PCA-based clustering is consistent with observed default behavior.

###############################################################################
# ****** STEP 5: DECISION TREE TO DESCRIBE CLUSTERS ******

# Use original data from Step 2 (risk theme vars).
# Predict CLUSTER using a decision tree.
# Plot the tree.
# Use the tree to "tell a story" of each cluster.

# Here I use the original risk-theme variables to predict
# cluster membership. The goal is to understand what
# variables are driving the separation between clusters.
#
# This helps translate the PCA + KMeans clusters into
# business-understandable rules.

df_tree <- HmeqScrubbed[, RiskThemeVars, drop = FALSE]
df_tree$CLUSTER <- HmeqScrubbed$CLUSTER


# Build a classification tree to predict cluster membership.
# The response variable is CLUSTER and the predictors are
# the selected continuous risk variables.
dt_cluster <- rpart(CLUSTER ~ .,
                    data   = df_tree,
                    method = "class")

rpart.plot(dt_cluster,
           main = "Decision Tree for Cluster Membership")


# Interpretation:

# The first split is on credit age (IMP_CLAGE),
# indicating that credit history length is a major factor
# separating the clusters.
#
# Subsequent splits include debt-to-income (IMP_DEBTINC),
# number of inquiries (IMP_NINQ), delinquencies (IMP_DELINQ),
# and derogatory marks (IMP_DEROG).
#
# These are traditional credit risk indicators.
# This confirms that the PCA-based clustering
# is primarily separating borrowers based on
# meaningful risk characteristics.

###############################################################################
# ****** STEP 6: COMMENT – HOW TO USE CLUSTERS IN BUSINESS ******

# These clusters could be used by a financial institution
# to segment borrowers into low-, medium-, and high-risk groups.

# For example:
# The low-risk cluster (Cluster 3) could qualify for lower
#   interest rates or faster approval processes.

# The high-risk cluster (Cluster 2), which shows a very high
#   default rate, could trigger stricter underwriting rules,
#   additional documentation requirements, or risk-based pricing.

# The moderate-risk cluster (Cluster 1) may benefit from
# targeted credit monitoring or structured repayment plans.

# Beyond lending decisions, these clusters could also support:
#  Portfolio risk monitoring
#  Targeted marketing strategies
#  Early warning systems for potential default

# Since the clusters are driven by meaningful credit variables
# (credit age, debt-to-income ratio, delinquencies, inquiries),
# they are interpretable and actionable in a real business setting.

# END OF SCRIPT
###############################################################################