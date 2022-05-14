#Mihir Nevpurkar
#MSN210003


# Clear the workspace
rm(list=ls())
cat("\014")

# LOADING THE DATA AND CONVERTING INTO DATA FRAME ####

bankdata <- read.csv("bank.csv", stringsAsFactors = FALSE)
b <- data.frame(bankdata)
b

#  CONVERTING VARIABLES TO THE FACTOR TYPE : job, marital, education, default,
#housing, loan, contact, month, campaign, poutcome, y ####

b$job <- as.factor(b$job)
b$marital <- as.factor(b$marital)
b$education <- as.factor(b$education)
b$default <- as.factor(b$default)
b$housing <- as.factor(b$housing)
b$loan <- as.factor(b$loan)
b$contact <- as.factor(b$contact)
b$month <- as.factor(b$month)
b$campaign <- as.factor(b$campaign)
b$poutcome <- as.factor(b$poutcome)
b$y <- as.factor(b$y)

#  Creating the required Data Model ##
set.seed(123) # for reproducible results
train <- sample(1:nrow(b), nrow(b)*(2/3)) # replace=FALSE by default


# Use the train index set to split the dataset
#  churn.train for building the model
#  churn.test for testing the model
b.train <- b[train,]   # 6,666 rows
b.test <- b[-train,]   # the other 3,334 rows

#  Classification Tree with rpart
# Important! Comment the following line after installing rpart.
#install.packages('rpart')
library(rpart)

fit <- rpart(y ~ ., # formula, all predictors will be considered in splitting
             data=b.train, # dataframe used
             method="class",  # treat churn as a categorical variable, default
             control=rpart.control(xval=10, minsplit=50), # xval: num of cross validation for gini estimation # minsplit=50: stop splitting if node has 1000 or fewer obs
             parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")

fit  # display basic results

#Q2. B)
# plot tree using built-in function
plot(fit, uniform=TRUE,  # space out the tree evenly
     branch=0.5,         # make elbow type branches
     main="Classification Tree for Churn Prediction",   # title
     margin=0.1)         # leave space so it all fits
text(fit,  use.n=TRUE,   # show numbers for each class
     all=TRUE,           # show data for internal nodes as well
     fancy=FALSE,            # draw ovals and boxes
     pretty=TRUE,           # show split details
     cex=0.8)            # compress fonts to 80%

# plot a prettier tree using rpart.plot

#install.packages('rpart.plot')
library(rpart.plot)
prp(fit, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for CD Prediction")    
rpart.plot(fit, type = 1, extra = 1, main="Classification Tree for Churn Prediction")  


#  Define Confusion Matrix for b.test
# extract the vector of predicted class for each observation in b.test
b.predict <- predict(fit, b.test, type="class")
# extract the actual class of each observation in b.test
b.actual <- b.test$y

# now build the "confusion matrix"
# which is the contingency matrix of predicted vs actual
# use this order: predicted then actual
cm <- table(b.predict, b.actual)  
cm

# Q2 - d) Extracting and defining TN,FN,FP, TP from CM 
TN <- cm[1,1]
FN <- cm[1,2]
FP <- cm[2,1]
TP <- cm[2,2]

# Q2 - e) Use tp, tn, fp and fn to compute the following statistics: False Positive Rate, False
#Negative Rate, Specificity, Sensitivity, and Accuracy.

FPR <- FP/(TN+FP); FPR
FNR <- FN/(FN+TP); FNR
Sensitivity <- TP/(FN+TP); Sensitivity
Specificity <- TN/(TN+FP); Specificity
Accuracy = (TP + TN)/(TP+FP+TN+FN); Accuracy


