####    
####     INTRUSION DETECTOR LEARNING
####

rm(list=ls(all=TRUE))

# Set directory path
dirpath <- "D:/IDLS/Data&Description"
setwd(dirpath)

## Load libraries
library(DMwR)
library(caret)

# Set seed
set.seed(1234)

# Load the train, test and evaluation datasets
train <- read.csv("train_final.csv",header=T)
test <- read.csv("test_final.csv",header=T)
eval <- read.csv("eval_final.csv",header=T)

## Data Preprocessing
# removing ID column = X from train and test
train <- train[,-1]
test <- test[,-1]

# Handling Missing Values
str(train)
sum(is.na(train)) # 257 missing values
data <- train[!is.na(train$Target),] # Removed records that have NAs in Target variable 

apply(data,2,function(x) {sum(is.na(x))}) # To check column wise counts

# Function to compute % of NAs in a vector
rem_na = function(x){
  return(sum(is.na(x))/length(x))
}

na_row_threshold = 0.3
na_rows = which(apply(data, 1, FUN = rem_na) > na_row_threshold)
length(na_rows) # None of the rows has more than 30% of NAs so we need to impute

dput(names(data)) # to get the column names from the data

# Subsetting the data
# since data has dicrete and continuous variables
disc_cols <- c("land", "logged_in",  "root_shell", "su_attempted", "is_host_login", "is_guest_login")
target_col <- "Target"
data_disc<-subset(data,select=disc_cols)
data_cont<-data[!(colnames(data)%in%c(disc_cols,target_col))]

# NAs imputation
repalceNAsWithMean_continuous <- function(x) {replace(x, is.na(x), mean(x[!is.na(x)]))}
repalceNAsWithMean_discrete <- function(x) {replace(x, is.na(x), ifelse(mean(x[!is.na(x)]) < 0.5,0,1) )}

data_disc  <- apply(data_disc,2,repalceNAsWithMean_discrete)
data_cont  <- apply(data_cont,2,repalceNAsWithMean_continuous)

data <- as.data.frame(cbind(data_disc,data_cont,data[,27]))
colnames(data)[27] <- "Target"

# check balance of Target variable
print(table(data$Target))
# class imbalance has to be handled !
prop.table(table(data$Target)) # there is a very less proportion(<2%) of 1's in the train set.

# SMOTE more 1 cases
# SMOTE requires the target to be as a factor.
data$Target <- as.factor(data$Target)
data_smote <- SMOTE(Target ~ ., data, perc.over = 100, perc.under=200)

prop.table(table(data_smote$Target)) # The proportion of classes are equal now & diemnsionality reduced as well
print(table(data_smote$Target))

# Model Building using CARET package
# Using Bagging of Trees (CART), method='treebag' to reduce variance and improve accuracy/recall
ctrl <- trainControl(method = "cv", number = 5)   # set control, using Cross Validation
model_smote <- train(Target ~ ., data = data_smote, method = "treebag",
                       trControl = ctrl)

predictors <- names(data_smote)[names(data_smote) != 'Target']
pred <- predict(model_smote$finalModel, test[,predictors])

# model_smote: Bagged CART 
# Resampling: Cross-Validated (5 fold) 

#Find accuracy & recall on Test data
ConfMat = table(actual=test$Target,pred) 
ConfMat
#         pred
#actual      0      1
#0      132666  11777
#1           7   2550
accuracy_test = sum(diag(ConfMat))/sum(ConfMat) 
recall_test = (ConfMat[2,2]/(ConfMat[2,2]+ConfMat[2,1])) 
accuracy_test  # 0.9198367
recall_test    # 0.9972624


## Model Submission 

# Predict on eval set with above model
eval_pred <- predict(model_smote$finalModel, eval[,predictors])
#eval_predictions <- ifelse(eval_pred==2,1,0)
submit <- data.frame("Predictions"=eval_pred)
str(submit)
write.csv(submit, "./submission_test.csv", row.names=F)
