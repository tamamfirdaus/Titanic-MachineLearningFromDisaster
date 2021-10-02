########## Data Preparation ##########

# Set directory
setwd("~/Documents/R/Titanic-MachineLearningFromDisaster")

# Load data
train_df <- read.csv(file = "data/train.csv", na.strings = c("", "NA"), stringsAsFactors = FALSE, header = TRUE)  
test_df <- read.csv(file = "data/test.csv", na.strings = c("", "NA"), stringsAsFactors = FALSE, header = TRUE)  

########## Data Exploration ##########

# Get information
head(train_df)
str(train_df)
summary(train_df)
colSums(is.na(train_df))

head(test_df)
str(test_df)
summary(test_df)
colSums(is.na(test_df))

# Fix missing Embarked data in train_df with common value 
sort(table(train_df$Embarked), decreasing = TRUE)
train_df[is.na(train_df$Embarked),"Embarked"] <- 'S'
         
# Fix missing Fare data in test_df with mean value
fare_mean <- round(mean(c(train_df$Fare,test_df$Fare), na.rm = TRUE),2)
test_df[is.na(test_df$Fare),"Fare"] <- fare_mean

# Fix missing Age data with mean value
age_mean <- round(mean(c(train_df$Age,test_df$Age), na.rm = TRUE),2)
train_df[is.na(train_df$Age),"Age"] <- age_mean
test_df[is.na(test_df$Age),"Age"] <- age_mean

# Drop Cabin data because it has many missing data
train_df <- subset(train_df, select = -c(Cabin))
test_df <- subset(test_df, select = -c(Cabin))

# Converting fatures
train_df$Survived <- as.factor(train_df$Survived)

train_df$Pclass <- as.factor(train_df$Pclass)
test_df$Pclass <- as.factor(test_df$Pclass)

train_df$Sex <- as.factor(train_df$Sex)
test_df$Sex <- as.factor(test_df$Sex)

train_df$Fare <- as.integer(train_df$Fare)
test_df$Fare <- as.integer(test_df$Fare)

train_df$Embarked <- as.factor(train_df$Embarked)
test_df$Embarked <- as.factor(test_df$Embarked)

# Make new feature
# group_age
train_df$GroupAge <- cut(train_df$Age,
                         breaks = c(0,10,20,30,40,50,100),
                         labels = c(1,2,3,4,5,6))

test_df$GroupAge <- cut(test_df$Age,
                         breaks = c(0,10,20,30,40,50,100),
                         labels = c(1,2,3,4,5,6))

train_df$GroupAge <- as.factor(train_df$GroupAge)
test_df$GroupAge <- as.factor(test_df$GroupAge)

# Drop unused feature
PassengerId <- test_df$PassengerId
train_df <- subset(train_df, select = -c(PassengerId,Name,Age,Ticket))
test_df <- subset(test_df, select = -c(PassengerId,Name,Age,Ticket))

########## Machine Learning ##########
summary(train_df)
str(train_df)

# Uncomment to install package
# install.packages("randomForest")
library(randomForest)

# Create ML Model using Random Forest
formula <- as.formula("Survived ~ .")
model <- randomForest(formula = formula,
                      data = train_df,
                      ntree = 500,
                      mtry = 3,
                      nodesize = 0.01*nrow(test_df))

# Confusion Matrix
# install.packages('caret', dependencies = TRUE)
library(caret)
prediction <- predict(model, newdata = train_df)
confusionMatrix(prediction, train_df$Survived)

# Make prediction
Survived <- predict(model, newdata = test_df)

# Create output in csv 
output_df <- as.data.frame(PassengerId)
output_df$Survived <- Survived
write.csv(output_df, file="kaggle_submission.csv", row.names = FALSE)

