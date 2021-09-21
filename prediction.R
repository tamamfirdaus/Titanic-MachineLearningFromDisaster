# Set directory
setwd("~/Documents/R/Titanic-MachineLearningFromDisaster")

########## Data Preparation ##########

# Load data
titanic.train <- read.csv(file = "data/train.csv", stringsAsFactors = FALSE, header = TRUE)  
titanic.test <- read.csv(file = "data/test.csv", stringsAsFactors = FALSE, header = TRUE)  

# Add column to table
titanic.train$IsTrainSet <- TRUE
titanic.test$IsTrainSet <- FALSE
titanic.test$Survived <- NA

# Combine data
titanic.full <- rbind(titanic.train, titanic.test)

########## Data Pre-processing ##########

# Check missing value
colSums(is.na(titanic.full))

# Fix missing data of Embarked with most embarked value
titanic.full[titanic.full$Embarked=='', "Embarked"] <- 'S'

# Fix missing data of Fare with average
fare.mean <- round(mean(titanic.full$Fare, na.rm = TRUE),2)
titanic.full[is.na(titanic.full$Fare),"Fare"] <- fare.mean

# Fix missing data of Age

# age.mean.mrs <- round(mean(titanic.full[(grepl("Mrs\\.",titanic.full$Name)),"Age"], na.rm = TRUE))
# age.mean.miss <- round(mean(titanic.full[(grepl("Miss",titanic.full$Name)),"Age"], na.rm = TRUE))
# age.mean.mr <- round(mean(titanic.full[(grepl("Mr\\.",titanic.full$Name)),"Age"], na.rm = TRUE))
# age.mean.master <- round(mean(titanic.full[(grepl("Master",titanic.full$Name)),"Age"], na.rm = TRUE))

# Fix missing data of Fare using linear regression
upper.whisker <- boxplot.stats(titanic.full$Age)$stats[5]
outlier.filter <- titanic.full$Fare < upper.whisker

age.equation = "Age ~ Pclass + Sex + SibSp + Parch + Embarked"
age.model <- lm(
  formula = age.equation,
  data = titanic.full[outlier.filter,]
)

age.row <- titanic.full[
  is.na(titanic.full$Age),
  c("Pclass", "Sex", "SibSp", "Parch","Embarked")
]

age.predictions <- predict(age.model, newdata = age.row)
titanic.full[is.na(titanic.full$Age),"Age"] <- age.predictions

# Casting features
titanic.full$Pclass <- as.factor(titanic.full$Pclass)
titanic.full$Sex <- as.factor(titanic.full$Sex)
titanic.full$Embarked <- as.factor(titanic.full$Embarked)

# Split data back to train and test
titanic.train <- titanic.full[titanic.full$IsTrainSet==TRUE,]
titanic.test <- titanic.full[titanic.full$IsTrainSet==FALSE,]

# Casting survived feature
titanic.train$Survived <- as.factor(titanic.train$Survived)

########## Machine Learning ##########

# Install package
install.packages("randomForest")
library(randomForest)

# Create ML Model using Random Forest
survived.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
survived.formula <- as.formula(survived.equation)
titanic.model <- randomForest(formula = survived.formula, data = titanic.train, 
                              ntree = 500, mtry = 3, nodesize = 0.01 * nrow(titanic.test))

# Make prediction
features.equation <- "Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
Survived <- predict(titanic.model, newdata = titanic.test)

# Create output
PassengerId <- titanic.test$PassengerId
output.df <- as.data.frame(PassengerId)
output.df$Survived <- Survived

# Convert to csv
write.csv(output.df, file="kaggle_submission.csv", row.names = FALSE)








