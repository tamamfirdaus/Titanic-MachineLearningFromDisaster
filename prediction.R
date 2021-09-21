# Set directory
setwd("~/Documents/R/Titanic-MachineLearningFromDisaster")

# Load data
titanic.train <- read.csv(file = "data/train.csv", stringsAsFactors = FALSE, header = TRUE)  
titanic.test <- read.csv(file = "data/test.csv", stringsAsFactors = FALSE, header = TRUE)  

# Add column to table
titanic.train$IsTrainSet <- TRUE
titanic.test$IsTrainSet <- FALSE
titanic.test$Survived <- NA

# Combine data
titanic.full <- rbind(titanic.train, titanic.test)

# Clean data using median
# titanic.full[titanic.full$Embarked=='', "Embarked"] <- 'S'
#
# age.median <- median(titanic.full$Age, na.rm = TRUE)
# titanic.full[is.na(titanic.full$Age),"Age"] <- age.median
#
# fare.median <- median(titanic.full$Fare, na.rm = TRUE)
# titanic.full[is.na(titanic.full$Fare),"Fare"] <- fare.median

# Clean data using median
titanic.full[titanic.full$Embarked=='', "Embarked"] <- 'S'

age.median <- median(titanic.full$Age, na.rm = TRUE)
titanic.full[is.na(titanic.full$Age),"Age"] <- age.median

fare.median <- median(titanic.full$Fare, na.rm = TRUE)
titanic.full[is.na(titanic.full$Fare),"Fare"] <- fare.median

upper.whisker <- boxplot.stats(titanic.full$Fare)$stats[5]
outlier.filter <- titanic.full$Fare < upper.whisker
titanic.full[outlier.filter,]

fare.equation = "Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked"
fare.model <- lm(
  formula = fare.equation,
  data = titanic.full[outlier.filter,]
)

fare.row <- titanic.full[
  is.na(titanic.full$Fare),
  c("Pclass", "Sex", "Age", "SibSp", "Parch","Embarked")
]

fare.predictions <- predict(fare.model, newdata = fare.row)
titanic.full[is.na(titanic.full$Fare),"Fare"] <- fare.predictions

# Casting 
titanic.full$Pclass <- as.factor(titanic.full$Pclass)
titanic.full$Sex <- as.factor(titanic.full$Sex)
titanic.full$Embarked <- as.factor(titanic.full$Embarked)

# Split data back to train and test
titanic.train <- titanic.full[titanic.full$IsTrainSet==TRUE,]
titanic.test <- titanic.full[titanic.full$IsTrainSet==FALSE,]

# Casting
titanic.train$Survived <- as.factor(titanic.train$Survived)

# Install package
install.packages("randomForest")
library(randomForest)


# Create Mode
survived.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
survived.forrmula <- as.formula(survived.equation)
titanic.model <- randomForest(formula = survived.forrmula, data = titanic.train, 
                              ntree = 500,mtry = 3, nodesize = 0.01 * nrow(titanic.test))

# Prediction
features.equation <- "Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
Survived <- predict(titanic.model, newdata = titanic.test)

# Create output
PassengerId <- titanic.test$PassengerId
output.df <- as.data.frame(PassengerId)
output.df$Survived <- Survived

# Convert to csv
write.csv(output.df, file="kaggle_submission.csv", row.names = FALSE)








