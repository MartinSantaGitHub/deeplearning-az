library(unbalanced)

data = read.csv("deeplearning-az/datasets/Part 1 - Artificial Neural Networks (ANN)/Churn_Modelling.csv")
View(data)

output <- data$Exited
output <- as.factor(output)

output

n <- ncol(data)

input <- data[,-n]
View(input)

data_b <- ubBalance(X = input, Y = output, type="ubSMOTE", percOver=300, percUnder=150, verbose=TRUE)

balancedData <- cbind(data_b$X, data_b$Y)
View(balancedData)

colnames(balancedData)[14] <- "Exited"

prop.table(table(data$Exited))
prop.table(table(balancedData$Exited))

write.csv(balancedData,"deeplearning-az/codigo/01 - Artificial Neural Networks (ANN)/Churn_Modelling_Balanced.csv")

data2 = read.csv("deeplearning-az/codigo/01 - Artificial Neural Networks (ANN)/Churn_Modelling_Balanced.csv")
View(data2)
