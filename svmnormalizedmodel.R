rm(list=ls()) 

library(caret)
library(doParallel)
library(xgboost)
library(pROC)
library(dplyr)
setwd("~/FYP/model/SVM/svm normalized during training")

library(readr)
traindata <- read_csv("balancetraindatasetelderly.csv")
dim(traindata)
table(traindata$status)

testdata <- read_csv("testdatasetelderly.csv")
dim(testdata)
table(testdata$status)

traindata <- traindata[,2:56]
testdata<- subset(testdata, select = -c(acsstratum,timiscorestemi,timiscorenstemi)) #remove these variables
testdata <- testdata[,2:56]

names <- c("ptsex","ptrace","smokingstatus","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk","canginapast2wk","cheartfail","clung","crenal","ccerebrovascular",
           "cpvascular","killipclass", "ecgabnormtypestelev1","ecgabnormtypestelev2" ,"ecgabnormtypestdep","ecgabnormtypetwave","ecgabnormtypebbb","ecgabnormlocationil", 
           "ecgabnormlocational" , "ecgabnormlocationll","ecgabnormlocationtp","ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin",
           "lmwh","bb","acei","arb","statin","lipidla","diuretic", "calcantagonist","oralhypogly","insulin","antiarr","status")


table(traindata$status) #1565 1449 (1 = alive,2 = dead)
table(testdata$status) # 1185 106

#set categorical data to factor for train dataset
traindata[,names] <- lapply(traindata[,names] , as.numeric)
str(traindata)
traindata$status <- ifelse(test = traindata$status ==1, yes = "Survived", no = "Died")
traindata$status <- as.factor(traindata$status)

#set categorical data to factor for test dataset
testdata[,names] <- lapply(testdata[,names] , as.numeric)
str(testdata)
testdata$status <- ifelse(test = testdata$status ==1, yes = "Survived", no = "Died")
testdata$status <- as.factor(testdata$status)


############## SVM ######################
library(caret)
library(doParallel)
registerDoParallel()

set.seed(112)
ctrl <- trainControl (method = "cv",number = 3,classProbs = TRUE,
                      summaryFunction=twoClassSummary)

train <- traindata[-55]
train_target <- traindata$status
set.seed(64)  

#Train SVM

svm.tune <- train(x = train,
                  y = train_target,
                  method = "svmRadial",
                  metric = "ROC",
                  #tuneGrid = grid,
                  preProcess = c("scale","center"),
                  trControl = ctrl)
svm.tune

result.predicted <- predict(svm.tune,newdata = testdata[,-55], type = "prob")
result.roc <- roc(testdata$status,result.predicted$Died, type = "prob")
plot(result.roc,print.thres = "best", print.thres.best.method = "closest.topleft")
result.roc

# save the model
saveRDS(svm.tune,"svmcns.rds")
readRDS("svmcns.rds")

cm <- confusionMatrix (predict(svm.tune, newdata = testdata[,-55]), as.factor(testdata$status))
cm

ci.auc(as.vector(testdata$status), as.vector(result.predicted$Died))


############### testing stemi #####################

model = readRDS("svmcns.rds")


#Rows: 592 Columns: 58
stemi <- read_csv("testdatasetelderly - stemi.csv")

stemi<- subset(stemi, select = -c(acsstratum,timiscorestemi)) #remove these variables
stemi <- stemi[,2:56]
str(stemi)

#set categorical data to factor for train dataset
stemi[,names] <- lapply(stemi[,names] , as.numeric)
stemi$status <- ifelse(test = stemi$status ==1, yes = "Survived", no = "Died")
stemi$status <- as.factor(stemi$status)
str(stemi)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

result.predicted <- predict(svm.tune,newdata = xstemi, type = "prob")
result.roc <- roc(ystemi,result.predicted$Died, type = "prob")
plot(result.roc,print.thres = "best", print.thres.best.method = "closest.topleft")
result.roc
# Area under the curve: 0.794


############# testing with nstemi ################

#Rows: 699 Columns: 58
nstemi <- read_csv("testdatasetelderly - nstemi.csv")

nstemi<- subset(nstemi, select = -c(acsstratum,timiscorenstemi)) #remove these variables
nstemi <- nstemi[,2:56]
str(nstemi)

#set categorical data to factor for train dataset
nstemi[,names] <- lapply(nstemi[,names] , as.numeric)
nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.factor(nstemi$status)
str(nstemi)
table(nstemi$status)

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) #remove status

result.predicted <- predict(svm.tune,newdata = xnstemi, type = "prob")
result.roc <- roc(ynstemi,result.predicted$Died, type = "prob")
plot(result.roc,print.thres = "best", print.thres.best.method = "closest.topleft")
result.roc
# Area under the curve: 0.8291







