rm(list=ls()) 
setwd("D:/Documents/FYP/model/random forest/random forest zscore using normalized data")

#load library
library(randomForest)
library(ggplot2)
library(cowplot) #improve ggplot2 settings
library(pROC)
library(ROCR)
library(caret)
library(readr)
library(dplyr)

############ ALL VAR ###########
traindata <- read_csv("normalizedtraindatasetelderly.csv")
dim(traindata) #3014   56

testdata <- read_csv("normalizedtestdatasetelderly.csv")
testdata<- subset(testdata, select = -c(acsstratum,timiscorestemi,timiscorenstemi))
dim(testdata) #1291   56

traindata <- traindata[,2:56]
testdata <- testdata[,2:56]

traindata <- traindata %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

traindata


#convert status to 0 and 1 test dataset
testdata <- testdata %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

testdata



names <- c("ptsex", "ptrace","smokingstatus","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk","canginapast2wk","cheartfail","clung","crenal","ccerebrovascular",
           "cpvascular", "ecgabnormtypestelev1","ecgabnormtypestelev2" ,"ecgabnormtypestdep","ecgabnormtypetwave","ecgabnormtypebbb","ecgabnormlocationil",
           "ecgabnormlocational" , "ecgabnormlocationll","ecgabnormlocationtp","ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin",
           "lmwh","bb","acei","arb","statin","lipidla","diuretic", "calcantagonist","oralhypogly","insulin","antiarr")



#change some categorical to 0 and 1
traindata$ptrace <- as.numeric(traindata$ptrace)-1
traindata$smokingstatus <- as.numeric(traindata$smokingstatus)-1
traindata$killipclass<- as.numeric(traindata$killipclass)-1

testdata$ptrace <- as.numeric(testdata$ptrace)-1
testdata$smokingstatus <- as.numeric(testdata$smokingstatus)-1
testdata$killipclass<- as.numeric(testdata$killipclass)-1

traindata[,names] <- lapply(traindata[,names] , factor)
traindata$status <- ifelse(test = traindata$status ==1, yes = "Survived", no = "Died")
traindata$status <- as.factor(traindata$status)

#set categorical data to factor for test dataset
testdata[,names] <- lapply(testdata[,names] , factor)
testdata$status <- ifelse(test = testdata$status ==1, yes = "Survived", no = "Died")
testdata$status <- as.factor(testdata$status)

#running optimized parameters 

library(randomForest)
set.seed(247)
ctrl <- trainControl (method = "cv",number = 10, classProbs = T,savePrediction = T, summaryFunction=twoClassSummary)

#Train the model(original dataset)
set.seed(19) 
modelrfop <- train(status~.,
                   data = traindata,
                   method = "rf",
                   ntree = 600,
                   maxnode = 6,
                   trControl = ctrl, metric = "ROC",
                   tuneLength =10)
modelrfop
attributes(modelrfop)

#positive class -> died
#Confusion Matrix
cm <- confusionMatrix (predict (modelrfop, newdata = testdata),
                       as.factor(testdata$status))
cm

# Confusion Matrix and Statistics
# 
# Reference
# Prediction Died Survived
# Died       55      108
# Survived   51     1077
# 
# Accuracy : 0.8768          
# 95% CI : (0.8577, 0.8943)
# No Information Rate : 0.9179          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3436          
# 
# Mcnemar's Test P-Value : 8.95e-06        
#                                           
#             Sensitivity : 0.51887         
#             Specificity : 0.90886         
#          Pos Pred Value : 0.33742         
#          Neg Pred Value : 0.95479         
#              Prevalence : 0.08211         
#          Detection Rate : 0.04260         
#    Detection Prevalence : 0.12626         
#       Balanced Accuracy : 0.71386         
#                                           
#        'Positive' Class : Died  

#Prediction
pred2 <- predict(modelrfop, newdata = testdata, type = "prob")
#write.csv(result.predicted1,"rf_selected_probability.csv")

pred2$Died = ifelse(pred2$Died > 0.5, 0, 1)

#Plot ROC ---- Nonsurvival
roc2 <- roc(testdata$status, pred2$Died,
            positive = "Died", type = "prob")

plot(roc2, print.thres = "best", print.thres.best.method = "closest.topleft")
roc2
# Area under the curve: 0.8309
ci.auc(as.vector(testdata$status), as.vector(pred$Died))

# save the model
saveRDS(modelrfop,"rfmodeloptimizeparameters.rds")
readRDS("rfmodeloptimizeparameters.rds")

modelrfop = readRDS("rfmodeloptimizeparameters.rds")



#testing stemi & nstemi 


stemi <- read_csv("normalizedtestdatasetelderly - stemi.csv")
nstemi <- read_csv("normalizedtestdatasetelderly - nstemi.csv")


stemi <- stemi[,-1]
nstemi <- nstemi[,-1]

stemi<- subset(stemi, select = -c(acsstratum,timiscorestemi ))
nstemi<- subset(nstemi, select = -c(acsstratum,timiscorenstemi ))

stemi <- stemi %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

stemi


#convert status to 0 and 1 test dataset
nstemi <- nstemi%>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

nstemi

stemi$ptrace <- as.numeric(stemi$ptrace)-1
stemi$smokingstatus <- as.numeric(stemi$smokingstatus)-1
stemi$killipclass<- as.numeric(stemi$killipclass)-1

nstemi$ptrace <- as.numeric(nstemi$ptrace)-1
nstemi$smokingstatus <- as.numeric(nstemi$smokingstatus)-1
nstemi$killipclass<- as.numeric(nstemi$killipclass)-1

stemi[,names] <- lapply(stemi[,names] , factor)
stemi$status <- ifelse(test = stemi$status ==1, yes = "Survived", no = "Died")
stemi$status <- as.factor(stemi$status)

#set categorical data to factor for test dataset
nstemi[,names] <- lapply(nstemi[,names] , factor)
nstemi$status <- ifelse(test = nstemi$status ==1, yes = "Survived", no = "Died")
nstemi$status <- as.factor(nstemi$status)



predstemi <- predict(modelrfop, newdata = stemi[,-55], type = "prob")
#write.csv(result.predicted1,"rf_selected_probability.csv")

predstemi$Died = ifelse(predstemi$Died > 0.5, 0, 1)

#Plot ROC ---- Nonsurvival
rocstemi <- roc(stemi$status, predstemi$Died,
                positive = "Died", type = "prob")

plot(rocstemi, print.thres = "best", print.thres.best.method = "closest.topleft")
rocstemi
# Area under the curve: 0.7115


prednstemi <- predict(modelrfop, newdata = nstemi[,-55], type = "prob")
#write.csv(result.predicted1,"rf_selected_probability.csv")

prednstemi$Died = ifelse(prednstemi$Died > 0.5, 0, 1)

#Plot ROC ---- Nonsurvival
rocnstemi <- roc(nstemi$status, prednstemi$Died,
                 positive = "Died", type = "prob")

plot(rocnstemi, print.thres = "best", print.thres.best.method = "closest.topleft")
rocnstemi
# Area under the curve: 0.708








######################################    SBE 12################################################################################
traindata <- read_csv("normalizedtraindatasetelderly.csv")
dim(traindata) #3014   56

testdata <- read_csv("normalizedtestdatasetelderly.csv")
testdata<- subset(testdata, select = -c(acsstratum,timiscorestemi,timiscorenstemi))
dim(testdata) #1291   56

traindata <- traindata[,2:56]
testdata <- testdata[,2:56]

traindata <- traindata %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

traindata


#convert status to 0 and 1 test dataset
testdata <- testdata %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

testdata



names <- c("ptsex", "ptrace","smokingstatus","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk","canginapast2wk","cheartfail","clung","crenal","ccerebrovascular",
           "cpvascular", "ecgabnormtypestelev1","ecgabnormtypestelev2" ,"ecgabnormtypestdep","ecgabnormtypetwave","ecgabnormtypebbb","ecgabnormlocationil",
           "ecgabnormlocational" , "ecgabnormlocationll","ecgabnormlocationtp","ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin",
           "lmwh","bb","acei","arb","statin","lipidla","diuretic", "calcantagonist","oralhypogly","insulin","antiarr")



#change some categorical to 0 and 1
# traindata$ptrace <- as.numeric(traindata$ptrace)-1
# traindata$smokingstatus <- as.numeric(traindata$smokingstatus)-1
# traindata$killipclass<- as.numeric(traindata$killipclass)-1
# 
# testdata$ptrace <- as.numeric(testdata$ptrace)-1
# testdata$smokingstatus <- as.numeric(testdata$smokingstatus)-1
# testdata$killipclass<- as.numeric(testdata$killipclass)-1

traindata[,names] <- lapply(traindata[,names] , factor)
traindata$status <- ifelse(test = traindata$status ==1, yes = "Survived", no = "Died")
traindata$status <- as.factor(traindata$status)

#set categorical data to factor for test dataset
testdata[,names] <- lapply(testdata[,names] , factor)
testdata$status <- ifelse(test = testdata$status ==1, yes = "Survived", no = "Died")
testdata$status <- as.factor(testdata$status)

traindata<- subset(traindata, select = c("ptageatnotification","ccap","heartrate","bpsys",
                                         "killipclass","ck","pci",
                                         "lmwh","bb","acei","arb","cardiaccath", "status"))

testdata<- subset(testdata, select = c("ptageatnotification","ccap","heartrate","bpsys",
                                       "killipclass","ck","pci",
                                       "lmwh","bb","acei","arb","cardiaccath", "status"))


########## tuning ##
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

control <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE)
tunegrid <- expand.grid(.mtry=c(1:10), .ntree=c(200, 400, 600))
set.seed(19)


custom <- train(status~.,
                data = traindata,
                method=customRF, metric="Accuracy", tuneGrid=tunegrid, trControl=control)

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were mtry = 5 and ntree = 200.

summary(custom)
plot(custom)

#running optimized parameters

library(randomForest)
set.seed(247)
ctrl <- trainControl (method = "cv",number = 10, classProbs = T,savePrediction = T, summaryFunction=twoClassSummary)

#Train the model(original dataset)
set.seed(19) 
modelrfop <- train(status~.,
                   data = traindata,
                   method = "rf",
                   ntree = 200,
                   maxnode = 6,
                   trControl = ctrl, metric = "ROC",
                   tuneLength =10)
modelrfop
attributes(modelrfop)

#positive class -> died
#Confusion Matrix
cm <- confusionMatrix (predict (modelrfop, newdata = testdata),
                       as.factor(testdata$status))
cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction Died Survived
# Died       68      210
# Survived   38      975
# 
# Accuracy : 0.8079          
# 95% CI : (0.7853, 0.8291)
# No Information Rate : 0.9179          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.267           
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.64151         
#             Specificity : 0.82278         
#          Pos Pred Value : 0.24460         
#          Neg Pred Value : 0.96249         
#              Prevalence : 0.08211         
#          Detection Rate : 0.05267         
#    Detection Prevalence : 0.21534         
#       Balanced Accuracy : 0.73215         
#                                           
#        'Positive' Class : Died

#Prediction
pred2 <- predict(modelrfop, newdata = testdata, type = "prob")
#write.csv(result.predicted1,"rf_selected_probability.csv")

pred2$Died = ifelse(pred2$Died > 0.5, 0, 1)

#Plot ROC ---- Nonsurvival
roc2 <- roc(testdata$status, pred2$Died,
            positive = "Died", type = "prob")

plot(roc2, print.thres = "best", print.thres.best.method = "closest.topleft")
roc2
# Area under the curve: 0.7334
ci.auc(as.vector(testdata$status), as.vector(pred$Died))

# save the model
# saveRDS(modelrfop,"rfsbe12.rds")

modelrfop = readRDS("rfsbe12.rds")

#testing sbe 12 
stemi <- read_csv("normalizedtestdatasetelderly - stemi.csv")
nstemi <- read_csv("normalizedtestdatasetelderly - nstemi.csv")


stemi <- stemi[,-1]
nstemi <- nstemi[,-1]

stemi<- subset(stemi, select = -c(acsstratum,timiscorestemi ))
nstemi<- subset(nstemi, select = -c(acsstratum,timiscorenstemi ))

stemi <- stemi %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

stemi


#convert status to 0 and 1 test dataset
nstemi <- nstemi%>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

nstemi

stemi$ptrace <- as.numeric(stemi$ptrace)-1
stemi$smokingstatus <- as.numeric(stemi$smokingstatus)-1
stemi$killipclass<- as.numeric(stemi$killipclass)-1

nstemi$ptrace <- as.numeric(nstemi$ptrace)-1
nstemi$smokingstatus <- as.numeric(nstemi$smokingstatus)-1
nstemi$killipclass<- as.numeric(nstemi$killipclass)-1

stemi[,names] <- lapply(stemi[,names] , factor)
stemi$status <- ifelse(test = stemi$status ==1, yes = "Survived", no = "Died")
stemi$status <- as.factor(stemi$status)

#set categorical data to factor for test dataset
nstemi[,names] <- lapply(nstemi[,names] , factor)
nstemi$status <- ifelse(test = nstemi$status ==1, yes = "Survived", no = "Died")
nstemi$status <- as.factor(nstemi$status)

stemi<- subset(stemi, select = c("ptageatnotification","ccap","heartrate","bpsys",
                                 "killipclass","ck","pci",
                                 "lmwh","bb","acei","arb","cardiaccath", "status"))

nstemi<- subset(nstemi, select = c("ptageatnotification","ccap","heartrate","bpsys",
                                   "killipclass","ck","pci",
                                   "lmwh","bb","acei","arb","cardiaccath", "status"))



predstemi <- predict(modelrfop, newdata = stemi[,-13], type = "prob")
#write.csv(result.predicted1,"rf_selected_probability.csv")

predstemi$Died = ifelse(predstemi$Died > 0.5, 0, 1)

#Plot ROC ---- Nonsurvival
rocstemi <- roc(stemi$status, predstemi$Died,
                positive = "Died", type = "prob")

plot(rocstemi, print.thres = "best", print.thres.best.method = "closest.topleft")
rocstemi
# Area under the curve: 0.7109
ci.auc(as.vector(stemi$status), as.vector(predstemi$Died))
# 95% CI: 0.6495-0.7723 (DeLong)

prednstemi <- predict(modelrfop, newdata = nstemi[,-13], type = "prob")
#write.csv(result.predicted1,"rf_selected_probability.csv")

prednstemi$Died = ifelse(prednstemi$Died > 0.5, 0, 1)

#Plot ROC ---- Nonsurvival
rocnstemi <- roc(nstemi$status, prednstemi$Died,
                 positive = "Died", type = "prob")

rocnstemi
# Area under the curve: 0.6959
ci.auc(as.vector(nstemi$status), as.vector(prednstemi$Died))
# 95% CI: 0.6156-0.7761 (DeLong












######################################    EMERGENCY  8################################################################################
traindata <- read_csv("normalizedtraindatasetelderly.csv")
dim(traindata) #3014   56

testdata <- read_csv("normalizedtestdatasetelderly.csv")
testdata<- subset(testdata, select = -c(acsstratum,timiscorestemi,timiscorenstemi))
dim(testdata) #1291   56

traindata <- traindata[,2:56]
testdata <- testdata[,2:56]

traindata <- traindata %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

traindata


#convert status to 0 and 1 test dataset
testdata <- testdata %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

testdata



names <- c("ptsex", "ptrace","smokingstatus","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk","canginapast2wk","cheartfail","clung","crenal","ccerebrovascular",
           "cpvascular", "ecgabnormtypestelev1","ecgabnormtypestelev2" ,"ecgabnormtypestdep","ecgabnormtypetwave","ecgabnormtypebbb","ecgabnormlocationil",
           "ecgabnormlocational" , "ecgabnormlocationll","ecgabnormlocationtp","ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin",
           "lmwh","bb","acei","arb","statin","lipidla","diuretic", "calcantagonist","oralhypogly","insulin","antiarr")



#change some categorical to 0 and 1
traindata$ptrace <- as.numeric(traindata$ptrace)-1
traindata$smokingstatus <- as.numeric(traindata$smokingstatus)-1
traindata$killipclass<- as.numeric(traindata$killipclass)-1

testdata$ptrace <- as.numeric(testdata$ptrace)-1
testdata$smokingstatus <- as.numeric(testdata$smokingstatus)-1
testdata$killipclass<- as.numeric(testdata$killipclass)-1

traindata[,names] <- lapply(traindata[,names] , factor)
traindata$status <- ifelse(test = traindata$status ==1, yes = "Survived", no = "Died")
traindata$status <- as.factor(traindata$status)

#set categorical data to factor for test dataset
testdata[,names] <- lapply(testdata[,names] , factor)
testdata$status <- ifelse(test = testdata$status ==1, yes = "Survived", no = "Died")
testdata$status <- as.factor(testdata$status)

traindata<- subset(traindata, select = c("ptageatnotification","ccap","heartrate","bpsys","killipclass",       
                                         "bb", "acei", "arb","status"))

testdata<- subset(testdata, select = c("ptageatnotification","ccap","heartrate","bpsys","killipclass",       
                                       "bb", "acei", "arb","status"))


########## tuning ##
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

control <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE)
tunegrid <- expand.grid(.mtry=c(5:10), .ntree=c(200, 400, 600))
set.seed(19)


custom <- train(status~.,
                data = traindata,
                method=customRF, metric="ROC", tuneGrid=tunegrid, trControl=control)

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were mtry = 5 and ntree = 400.

summary(custom)
plot(custom)

#################### running optimized parameters ######################

library(randomForest)
set.seed(247)
ctrl <- trainControl (method = "cv",number = 10, classProbs = T,savePrediction = T, summaryFunction=twoClassSummary)

#Train the model(original dataset)
set.seed(19) 
modelrfop <- train(status~.,
                   data = traindata,
                   method = "rf",
                   ntree = 400,
                   maxnode = 5,
                   trControl = ctrl, metric = "ROC",
                   tuneLength =10)
modelrfop
attributes(modelrfop)

#positive class -> died
#Confusion Matrix
cm <- confusionMatrix (predict (modelrfop, newdata = testdata),
                       as.factor(testdata$status))
cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction Died Survived
# Died       69      196
# Survived   37      989
# 
# Accuracy : 0.8195          
# 95% CI : (0.7974, 0.8401)
# No Information Rate : 0.9179          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.2885          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.65094         
#             Specificity : 0.83460         
#          Pos Pred Value : 0.26038         
#          Neg Pred Value : 0.96394         
#              Prevalence : 0.08211         
#          Detection Rate : 0.05345         
#    Detection Prevalence : 0.20527         
#       Balanced Accuracy : 0.74277         
#                                           
#        'Positive' Class : Died 

#Prediction
pred2 <- predict(modelrfop, newdata = testdata, type = "prob")
#write.csv(result.predicted1,"rf_selected_probability.csv")

pred2$Died = ifelse(pred2$Died > 0.5, 0, 1)

#Plot ROC ---- Nonsurvival
roc2 <- roc(testdata$status, pred2$Died,
            positive = "Died", type = "prob")

plot(roc2, print.thres = "best", print.thres.best.method = "closest.topleft")
roc2
# Area under the curve: 0.7432
ci.auc(as.vector(testdata$status), as.vector(pred$Died))

# save the model
saveRDS(modelrfop,"rfemergency8.rds")

modelrfop = readRDS("rfemergency8.rds")

############# testing sbe 12 ##########
stemi <- read_csv("normalizedtestdatasetelderly - stemi.csv")
nstemi <- read_csv("normalizedtestdatasetelderly - nstemi.csv")


stemi <- stemi[,-1]
nstemi <- nstemi[,-1]

stemi<- subset(stemi, select = -c(acsstratum,timiscorestemi ))
nstemi<- subset(nstemi, select = -c(acsstratum,timiscorenstemi ))

stemi <- stemi %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

stemi


#convert status to 0 and 1 test dataset
nstemi <- nstemi%>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

nstemi

stemi$ptrace <- as.numeric(stemi$ptrace)-1
stemi$smokingstatus <- as.numeric(stemi$smokingstatus)-1
stemi$killipclass<- as.numeric(stemi$killipclass)-1

nstemi$ptrace <- as.numeric(nstemi$ptrace)-1
nstemi$smokingstatus <- as.numeric(nstemi$smokingstatus)-1
nstemi$killipclass<- as.numeric(nstemi$killipclass)-1

stemi[,names] <- lapply(stemi[,names] , factor)
stemi$status <- ifelse(test = stemi$status ==1, yes = "Survived", no = "Died")
stemi$status <- as.factor(stemi$status)

#set categorical data to factor for test dataset
nstemi[,names] <- lapply(nstemi[,names] , factor)
nstemi$status <- ifelse(test = nstemi$status ==1, yes = "Survived", no = "Died")
nstemi$status <- as.factor(nstemi$status)

stemi<- subset(stemi, select = c("ptageatnotification","ccap","heartrate","bpsys","killipclass",       
                                 "bb", "acei", "arb","status"))

nstemi<- subset(nstemi, select = c("ptageatnotification","ccap","heartrate","bpsys","killipclass",       
                                   "bb", "acei", "arb","status"))

############################################

predstemi <- predict(modelrfop, newdata = stemi[,-9], type = "prob")
#write.csv(result.predicted1,"rf_selected_probability.csv")

predstemi$Died = ifelse(predstemi$Died > 0.5, 0, 1)

#Plot ROC ---- Nonsurvival
rocstemi <- roc(stemi$status, predstemi$Died,
                positive = "Died", type = "prob")

rocstemi
# Area under the curve: 0.7134
ci.auc(as.vector(stemi$status), as.vector(predstemi$Died))
# 95% CI: 0.6526-0.7742 (DeLong)


prednstemi <- predict(modelrfop, newdata = nstemi[,-9], type = "prob")
#write.csv(result.predicted1,"rf_selected_probability.csv")

prednstemi$Died = ifelse(prednstemi$Died > 0.5, 0, 1)

#Plot ROC ---- Nonsurvival
rocnstemi <- roc(nstemi$status, prednstemi$Died,
                 positive = "Died", type = "prob")

plot(rocnstemi, print.thres = "best", print.thres.best.method = "closest.topleft")
rocnstemi
# Area under the curve: 0.778
ci.auc(as.vector(nstemi$status), as.vector(prednstemi$Died))
0.7034 - 0.8525













################################ Training random forest default parameters ################################
# library(randomForest)
# set.seed(222)
# ctrl <- trainControl (method = "cv",number = 10, classProbs = T,savePrediction = T, summaryFunction=twoClassSummary)
# 
# #Train the model(original dataset)
# set.seed(19) 
# model_rf <- train(status~.,
#                   data = traindata,
#                   method = "rf",
#                   trControl = ctrl, metric = "ROC")
# model_rf
# attributes(model_rf)
# 
# #positive class -> died
# #Confusion Matrix
# cm <- confusionMatrix (predict (model_rf, newdata = testdata),
#                              as.factor(testdata$status))
# cm
# 
# #Prediction
# pred <- predict(model_rf, newdata = testdata, type = "prob")
# #write.csv(result.predicted1,"rf_selected_probability.csv")
# 
# #Plot ROC ---- Nonsurvival
# roc <- roc(testdata$status, pred$Died,
#                    positive = "Died", type = "prob")
# 
# plot(roc, print.thres = "best", print.thres.best.method = "closest.topleft")
# roc
# # Area under the curve: 0.8275
# ci.auc(as.vector(testdata$status), as.vector(pred$Died))
# 
# # save the model
# saveRDS(model_rf,"zscorerfmodel1.rds")
# readRDS("zscorerfmodel.rds")

#######################################################################################

# customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
# customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
# customRF$grid <- function(x, y, len = NULL, search = "grid") {}
# customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
#   randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
# }
# customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
#   predict(modelFit, newdata)
# customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
#   predict(modelFit, newdata, type = "prob")
# customRF$sort <- function(x) x[order(x[,1]),]
# customRF$levels <- function(x) x$classes
# 
# control <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE)
# tunegrid <- expand.grid(.mtry=c(5:10), .ntree=c(200, 400, 600))
# set.seed(19)
# 
# 
# custom <- train(status~.,
#                 data = traindata,
#                 method=customRF, metric="ROC", tuneGrid=tunegrid, trControl=control)
# 
# summary(custom)
# plot(custom)

# mtry  ntree  Accuracy   Kappa    
# 5    200    0.9866182  0.9732256
# 5    400    0.9870582  0.9741064
# 5    600    0.9873908  0.9747699
# 6    200    0.9866163  0.9732252
# 6    400    0.9873908  0.9747720
# 6    600    0.9873919  0.9747728
# 7    200    0.9872822  0.9745552
# 7    400    0.9867271  0.9734443
# 7    600    0.9867271  0.9734461
# 8    200    0.9866156  0.9732223
# 8    400    0.9863956  0.9727825
# 8    600    0.9862867  0.9725627
# 9    200    0.9856204  0.9712311
# 9    400    0.9860637  0.9721174
# 9    600    0.9863948  0.9727805
# 10    200    0.9855093  0.9710129
# 10    400    0.9860626  0.9721157
# 10    600    0.9866163  0.9732226
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were mtry = 6 and ntree = 600.

# save the model
# saveRDS(custom,"tunedrfmodel.rds")
# model = readRDS("tunedrfmodel.rds")


#################################parameter tuning############################################
#find best maxnodes
#maxnode = 15
# store_maxnode <- list()
# tuneGrid <- expand.grid(.mtry = best_mtry)
# for (maxnodes in c(5: 15)) {
#   set.seed(1234)
#   rf_maxnode <- train(status~.,
#                       data = traindata,
#                       method = "rf",
#                       metric = "Accuracy",
#                       trControl = trControl,
#                       importance = TRUE,
#                       nodesize = 14,
#                       maxnodes = maxnodes,
#                       ntree = 600)
#   current_iteration <- toString(maxnodes)
#   store_maxnode[[current_iteration]] <- rf_maxnode
# }
# results_mtry <- resamples(store_maxnode)
# summary(results_mtry)
# 
# 
# #find best mtry = 4
# fitControl <- trainControl(## 10-fold CV
#   method = "repeatedcv",
#   number = 8,
#   ## repeated ten times
#   repeats = 10)
# 
# sqtmtry<- round(sqrt(ncol(traindata) - 1))
# rfGrid <- expand.grid(mtry = c(round(sqtmtry / 2), sqtmtry, 2 * sqtmtry))
# 
# 
# tunegrid <- expand.grid(.mtry=c(4:15))
# tunegrid <- expand.grid(.ntree=c(500,700,1000,1500))
# rf_gridsearch <- train(status~., data=traindata, method="rf", tuneGrid=tunegrid, trControl=fitControl)
# print(rf_gridsearch)
# plot(rf_gridsearch)
# 
# #to find best ntrees
# #best tree = 600
# library(randomForest)
# library(caret)
# library(e1071)
# 
# # Define the control
# trControl <- trainControl(method = "cv",
#                           number = 10,
#                           search = "grid",classProbs = TRUE)
# 
# store_maxtrees <- list()
# for (ntree in c(200, 600, 800, 1000, 1200)) {
#   set.seed(5678)
#   rf_maxtrees <- train(status~.,
#                        data = traindata,
#                        method = "rf",
#                        metric = "ROC",
#                        tuneGrid = NULL,
#                        trControl = trControl,
#                        importance = TRUE,
#                        nodesize = 14,
#                        maxnodes = 24,
#                        ntree = ntree)
#   key <- toString(ntree)
#   store_maxtrees[[key]] <- rf_maxtrees
# }
# results_tree <- resamples(store_maxtrees)
# summary(results_tree)

#################################Variables Importance###########################

# model_rf = readRDS("zscorerfmodel.rds")
# rf_importance <- varImp(model_rf)
# rf_importance
# write.csv(rf_importance[[1]], "Variables_Important_RF.csv")
# 
# 
# # To get the area under the ROC curve for each predictor
# roc_imp <- filterVarImp(x = traindata[, -ncol(testdata)], y =traindata$status)
# roc_imp
# head(roc_imp)   # first 5
# 
# # plot variable importace 
# plot(rf_importance, top = 54) 












############################################# shap ####################################################33
# 
# shap.score.rank <- function(xgb_model = xgb_mod, shap_approx = TRUE, 
#                             X_train = mydata$train_mm){
#   require(xgboost)
#   require(data.table)
#   shap_contrib <- predict(xgb_model, X_train,
#                           predcontrib = TRUE, approxcontrib = shap_approx)
#   shap_contrib <- as.data.table(shap_contrib)
#   shap_contrib[,BIAS:=NULL]
#   cat('make SHAP score by decreasing order\n\n')
#   mean_shap_score <- colMeans(abs(shap_contrib))[order(colMeans(abs(shap_contrib)), decreasing = T)]
#   return(list(shap_score = shap_contrib,
#               mean_shap_score = (mean_shap_score)))
# }
# 
# # a function to standardize feature values into same range
# std1 <- function(x){
#   return ((x - min(x, na.rm = T))/(max(x, na.rm = T) - min(x, na.rm = T)))
# }
# 
# 
# # prep shap data
# shap.prep <- function(shap  = shap_result, X_train = mydata$train_mm, top_n){
#   require(ggforce)
#   # descending order
#   if (missing(top_n)) top_n <- dim(X_train)[2] # by default, use all features
#   if (!top_n%in%c(1:dim(X_train)[2])) stop('supply correct top_n')
#   require(data.table)
#   shap_score_sub <- as.data.table(shap$shap_score)
#   shap_score_sub <- shap_score_sub[, names(shap$mean_shap_score)[1:top_n], with = F]
#   shap_score_long <- melt.data.table(shap_score_sub, measure.vars = colnames(shap_score_sub))
#   
#   # feature values: the values in the original dataset
#   fv_sub <- as.data.table(X_train)[, names(shap$mean_shap_score)[1:top_n], with = F]
#   # standardize feature values
#   fv_sub_long <- melt.data.table(fv_sub, measure.vars = colnames(fv_sub))
#   fv_sub_long[, stdfvalue := std1(value), by = "variable"]
#   # SHAP value: value
#   # raw feature value: rfvalue; 
#   # standarized: stdfvalue
#   names(fv_sub_long) <- c("variable", "rfvalue", "stdfvalue" )
#   shap_long2 <- cbind(shap_score_long, fv_sub_long[,c('rfvalue','stdfvalue')])
#   shap_long2[, mean_value := mean(abs(value)), by = variable]
#   setkey(shap_long2, variable)
#   return(shap_long2) 
# }
# 
# plot.shap.summary <- function(data_long){
#   x_bound <- max(abs(data_long$value))
#   require('ggforce') # for `geom_sina`
#   plot1 <- ggplot(data = data_long)+
#     coord_flip() + 
#     # sina plot: 
#     geom_sina(aes(x = variable, y = value, color = stdfvalue)) +
#     # print the mean absolute value: 
#     geom_text(data = unique(data_long[, c("variable", "mean_value"), with = F]),
#               aes(x = variable, y=-Inf, label = sprintf("%.3f", mean_value)),
#               size = 3, alpha = 0.7,
#               hjust = -0.2, 
#               fontface = "bold") + # bold
#     # # add a "SHAP" bar notation
#     # annotate("text", x = -Inf, y = -Inf, vjust = -0.2, hjust = 0, size = 3,
#     #          label = expression(group("|", bar(SHAP), "|"))) + 
#     scale_color_gradient(low="#FFCC33", high="#6600CC", 
#                          breaks=c(0,1), labels=c("Low","High")) +
#     theme_bw() + 
#     theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), # remove axis line
#           legend.position="bottom") + 
#     geom_hline(yintercept = 0) + # the vertical line
#     scale_y_continuous(limits = c(-x_bound, x_bound)) +
#     # reverse the order of features
#     scale_x_discrete(limits = rev(levels(data_long$variable)) 
#     ) + 
#     labs(y = "SHAP value (impact on model output)", x = "", color = "Feature value") 
#   return(plot1)
# }
# 
# var_importance <- function(shap_result, top_n=10)
# {
#   var_importance=tibble(var=names(shap_result$mean_shap_score), importance=shap_result$mean_shap_score)
#   
#   var_importance=var_importance[1:top_n,]
#   
#   ggplot(var_importance, aes(x=reorder(var,importance), y=importance)) + 
#     geom_bar(stat = "identity") + 
#     coord_flip() + 
#     theme_light() + 
#     theme(axis.title.y=element_blank()) 
# }
# ######- END OF FUNCTION -######
# 
# #SHAP importance
# shap_result = shap.score.rank(xgb_model = modelrfop, 
#                               X_train =traindata,
#                               shap_approx = F)
# 
# var_importance(shap_result, top_n=10)
# 
# #SHAP summary
# shap_long = shap.prep(shap = shap_result,
#                       X_train = traindata, 
#                       top_n = 10)
# 
# plot.shap.summary(data_long = shap_long)
# 
# #SHAP individual plot
# xgb.plot.shap(data = traindata, # input data
#               model = xgb1, # xgboost model
#               features = names(shap_result$mean_shap_score[1:10]), # only top 10 var
#               n_col = 3, # layout option
#               plot_loess = T # add red line to plot
# )
  
  
  
  
  
  