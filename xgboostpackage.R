rm(list=ls())
setwd("D:/Documents/FYP/model/xgboost/xgboost using package")
set.seed(1502)
library(caret)
library(doParallel)
library(xgboost)
library(pROC)
library(readr)
# library(SHAPforxgboost)
library(dplyr)

library(readr)
traindata <- read_csv("normalizedtraindatasetelderly.csv")
dim(traindata) #3014   56

testdata <- read_csv("normalizedtestdatasetelderly.csv")
dim(testdata) #1291   59

traindata <- traindata[,2:56]
testdata<- subset(testdata, select = -c(acsstratum,timiscorestemi,timiscorenstemi)) #remove these variables
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


#set categorical data to factor for train dataset
traindata[,names] <- lapply(traindata[,names] , as.numeric)
traindata$status <- ifelse(test = traindata$status ==1, yes = "1", no = "0")
traindata$status <- as.numeric(traindata$status)
str(traindata)

#set categorical data to factor for test dataset
testdata[,names] <- lapply(testdata[,names] , as.numeric)
testdata$status <- ifelse(test = testdata$status ==1, yes = "1", no = "0")
testdata$status <- as.numeric(testdata$status) 
str(testdata)

#target variable
library(dplyr)
ytrain = traindata$status
xtrain = data.matrix(traindata %>% select(-status)) #remove status
str(xtrain)

ytest = testdata$status
xtest = data.matrix(testdata %>% select(-status))

label=as.numeric(as.character(traindata$status))
ts_label=as.numeric(as.character(testdata$status))

sumwpos=sum(label==1)
sumwneg=sum(label==0)
#print(sumwneg/sumwpos)

library(Matrix)
dtest=sparse.model.matrix(status~.-1, data = data.frame(testdata))
dtrain=sparse.model.matrix(status~.-1, traindata)

xgb.grid=expand.grid(nrounds=c(25,50,100), 
                     eta=c(0.01,0.05),
                     max_depth=3,
                     gamma = 0, 
                     subsample = c(0.7,0.8,0.9),
                     min_child_weight = c(1, 2), 
                     colsample_bytree = 1)

myCl=makeCluster(detectCores()-1)
registerDoParallel(myCl)

xgb.control=trainControl(method = "cv",
                         number = 5,
                         verboseIter = TRUE,
                         returnData = FALSE,
                         returnResamp = "none",
                         classProbs = TRUE,
                         allowParallel = TRUE)

xgb.train = train(x = traindata,
                  y = factor(label, labels = c("Dead", "Alive")),
                  trControl = xgb.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree",
                  scale_pos_weight=sumwneg/sumwpos)


stopCluster(myCl)


xgb.train$bestTune
# nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
# 1      25         3 0.01     0                1                1       0.7

params=list("eta"=xgb.train$bestTune$eta,
            "max_depth"=xgb.train$bestTune$max_depth,
            "gamma"=xgb.train$bestTune$gamma,
            "min_child_weight"=xgb.train$bestTune$min_child_weight,
            "nthread"=4,
            "objective"="binary:logistic")

label <- as.numeric(label)

xgb.crv=xgb.cv(params = params,
               data = dtrain,
               nrounds = 5000,
               nfold = 10,
               label = label,
               showsd = TRUE,
               metrics = "auc",
               stratified = TRUE,
               verbose = TRUE,
               print_every_n = 1L,
               early_stopping_rounds = 100,
               scale_pos_weight=sumwneg/sumwpos)

xgb.crv$best_iteration
# 4989
xgb.mod=xgboost(data = dtrain, 
                label = label, 
                max.depth=xgb.train$bestTune$max_depth, 
                eta=xgb.train$bestTune$eta, 
                nthread=4, 
                min_child_weight=xgb.train$bestTune$min_child_weight,
                scale_pos_weight=sumwneg/sumwpos, 
                eval_metric="auc", 
                nrounds=xgb.crv$best_iteration, 
                objective="binary:logistic")

saveRDS(xgb.mod,"xgbtuned.rds")

importance=xgb.importance(feature_names = colnames(dtrain), model = xgb.mod)
importance$Feature[1:42]
(gg=xgb.ggplot.importance(importance_matrix = importance[1:42]))


result.predicted1 <- predict(xgb.mod, newdata = dtest, type = "prob")

result.roc1 <- roc(ts_label, result.predicted1)
result.roc1
# Area under the curve: 0.8394

result.predicted1 <- ifelse (result.predicted1 > 0.5,1,0)
result.predicted1 <- as.factor(result.predicted1)

cm_all_1 <- confusionMatrix (result.predicted1,
                             as.factor(ts_label))
cm_all_1

########################################################################################################################################
#####- FUNCTIONS FOR SHAP ANALYSIS -########
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
# shap_result = shap.score.rank(xgb_model = xgb1, 
#                               X_train =xtrain,
#                               shap_approx = F)
# 
# var_importance(shap_result, top_n=10)
# 
# #SHAP summary
# shap_long = shap.prep(shap = shap_result,
#                       X_train = xtrain, 
#                       top_n = 10)
# 
# plot.shap.summary(data_long = shap_long)
# 
# #SHAP individual plot
# xgb.plot.shap(data = xtrain, # input data
#               model = xgb1, # xgboost model
#               features = names(shap_result$mean_shap_score[1:10]), # only top 10 var
#               n_col = 3, # layout option
#               plot_loess = T # add red line to plot
# )

#########################################################################################################################################
# Boruta


# library(Boruta)
# boruta = Boruta(status ~ ., data = traindata, doTrace = 2, maxRuns = 500)
# 
# # After 13 iterations, +42 secs: 
# #   confirmed 54 attributes: acei, antiarr, arb, asa, bb and 49 more;
# # no more attributes left.
# 
# xgb.boruta=Boruta(xtrain,
#                   y=as.numeric(ytrain),
#                   maxRuns=500, 
#                   doTrace=2,
#                   holdHistory=TRUE,
#                   getImp=getImpXgboost,
#                   nthread=4,
#                   eval_metric="auc", 
#                   eval_metric="rmse", 
#                   eval_metric="logloss",
#                   objective="binary:logistic",
#                   tree_method="hist",
#                   lambda=0,
#                   alpha=0,
#                   max.depth=6,
#                   eta=0.5, 
#                   min_child_weight=1,
#                   gamma=1.5,
#                   nrounds= 1000 )
# 
# boruta_dec=attStats(xgb.boruta)
# boruta_dec[boruta_dec$decision!="Rejected",]

########################################### testing with stemi################## ###########################

model = readRDS("xgbtuned.rds")

#Rows: 592 Columns: 58
stemi <- read_csv("normalizedtestdatasetelderly - stemi.csv")

stemi<- subset(stemi, select = -c(acsstratum,timiscorestemi)) #remove these variables
stemi <- stemi[,2:56]

stemi <- stemi %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

stemi

#set categorical data to factor for train dataset
stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.factor(stemi$status)
str(stemi)
# 0   1 
# 67 525
ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

############### different method to plot roc #####################
stemipred <- predict (model,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
# Area under the curve: 0.6784

#############################################################
#predict
xgbooststemi = xgb.DMatrix(data=xstemi, label=ystemi)
predictionstemi = predict(model, newdata = xgbooststemi )
predictionstemi = ifelse(predictionstemi > 0.5, 1, 0)
predictionstemi
table(predictionstemi)

####### confusion matrix ##########
predictionstemi = as.factor(predictionstemi) 
cmstemi = confusionMatrix(predictionstemi, as.factor(ystemi), positive = '0')
cmstemi
# Accuracy : 0.8784 


############################################# testing with nstemi ##############################################

#Rows: 699 Columns: 58
nstemi <- read_csv("normalizedtestdatasetelderly - nstemi.csv")

nstemi<- subset(nstemi, select = -c(acsstratum,timiscorenstemi)) #remove these variables
nstemi <- nstemi[,2:56]
str(nstemi)

nstemi <- nstemi%>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

nstemi

#set categorical data to factor for train dataset
nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.factor(nstemi$status)
str(nstemi)
table(nstemi$status)
# 0   1 
# 39 660 

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) #remove status

############### different method to plot roc #####################
nstemipred <- predict (model,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
# Area under the curve: 0.7151
#############################################

xgboostnstemi = xgb.DMatrix(data=xnstemi, label=ynstemi)

#predict
predictionnstemi = predict(model, newdata = xgboostnstemi )
predictionnstemi = ifelse(predictionnstemi > 0.5, 1, 0)
predictionnstemi
table(predictionnstemi)

####### confusion matrix ##########
predictionnstemi = as.factor(predictionnstemi) 
cmnstemi = confusionMatrix(predictionnstemi, as.factor(ynstemi), positive = '0')
cmnstemi
# Accuracy : 0.9285 
########roc ##############
predictionnstemi = as.numeric(predictionnstemi) 
rocnstemi=roc(ynstemi,predictionnstemi,levels = c(0, 1), direction = "<") 
plot(rocnstemi, col="red", lwd=3, main="ROC curve")
auc(rocnstemi)
#Area under the curve: 0.7171


