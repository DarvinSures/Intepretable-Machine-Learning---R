#xgboost for all variables
rm(list=ls())
library(mlbench)
library(caret)
library(doParallel)
library(tidyverse)
library(xgboost)
library(pROC)
library(SHAPforxgboost)
library(Matrix)
library(data.table)
library(ggplot2)
library(dplyr)

set.seed(1234)

setwd("D:/Documents/FYP/model/xgboost/xgboost complete cases")

library(readxl)
ds <- read_excel("elderlycompletecases.xlsx")
ds <- ds[,-1]
ds<- subset(ds, select = -c(timiscorestemi,timiscorenstemi,acsstratum))

ds <- ds %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

ds

ds$ptrace <- as.numeric(ds$ptrace)-1
ds$smokingstatus <- as.numeric(ds$smokingstatus)-1
ds$killipclass<- as.numeric(ds$killipclass)-1

# ds$status <- ifelse(test = ds$status ==1, yes = "1", no = "0")
ds$status <- as.numeric(ds$status)

train.ind=createDataPartition(ds$status, times = 1, p=0.7, list = FALSE)
training.df=ds[train.ind, ]

#save testing.df into another excel file
testing.df=ds[-train.ind, ]
#write.csv(testing.df,'testing.dfcompletecases.csv')

label=as.numeric(as.character(training.df$status))
ts_label=as.numeric(as.character(testing.df$status))

sumwpos=sum(label==1)
sumwneg=sum(label==0)
#print(sumwneg/sumwpos)

dtest=sparse.model.matrix(status~.-1, data = data.frame(testing.df))
dtrain=sparse.model.matrix(status~.-1, training.df)

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

xgb.train = train(x = training.df,
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
# 871
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

importance=xgb.importance(feature_names = colnames(dtrain), model = xgb.mod)
importance$Feature[1:42]
(gg=xgb.ggplot.importance(importance_matrix = importance[1:42]))


result.predicted1 <- predict(xgb.mod, newdata = dtest, type = "prob")

result.roc1 <- roc(ts_label, result.predicted1)
result.roc1
# Area under the curve: 0.8612

result.predicted1 <- ifelse (result.predicted1 > 0.5,1,0)
result.predicted1 <- as.factor(result.predicted1)

cm_all_1 <- confusionMatrix (result.predicted1,
                             as.factor(ts_label))
cm_all_1
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1
# 0   65  136
# 1   34 1056
# 
# Accuracy : 0.8683          
# 95% CI : (0.8486, 0.8863)
# No Information Rate : 0.9233          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3684          
# 
# Mcnemar's Test P-Value : 9.457e-15       
#                                           
#             Sensitivity : 0.65657         
#             Specificity : 0.88591         
#          Pos Pred Value : 0.32338         
#          Neg Pred Value : 0.96881         
#              Prevalence : 0.07668         
#          Detection Rate : 0.05035         
#    Detection Prevalence : 0.15569         
#       Balanced Accuracy : 0.77124         
#                                           
#        'Positive' Class : 0

############################################################################################################
#testing using stemi and nstemi dataset separated for xgboost complete cases
saveRDS(xgb.mod,"xgbusingcompletecases.rds")
model = readRDS("xgbusingcompletecases.rds")

###########stemi  complete cases##########

stemi <- read_csv("testing.dfcompletecases - stemi.csv")

stemi<- subset(stemi, select = -c(acsstratum,timiscorestemi)) #remove these variables
# stemi <- stemi[,2:55]

stemi <- stemi %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

stemi

stemi$ptrace <- as.numeric(stemi$ptrace)-1
stemi$smokingstatus <- as.numeric(stemi$smokingstatus)-1
stemi$killipclass<- as.numeric(stemi$killipclass)-1

# stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.numeric(stemi$status)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

############### different method to plot roc #####################
stemipred <- predict (model,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
ci.auc(as.vector(ystemi), as.vector(stemipred))
# 95% CI: 0.8038-0.9068
# Area under the curve:  0.7913

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
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0  10   8
# 1  50 495
# 
# Accuracy : 0.897           
# 95% CI : (0.8689, 0.9208)
# No Information Rate : 0.8934          
# P-Value [Acc > NIR] : 0.4256          
# 
# Kappa : 0.2179          
# 
# Mcnemar's Test P-Value : 7.303e-08       
#                                           
#             Sensitivity : 0.16667         
#             Specificity : 0.98410         
#          Pos Pred Value : 0.55556         
#          Neg Pred Value : 0.90826         
#              Prevalence : 0.10657         
#          Detection Rate : 0.01776         
#    Detection Prevalence : 0.03197         
#       Balanced Accuracy : 0.57538         
#                                           
#        'Positive' Class : 0  

##############nstemi complete cases ####################
nstemi <- read_csv("testing.dfcompletecases - nstemi.csv")

nstemi<- subset(nstemi, select = -c(acsstratum,timiscorenstemi)) #remove these variables
# nstemi <- nstemi[,2:56]

nstemi <- nstemi%>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

nstemi

nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.numeric(nstemi$status)

nstemi$ptrace <- as.numeric(nstemi$ptrace)-1
nstemi$smokingstatus <- as.numeric(nstemi$smokingstatus)-1
nstemi$killipclass<- as.numeric(nstemi$killipclass)-1

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) #remove status

############### different method to plot roc #####################
nstemipred <- predict (model,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
ci.auc(as.vector(ynstemi), as.vector(nstemipred))
# Area under the curve: 0.7768

#predict
xgboostnstemi = xgb.DMatrix(data=xnstemi, label=ynstemi)
predictionnstemi = predict(model, newdata = xgboostnstemi )
predictionnstemi = ifelse(predictionnstemi > 0.5, 1, 0)
predictionnstemi
table(predictionnstemi)

####### confusion matrix ##########
predictionnstemi = as.factor(predictionnstemi) 
cmnstemi = confusionMatrix(predictionnstemi, as.factor(ynstemi), positive = '0')
cmnstemi
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0   1   2
# 1  38 687
# 
# Accuracy : 0.9451          
# 95% CI : (0.9259, 0.9605)
# No Information Rate : 0.9464          
# P-Value [Acc > NIR] : 0.6064          
# 
# Kappa : 0.0403          
# 
# Mcnemar's Test P-Value : 3.13e-08        
#                                           
#             Sensitivity : 0.025641        
#             Specificity : 0.997097        
#          Pos Pred Value : 0.333333        
#          Neg Pred Value : 0.947586        
#              Prevalence : 0.053571        
#          Detection Rate : 0.001374        
#    Detection Prevalence : 0.004121        
#       Balanced Accuracy : 0.511369        
#                                           
#        'Positive' Class : 0 

#############################################################################################################
#testing using stemi and nstemi dataset separated previously, not one separated for xgboost complete cases
################ testing stemi ####################
# model = readRDS("xgbusingcompletecases.rds")

stemi <- read_csv("testdatasetelderly - stemi.csv")

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

stemi$ptrace <- as.numeric(stemi$ptrace)-1
stemi$smokingstatus <- as.numeric(stemi$smokingstatus)-1
stemi$killipclass<- as.numeric(stemi$killipclass)-1

stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.numeric(stemi$status)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

############### different method to plot roc #####################
stemipred <- predict (model,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
# Area under the curve: 0.7977

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
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0   4   6
# 1  63 519
# 
# Accuracy : 0.8834          
# 95% CI : (0.8548, 0.9082)
# No Information Rate : 0.8868          
# P-Value [Acc > NIR] : 0.6328          
# 
# Kappa : 0.0768          
# 
# Mcnemar's Test P-Value : 1.566e-11       
#                                           
#             Sensitivity : 0.059701        
#             Specificity : 0.988571        
#          Pos Pred Value : 0.400000        
#          Neg Pred Value : 0.891753        
#              Prevalence : 0.113176        
#          Detection Rate : 0.006757        
#    Detection Prevalence : 0.016892        
#       Balanced Accuracy : 0.524136        
#                                           
#        'Positive' Class : 0  



###################### testing nstemi ##################
# model = readRDS("xgbusingcompletecases.rds")

nstemi <- read_csv("testdatasetelderly - nstemi.csv")

nstemi<- subset(nstemi, select = -c(acsstratum,timiscorenstemi)) #remove these variables
nstemi <- nstemi[,2:56]
nstemi <- nstemi%>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

nstemi

nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.numeric(nstemi$status)

nstemi$ptrace <- as.numeric(nstemi$ptrace)-1
nstemi$smokingstatus <- as.numeric(nstemi$smokingstatus)-1
nstemi$killipclass<- as.numeric(nstemi$killipclass)-1


ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) #remove status

############### different method to plot roc #####################
nstemipred <- predict (model,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
# Area under the curve: 0.8022

#predict
xgboostnstemi = xgb.DMatrix(data=xnstemi, label=ynstemi)
predictionnstemi = predict(model, newdata = xgboostnstemi )
predictionnstemi = ifelse(predictionnstemi > 0.5, 1, 0)
predictionnstemi
table(predictionnstemi)

####### confusion matrix ##########
predictionnstemi = as.factor(predictionnstemi) 
cmnstemi = confusionMatrix(predictionnstemi, as.factor(ynstemi), positive = '0')
cmnstemi
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0   4   2
# 1  35 658
# 
# Accuracy : 0.9471          
# 95% CI : (0.9278, 0.9625)
# No Information Rate : 0.9442          
# P-Value [Acc > NIR] : 0.4114          
# 
# Kappa : 0.1654          
# 
# Mcnemar's Test P-Value : 1.435e-07       
#                                           
#             Sensitivity : 0.102564        
#             Specificity : 0.996970        
#          Pos Pred Value : 0.666667        
#          Neg Pred Value : 0.949495        
#              Prevalence : 0.055794        
#          Detection Rate : 0.005722        
#    Detection Prevalence : 0.008584        
#       Balanced Accuracy : 0.549767        
#                                           
#        'Positive' Class : 0 
  


####################################################################################################################
#####- FUNCTIONS FOR SHAP ANALYSIS -########
shap.score.rank <- function(xgb_model = xgb_mod, shap_approx = TRUE, 
                            X_train = mydata$train_mm){
  require(xgboost)
  require(data.table)
  shap_contrib <- predict(xgb_model, X_train,
                          predcontrib = TRUE, approxcontrib = shap_approx)
  shap_contrib <- as.data.table(shap_contrib)
  shap_contrib[,BIAS:=NULL]
  cat('make SHAP score by decreasing order\n\n')
  mean_shap_score <- colMeans(abs(shap_contrib))[order(colMeans(abs(shap_contrib)), decreasing = T)]
  return(list(shap_score = shap_contrib,
              mean_shap_score = (mean_shap_score)))
}

# a function to standardize feature values into same range
std1 <- function(x){
  return ((x - min(x, na.rm = T))/(max(x, na.rm = T) - min(x, na.rm = T)))
}


# prep shap data
shap.prep <- function(shap  = shap_result, X_train = mydata$train_mm, top_n){
  require(ggforce)
  # descending order
  if (missing(top_n)) top_n <- dim(X_train)[2] # by default, use all features
  if (!top_n%in%c(1:dim(X_train)[2])) stop('supply correct top_n')
  require(data.table)
  shap_score_sub <- as.data.table(shap$shap_score)
  shap_score_sub <- shap_score_sub[, names(shap$mean_shap_score)[1:top_n], with = F]
  shap_score_long <- melt.data.table(shap_score_sub, measure.vars = colnames(shap_score_sub))
  
  # feature values: the values in the original dataset
  fv_sub <- as.data.table(X_train)[, names(shap$mean_shap_score)[1:top_n], with = F]
  # standardize feature values
  fv_sub_long <- melt.data.table(fv_sub, measure.vars = colnames(fv_sub))
  fv_sub_long[, stdfvalue := std1(value), by = "variable"]
  # SHAP value: value
  # raw feature value: rfvalue; 
  # standarized: stdfvalue
  names(fv_sub_long) <- c("variable", "rfvalue", "stdfvalue" )
  shap_long2 <- cbind(shap_score_long, fv_sub_long[,c('rfvalue','stdfvalue')])
  shap_long2[, mean_value := mean(abs(value)), by = variable]
  setkey(shap_long2, variable)
  return(shap_long2) 
}

plot.shap.summary <- function(data_long){
  x_bound <- max(abs(data_long$value))
  require('ggforce') # for `geom_sina`
  plot1 <- ggplot(data = data_long)+
    coord_flip() + 
    # sina plot: 
    geom_sina(aes(x = variable, y = value, color = stdfvalue)) +
    # print the mean absolute value: 
    geom_text(data = unique(data_long[, c("variable", "mean_value"), with = F]),
              aes(x = variable, y=-Inf, label = sprintf("%.3f", mean_value)),
              size = 3, alpha = 0.7,
              hjust = -0.2, 
              fontface = "bold") + # bold
    # # add a "SHAP" bar notation
    # annotate("text", x = -Inf, y = -Inf, vjust = -0.2, hjust = 0, size = 3,
    #          label = expression(group("|", bar(SHAP), "|"))) + 
    scale_color_gradient(low="#FFCC33", high="#6600CC", 
                         breaks=c(0,1), labels=c("Low","High")) +
    theme_bw() + 
    theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), # remove axis line
          legend.position="bottom") + 
    geom_hline(yintercept = 0) + # the vertical line
    scale_y_continuous(limits = c(-x_bound, x_bound)) +
    # reverse the order of features
    scale_x_discrete(limits = rev(levels(data_long$variable)) 
    ) + 
    labs(y = "SHAP value (impact on model output)", x = "", color = "Feature value") 
  return(plot1)
}

var_importance <- function(shap_result, top_n=10)
{
  var_importance=tibble(var=names(shap_result$mean_shap_score), importance=shap_result$mean_shap_score)
  
  var_importance=var_importance[1:top_n,]
  
  ggplot(var_importance, aes(x=reorder(var,importance), y=importance)) + 
    geom_bar(stat = "identity") + 
    coord_flip() + 
    theme_light() + 
    theme(axis.title.y=element_blank()) 
}
######- END OF FUNCTION -######
model = readRDS("xgbusingcompletecases.rds")
#SHAP importance
shap_result = shap.score.rank(xgb_model = model, 
                              X_train =dtrain,
                              shap_approx =F)

var_importance(shap_result, top_n=20)

#SHAP summary
shap_long = shap.prep(shap = shap_result,
                      X_train = as.matrix(dtrain), 
                      top_n = 10)

plot.shap.summary(data_long = shap_long)

#SHAP individual plot
xgb.plot.shap(data = dtrain, # input data
              model = model, # xgboost model
              features = names(shap_result$mean_shap_score[1:20]), # only top 10 var
              n_col = 3, # layout option
              plot_loess = T # add red line to plot
)
# 
# killipclass                  fbg                   bb                   ck          oralhypogly 
# 0.4839137957         0.3943104557         0.3669658760         0.3464916502         0.3421821685 
# bpsys                 acei  ptageatnotification               ptrace            heartrate 
# 0.3086193954         0.2873125901         0.2737420032         0.2245642694         0.2128006857 
# ecgabnormtypestdep                 ldlc                  pci                 hdlc                 ccap 
# 0.1672954951         0.1100134454         0.1069444072         0.1064478872         0.0954042291 
# tg               bpdias                   tc                  arb                 lmwh 
# 0.0898009419         0.0773855277         0.0700740065         0.0687199957         0.0641216000 
# cpremcvd               crenal               statin        smokingstatus              antiarr 
# 0.0550448612         0.0449695552         0.0423451394         0.0386825663         0.0375509166 
# cdm  ecgabnormlocational              insulin       canginapast2wk ecgabnormtypestelev2 
# 0.0249190665         0.0231671033         0.0223792613         0.0219935388         0.0146540310 
# diuretic   ecgabnormtypetwave ecgabnormtypestelev1  ecgabnormlocationll                ptsex 
# 0.0143527049         0.0139423827         0.0122979919         0.0112421398         0.0107802317 
# clung                 cdys  ecgabnormlocationil  ecgabnormlocationrv       calcantagonist 
# 0.0084112577         0.0077980138         0.0059303141         0.0034954044         0.0015147149 
# cardiaccath           cheartfail                 chpt                  cmi         canginamt2wk 
# 0.0011532209         0.0003024076         0.0000000000         0.0000000000         0.0000000000 
# ccerebrovascular           cpvascular     ecgabnormtypebbb  ecgabnormlocationtp                 cabg 
# 0.0000000000         0.0000000000         0.0000000000         0.0000000000         0.0000000000 
# asa                 gpri              heparin              lipidla 
# 0.0000000000         0.0000000000         0.0000000000         0.0000000000 


########### remove variables with zero importance based on shap ###########

library(readxl)
ds <- read_excel("elderlycompletecases.xlsx")
ds <- ds[,-1]
ds<- subset(ds, select = -c(timiscorestemi,timiscorenstemi,acsstratum, chpt, cmi,canginamt2wk,ccerebrovascular ,
                            cpvascular, ecgabnormtypebbb, ecgabnormlocationtp, cabg, asa , gpri, heparin, lipidla ))

ds$status <- ifelse(test = ds$status ==1, yes = "1", no = "0")
ds$status <- as.numeric(ds$status)

train.ind=createDataPartition(ds$status, times = 1, p=0.7, list = FALSE)
training.df=ds[train.ind, ]
testing.df=ds[-train.ind, ]

label=as.numeric(as.character(training.df$status))
ts_label=as.numeric(as.character(testing.df$status))

sumwpos=sum(label==1)
sumwneg=sum(label==0)
print(sumwneg/sumwpos)

dtest=sparse.model.matrix(status~.-1, data = data.frame(testing.df))
dtrain=sparse.model.matrix(status~.-1, training.df)

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

xgb.train = train(x = training.df,
                  y = factor(label, labels = c("Dead", "Alive")),
                  trControl = xgb.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree",
                  scale_pos_weight=sumwneg/sumwpos)


stopCluster(myCl)


xgb.train$bestTune
# nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
# 1     500         3 0.01     0                1                1       0.7

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

xgb.modshap=xgboost(data = dtrain, 
                label = label, 
                max.depth=xgb.train$bestTune$max_depth, 
                eta=xgb.train$bestTune$eta, 
                nthread=4, 
                min_child_weight=xgb.train$bestTune$min_child_weight,
                scale_pos_weight=sumwneg/sumwpos, 
                eval_metric="auc", 
                nrounds=xgb.crv$best_iteration, 
                objective="binary:logistic")

importance=xgb.importance(feature_names = colnames(dtrain), model = xgb.mod)
importance$Feature[1:20]
(gg=xgb.ggplot.importance(importance_matrix = importance[1:20,]))


result.predicted1 <- predict(xgb.mod, newdata = dtest, type = "prob")

result.roc1 <- roc(ts_label, result.predicted1)
result.roc1
# Area under the curve: 0.8618

############# testing stemi shapp ###################
saveRDS(xgb.modshap,"xgbusingcompletecasesshap.rds")
modelshap = readRDS("xgbusingcompletecasesshap.rds")

stemi <- read_csv("testdatasetelderly - stemi.csv")

stemi<- subset(stemi, select = -c(acsstratum,timiscorestemi,chpt, cmi,canginamt2wk,ccerebrovascular ,
                                  cpvascular, ecgabnormtypebbb, ecgabnormlocationtp, cabg, asa , gpri, heparin, lipidla )) #remove these variables
stemi <- stemi[,2:44]

stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.numeric(stemi$status)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

############### different method to plot roc #####################
stemipred <- predict (modelshap,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
# Area under the curve: 0.9386

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


################### testing nstemi  shap #################

nstemi <- read_csv("testdatasetelderly - nstemi.csv")

nstemi<- subset(nstemi, select = -c(acsstratum,timiscorenstemi,chpt, cmi,canginamt2wk,ccerebrovascular ,
                                    cpvascular, ecgabnormtypebbb, ecgabnormlocationtp, cabg, asa , gpri, heparin, lipidla)) #remove these variables
nstemi <- nstemi[,2:44]

nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.numeric(nstemi$status)

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) #remove status

############### different method to plot roc #####################
nstemipred <- predict (modelshap,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
# Area under the curve: 0.959

#predict
xgboostnstemi = xgb.DMatrix(data=xnstemi, label=ynstemi)
predictionnstemi = predict(model, newdata = xgboostnstemi )
predictionnstemi = ifelse(predictionnstemi > 0.5, 1, 0)
predictionnstemi
table(predictionnstemi)

####### confusion matrix ##########
predictionnstemi = as.factor(predictionnstemi) 
cmnstemi = confusionMatrix(predictionnstemi, as.factor(ynstemi), positive = '0')
cmnstemi



###########################################################################################################
#use boruta based on variables selected by shap
library(readxl)
ds <- read_excel("elderlycompletecases.xlsx")
ds
dim(ds)
ds<- subset(ds, select = -c(acsstratum,timiscorestemi,timiscorenstemi,patientid,ptageatnotification))

train.ind=createDataPartition(ds$status, times = 1, p=0.7, list = FALSE)
training.df=ds[train.ind, ]
testing.df=ds[-train.ind, ]

training.df<- subset(training.df, select = -c(chpt, cmi,canginamt2wk,ccerebrovascular ,
                                              cpvascular, ecgabnormtypebbb, ecgabnormlocationtp, cabg, asa , gpri, heparin, lipidla ))

#remove these variables
testing.df<- subset(testing.df, select = -c(chpt, cmi,canginamt2wk,ccerebrovascular ,
                                            cpvascular, ecgabnormtypebbb, ecgabnormlocationtp, cabg, asa , gpri, heparin, lipidla ))

training.df$status <- ifelse(test = training.df$status ==1, yes = "1", no = "0")
testing.df$status <- ifelse(test = testing.df$status ==1, yes = "1", no = "0")

training.df$status <- as.numeric(training.df$status)
testing.df$status <- as.numeric(testing.df$status)

label=as.numeric(as.character(training.df$status))
ts_label=as.numeric(as.character(testing.df$status))

sumwpos=sum(label==1)
sumwneg=sum(label==0)
print(sumwneg/sumwpos)

dtest=sparse.model.matrix(status~.-1, data = data.frame(testing.df))
dtrain=sparse.model.matrix(status~.-1, training.df)

xgb.grid=expand.grid(nrounds=c(25,50,100),
                     eta=c(0.01,0.05),
                     max_depth=3,
                     gamma = 0,
                     subsample = c(0.7,0.8,0.9),
                     min_child_weight = c(1,2),
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

xgb.train = train(x = training.df,
                  y = label,
                  trControl = xgb.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree",
                  scale_pos_weight=sumwneg/sumwpos)

stopCluster(myCl)

xgb.train$bestTune

params=list("eta"=xgb.train$bestTune$eta,
            "max_depth"=xgb.train$bestTune$max_depth,
            "gamma"=xgb.train$bestTune$gamma,
            "min_child_weight"=xgb.train$bestTune$min_child_weight,
            "nthread"=4,
            "objective"="binary:logistic")

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

xgb.modtop10=xgboost(data = dtrain,
                     label = label,
                     max.depth=xgb.train$bestTune$max_depth,
                     eta=xgb.train$bestTune$eta,
                     nthread=4,
                     min_child_weight=xgb.train$bestTune$min_child_weight,
                     scale_pos_weight=sumwneg/sumwpos,
                     eval_metric="auc",
                     nrounds=xgb.crv$best_iteration,
                     objective="binary:logistic")

importance=xgb.importance(feature_names = colnames(dtrain), model = xgb.modtop10)
importance$Feature[1:9]
(gg=xgb.ggplot.importance(importance_matrix = importance[1:9,]))
# [1] "killipclass" "ck"          "fbg"         "bb"          "bpsys"       "heartrate"   "oralhypogly"
# [8] "acei"        "tg" 

result.predicted1 <- predict(xgb.modtop10, newdata = dtest, type = "prob")

result.roc1 <- roc(ts_label, result.predicted1)
result.roc1
# Area under the curve: 0.853

result.predicted1 <- ifelse (result.predicted1 > 0.5,1,0)
result.predicted1 <- as.factor(result.predicted1)

cm_all_1 <- confusionMatrix (result.predicted1,
                             as.factor(ts_label))
cm_all_1

