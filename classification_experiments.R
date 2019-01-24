library(stats)
library(slam)
library(e1071)
library(xgboost)
library(ROCR)
library(dismo)
library(cluster)
library(e1071)
library(caret)
library(mlbench)
library(pROC)
library(proxy)
library(fossil)
library(factoextra)
library(rlist)

path <- getwd()
file1 <- "/datasets/haszysz_glove.csv"
file2 <- "/datasets/haszysz_tfidf.csv"
file3 <- "/datasets/bmw_glove.csv"
file4 <- "/datasets/bmw_tfidf.csv"

#Get results of folds with given parameters
getAllFoldsXgb <- function(modxgb, nr, mx, eta, gamma,cb, mcw, ss) {
    all_folds <- modxgb$pred[modxgb$pred$nrounds == nr
    & modxgb$pred$max_depth == mx
    & modxgb$pred$eta == eta
    & modxgb$pred$gamma == gamma
    & modxgb$pred$colsample_bytree == cb
    & modxgb$pred$min_child_weight == mcw
    & modxgb$pred$subsample == ss,]
    return(all_folds)
}

#Pick folds of svm result
getAllFoldsSvm <- function(modxgb) {
    all_folds <- modxgb$pred
    return(all_folds)
}

drawFirstPlot <- function(preds, obs, repr, color) {
    ROCRpred <- prediction(preds, obs)
    plot(performance(ROCRpred, 'tpr', 'fpr'), col = color)
    auc <- performance(ROCRpred, measure = "auc")@y.values[[1]]
    auc <- round(auc, 4)
    label <- paste0(repr, ", AUC=", auc)
    return(label)
}

drawPlot <- function(preds, obs, repr, color) {
    ROCRpred <- prediction(preds, obs)
    plot(performance(ROCRpred, 'tpr', 'fpr'), col = color, add = TRUE)
    auc <- performance(ROCRpred, measure = "auc")@y.values[[1]]
    auc <- round(auc, 4)
    label <- paste0(repr, ", AUC=", auc)
    return(label)
}


#Classification TF-IDF haszysz
data_frame <- read.csv(file = paste0(path, file2), header = TRUE, sep = ";")
data_frame$X <- NULL
classes <- data_frame$belongs_to
if (is.null(classes)) {
    classes <- data_frame$category
    data_frame$category <- NULL
}
classes <- make.names(classes)
data_frame$belongs_to <- NULL

#XGBoost
ctrlxgb_tfidf <- trainControl(method = "cv", savePred = TRUE, classProb = TRUE, summaryFunction = mnLogLoss, verboseIter = TRUE, number = 5)
modxgb_tfidf <- train(data_frame, classes, method = "xgbTree", metric = "logLoss", trControl = ctrlxgb_tfidf)

#SVM
ctrlsvm_tfidf <- trainControl(method = "cv", savePred = TRUE, classProb = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE, number = 5)
modsvm_tfidf <- train(data_frame, classes, method = "svmLinear", trControl = ctrlsvm_tfidf)

#Classification GloVe haszysz
data_frame <- read.csv(file = paste0(path, file1), header = TRUE, sep = ";")
data_frame$X <- NULL
classes <- data_frame$belongs_to
if (is.null(classes)) {
    classes <- data_frame$category
    data_frame$category <- NULL
}
classes <- make.names(classes)
data_frame$belongs_to <- NULL
#XGBoost
ctrlxgb_glove <- trainControl(method = "cv", savePred = TRUE, classProb = TRUE, summaryFunction = mnLogLoss, verboseIter = TRUE, number = 5)
modxgb_glove <- train(data_frame, classes, method = "xgbTree", metric = "logLoss", trControl = ctrlxgb_glove)
#SVM
ctrlsvm_glove <- trainControl(method = "cv", savePred = TRUE, classProb = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE, number = 5)
modsvm_glove <- train(data_frame, classes, method = "svmLinear", trControl = ctrlsvm_glove)


# Classification GloVe BMW
data_frame <- read.csv(file = paste0(path, file3), header = TRUE, sep = ";")
data_frame$X <- NULL
classes <- data_frame$belongs_to
if (is.null(classes)) {
    classes <- data_frame$category
    data_frame$category <- NULL
}
classes <- make.names(classes)
data_frame$belongs_to <- NULL

#XGBoost
ctrlxgb_glove_bmw <- trainControl(method = "cv", savePred = TRUE, classProb = TRUE, summaryFunction = mnLogLoss, verboseIter = TRUE, number = 5)
modxgb_glove_bmw <- train(data_frame, classes, method = "xgbTree", metric = "logLoss", trControl = ctrlxgb_glove_bmw)

#SVM
ctrlsvm_glove_bmw <- trainControl(method = "cv", savePred = TRUE, classProb = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE, number = 5)
modsvm_glove_bmw <- train(data_frame, classes, method = "svmLinear", trControl = ctrlsvm_glove_bmw)

# Classificaton TF-IDF BMW
data_frame <- read.csv(file = paste0(path, file4), header = TRUE, sep = ";")
data_frame$X <- NULL
classes <- data_frame$belongs_to
if (is.null(classes)) {
    classes <- data_frame$category
    data_frame$category <- NULL
}
classes <- make.names(classes)
data_frame$belongs_to <- NULL

### XGBoost
ctrlxgb_tfidf_bmw <- trainControl(method = "cv", savePred = TRUE, classProb = TRUE, summaryFunction = mnLogLoss, verboseIter = TRUE, number = 5)
modxgb_tfidf_bmw <- train(data_frame, classes, method = "xgbTree", metric = "logLoss", trControl = ctrlxgb_tfidf_bmw)

### SVM unsuccessfull attempt - too long execution time
ctrlsvm_tfidf_bmw <- trainControl(method = "cv", savePred = TRUE, classProb = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE, number = 5)
modsvm_tfidf_bmw <- train(data_frame, classes, method = "svmLinear", trControl = ctrlsvm_tfidf_bmw)


#PLOTS FOR HASZYSZ

all_folds_xgb_glove <- getAllFoldsXgb(modxgb_glove, 100, 3, 0.3, 0, 0.6, 1, 1)
preds_xgb_glove <- all_folds_xgb_glove$`Outdoor.MiejscĂłwka`
obs_xgb_glove <- all_folds_xgb_glove$obs
label3 <- drawFirstPlot(preds_xgb_glove, obs_xgb_glove, "XGBoost, GloVe", "blue")


all_folds_xgb_tfidf <- getAllFoldsXgb(modxgb_tfidf, 100, 3, 0.3, 0, 0.6, 1, 1)
preds_xgb_tfidf <- all_folds_xgb_tfidf$`Outdoor.MiejscĂłwka`
obs_xgb_tfidf <- all_folds_xgb_tfidf$obs
label4 <- drawPlot(preds_xgb_tfidf, obs_xgb_tfidf, "XGBoost, TF-IDF", "red")


all_folds_svm_glove <- getAllFoldsSvm(modsvm_glove)
preds_svm_glove <- all_folds_svm_glove$`Outdoor.MiejscĂłwka`
obs_svm_glove <- all_folds_svm_glove$obs
label1 <- drawPlot(preds_svm_glove, obs_svm_glove, "SVM, GloVe", "green2")

all_folds_svm_tfidf <- getAllFoldsSvm(modsvm_tfidf)
preds_svm_tfidf <- all_folds_svm_tfidf$`Outdoor.MiejscĂłwka`
obs_svm_tfidf <- all_folds_svm_tfidf$obs
label2 <- drawPlot(preds_svm_tfidf, obs_svm_tfidf, "SVM, TF-IDF", "darkorange1")


op <- par(cex = 1.5)
legend("bottomright", c(label3, label4, label1, label2), lty = 1, col = c("blue", "red", "green2", "darkorange1"), bty = "n", inset = c(0.0, 0))
abline(0, 1)


#PLOTS FOR BMW

all_folds_xgb_glove_bmw <- getAllFoldsXgb(modxgb_glove_bmw, 100, 3, 0.3, 0, 0.6, 1, 1)
preds_xgb_glove_bmw <- all_folds_xgb_glove_bmw$Spalanie...Paliwa.alternatywne
obs_xgb_glove_bmw <- all_folds_xgb_glove_bmw$obs
label3_bmw <- drawFirstPlot(preds_xgb_glove_bmw, obs_xgb_glove_bmw, "XGBoost, GloVe", "blue")


all_folds_xgb_tfidf_bmw <- getAllFoldsXgb(modxgb_tfidf_bmw, 100, 3, 0.3, 0, 0.6, 1, 1)
preds_xgb_tfidf_bmw <- all_folds_xgb_tfidf_bmw$Spalanie...Paliwa.alternatywne
obs_xgb_tfidf_bmw <- all_folds_xgb_tfidf_bmw$obs
label5_bmw <- drawPlot(preds_xgb_tfidf_bmw, obs_xgb_tfidf_bmw, "XGBoost, TF-IDF", "red")

all_folds_svm_glove_bmw <- getAllFoldsSvm(modsvm_glove_bmw)
preds_svm_glove_bmw <- all_folds_svm_glove_bmw$Spalanie...Paliwa.alternatywne
obs_svm_glove_bmw <- all_folds_svm_glove_bmw$obs
label4_bmw <- drawPlot(preds_svm_glove_bmw, obs_svm_glove_bmw, "SVM, GloVe", "green2")

op <- par(cex = 1.5)
legend("bottomright", c(label3_bmw, label5_bmw, label4_bmw), lty = 1, col = c("blue", "red", "green2"), bty = "n", inset = c(0.0, 0))
abline(0, 1)
