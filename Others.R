##################################################
## Author: Mohsen
## Maize Yield and Nitrate Loss prediction with ML
## Linear Regression, LASSO, Ridge
##################################################

library(dplyr)
library(tidyr)
library(glmnet)
library(ranger)
library(ggplot2)
library(Metrics)
library(caret)
library(xgboost)
library(pdp)
library(mmpf)


# Data Preprocessing ------------------------------------------------------

maizeData2 <- readRDS('maizeData2.rds') 

maizeData <- maizeData2 %>%
  mutate(year=as.factor(year)) %>%
  filter(maize_yield != 0) %>%
  filter(uniqueid != "SEPAC") %>%
  as.data.frame() 

dmy <- dummyVars(~sowTime+cultivar+Nrate+Nstrategy+residue+residueRemoval+
                   covercrop+initsoilN+waterTable+texture, data=maizeData)
maizeData_oh <- data.frame(predict(dmy, maizeData))

maizeData <- maizeData %>%
  select(uniqueid:sowTime_doy,cultivar_mg,Nrate_kg,residue_kg,initsoilN_kg,
         waterTable_mm,OC:maize_yield) %>%
  bind_cols(maizeData_oh)

maizeData <- maizeData[sample(1:nrow(maizeData)), ]

test <- maizeData %>%
  filter(year %in% c(2013:2016)) %>%
  select(-year, -uniqueid)

test_Y <- test %>% select(-NLoss)

test_N <- test %>% select(-maize_yield)

training <- maizeData %>%
  filter(year %in% c(1983:2012))

training_Y <- training %>% 
  select(-NLoss)

training_N <- training %>% 
  select(-maize_yield)

str(maizeData)
set.seed(1369)

# Average predictions
Average_Y <- mean(training_Y$maize_yield)
test_Y <- test_Y %>%
  mutate(Average_Y = Average_Y)
rmse(test_Y$maize_yield , test_Y$Average_Y)
test_Y <- test_Y[,-ncol(test_Y)]

Average_N <- mean(training_N$NLoss)
test_N <- test_N %>%
  mutate(Average_N = Average_N)
rmse(test_N$NLoss , test_N$Average_N)
test_N <- test_N[,-ncol(test_N)]



# Ridge Regression --------------------------------------------------------

## ---------- Maize Yield ---------- ## 

years_cut <- c(1988,1993,1998,2003,2008,2013)
lambda_ry <- 10^seq(-8, 3, length = 20)
lambdaDF_ry <- as.data.frame(lambda_ry)

for (i in 1:5) {
  valid_Y <- training_Y %>%
    filter(year %in% c(years_cut[i]:(years_cut[i+1]-1))) %>%
    select(-year, -uniqueid)
  
  train_Y <- training_Y %>%
    filter(year %in% c((years_cut[i]-5):(years_cut[i]-1))) %>%
    select(-year, -uniqueid)
  
  Ridge_Y <- train(maize_yield~., data=train_Y, method="glmnet",
                   tuneGrid = expand.grid(alpha = 0, lambda = lambda_ry))
  
  lambdaDF_ry <- cbind(lambdaDF_ry, Ridge_Y$results$RMSE)
}

lambdaDF_ry$meanError <- rowMeans(lambdaDF_ry[,c(2:ncol(lambdaDF_ry))])
position_ridgey = which.min(lambdaDF_ry$meanError)

training_Y1 <- training_Y %>%
  select(-year, -uniqueid)

Ridge_Y_final <- train(maize_yield~., data=training_Y1, method="glmnet",
                       tuneGrid = expand.grid(
                         alpha = 0,
                         lambda=lambdaDF_ry$lambda_ry[position_ridgey]))

saveRDS(Ridge_Y_final, 'model_ridgeY.rds')

Ridge_Y.predict <- predict(Ridge_Y_final,test_Y)
DFridge_Y <- data.frame(predicted = Ridge_Y.predict, actual = test_Y$maize_yield)
DFridge_Y$difference <- abs(DFridge_Y$predict-DFridge_Y$actual)
write.csv(DFridge_Y, file="DFridge_Y.csv")
RMSE_Ridge_Y <- rmse(predicted = DFridge_Y$predict, actual = DFridge_Y$actual)


## Training Error
PredT.Ridge_Y<- predict(Ridge_Y_final, training_Y1)
DFridge_Yt <- data.frame(predicted = PredT.Ridge_Y, actual = training_Y1$maize_yield)
DFridge_Yt$difference <- abs(DFridge_Yt$predict-DFridge_Yt$actual)
RMSE_T_Ridge_Y <- rmse(predicted = DFridge_Yt$predict, actual = DFridge_Yt$actual)


# hyperparameters plot
write.csv(lambdaDF_ry,'lambdaDF_ry.csv')


# hyperparameters plot (test)
hplot_ridgeY <- as.data.frame(lambda_ry)
for (i in 1:length(lambda_ry)) {
  final <- train(maize_yield~., data=training_Y1, method="glmnet",
                 tuneGrid = expand.grid(alpha = 0, lambda=lambdaDF_ry$lambda_ry[position_ridgey]))
  predict <- predict(final,test_Y)
  df <- data.frame(predicted = predict, actual = test_Y$maize_yield)
  df$difference <- abs(df$predict-df$actual)
  rmse <- rmse(predicted = df$predict, actual = df$actual)
  hplot_ridgeY[i,2] <- rmse
}
write.csv(hplot_ridgeY, 'hplot_ridgeY.csv')


# permutation importance
imp_ridgeY <- as.data.frame(names(training_Y1))
for (i in 1:nrow(imp_ridgeY)) {
  imp_ridgeY[i,2] <- permutationImportance(data=training_Y1,
                                           vars=as.character(imp_ridgeY[i,1]),
                                           y='maize_yield',
                                           model=Ridge_Y_final, nperm=10)
}
write.csv(imp_ridgeY, 'imp_ridgeY.csv')




## ---------- Nitrogen Loss ---------- ##

years_cut <- c(1988,1993,1998,2003,2008,2013)
lambda_rn <- 10^seq(-8, 3, length = 20)
lambdaDF_rn <- as.data.frame(lambda_rn)

for (i in 1:5) {
  valid_N <- training_N %>%
    filter(year %in% c(years_cut[i]:(years_cut[i+1]-1))) %>%
    select(-year, -uniqueid)
  
  train_N <- training_N %>%
    filter(year %in% c((years_cut[i]-5):(years_cut[i]-1))) %>%
    select(-year, -uniqueid)
  
  Ridge_N <- train(NLoss~., data=train_N, method="glmnet",
                   tuneGrid = expand.grid(alpha = 0, lambda = lambda_rn))
  
  lambdaDF_rn <- cbind(lambdaDF_rn, Ridge_N$results$RMSE)
}

lambdaDF_rn$meanError <- rowMeans(lambdaDF_rn[,c(2:ncol(lambdaDF_rn))])
position_ridgen = which.min(lambdaDF_rn$meanError)

training_N1 <- training_N %>%
  select(-year, -uniqueid)

Ridge_N_final <- train(NLoss~., data=training_N1, method="glmnet",
                       tuneGrid = expand.grid(
                         alpha = 0,
                         lambda=lambdaDF_rn$lambda_rn[position_ridgen]))

saveRDS(Ridge_N_final, 'model_ridgeN.rds')

Ridge_N.predict <- predict(Ridge_N_final,test_N)
DFridge_N <- data.frame(predicted = Ridge_N.predict, actual = test_N$NLoss)
DFridge_N$difference <- abs(DFridge_N$predict-DFridge_N$actual)
write.csv(DFridge_N, file="DFridge_N.csv")
RMSE_Ridge_N <- rmse(predicted = DFridge_N$predict, actual = DFridge_N$actual)


## Training Error
PredT.Ridge_N <- predict(Ridge_N_final, training_N1)
DFridge_Nt <- data.frame(predicted = PredT.Ridge_N, actual = training_N1$NLoss)
DFridge_Nt$difference <- abs(DFridge_Nt$predict-DFridge_Nt$actual)
RMSE_T_Ridge_N <- rmse(predicted = DFridge_Nt$predict, actual = DFridge_Nt$actual)


# hyperparameters plot
write.csv(lambdaDF_rn,'lambdaDF_rn.csv')


# hyperparameters plot (test)
hplot_ridgeN <- as.data.frame(lambda_rn)
for (i in 1:length(lambda_rn)) {
  final <- train(NLoss~., data=training_N1, method="glmnet",
                 tuneGrid = expand.grid(alpha = 0, lambda=lambdaDF_rn$lambda_rn[position_ridgen]))
  predict <- predict(final,test_N)
  df <- data.frame(predicted = predict, actual = test_N$NLoss)
  df$difference <- abs(df$predict-df$actual)
  rmse <- rmse(predicted = df$predict, actual = df$actual)
  hplot_ridgeN[i,2] <- rmse
}
write.csv(hplot_ridgeN, 'hplot_ridgeN.csv')


# permutation importance
imp_ridgeN <- as.data.frame(names(training_N1))
for (i in 1:nrow(imp_ridgeN)) {
  imp_ridgeN[i,2] <- permutationImportance(data=training_N1,
                                           vars=as.character(imp_ridgeN[i,1]),
                                           y='NLoss',
                                           model=Ridge_N_final, nperm=10)
}
write.csv(imp_ridgeN, 'imp_ridgeN.csv')




# Lasso Regression --------------------------------------------------------

## ---------- Maize Yield ---------- ## 

years_cut <- c(1988,1993,1998,2003,2008,2013)
lambda_ly <- 10^seq(-8, 3, length = 20)
lambdaDF_ly <- as.data.frame(lambda_ly)

for (i in 1:5) {
  valid_Y <- training_Y %>%
    filter(year %in% c(years_cut[i]:(years_cut[i+1]-1))) %>%
    select(-year, -uniqueid)
  
  train_Y <- training_Y %>%
    filter(year %in% c((years_cut[i]-5):(years_cut[i]-1))) %>%
    select(-year, -uniqueid)
  
  Lasso_Y <- train(maize_yield~., data=train_Y, method="glmnet",
                   tuneGrid = expand.grid(alpha = 1, lambda = lambda_ly))
  
  lambdaDF_ly <- cbind(lambdaDF_ly, Lasso_Y$results$RMSE)
}

lambdaDF_ly$meanError <- rowMeans(lambdaDF_ly[,c(2:ncol(lambdaDF_ly))])
position_lassoy = which.min(lambdaDF_ly$meanError)
position_lassoy

training_Y1 <- training_Y %>%
  select(-year, -uniqueid)

Lasso_Y_final <- train(maize_yield~., data=training_Y1, method="glmnet",
                       tuneGrid = expand.grid(
                         alpha = 1,
                         lambda=lambdaDF_ly$lambda_ly[position_lassoy]))

saveRDS(Lasso_Y_final, 'model_lassoY.rds')

Lasso_Y.predict <- predict(Lasso_Y_final,test_Y)
DFlasso_Y <- data.frame(predicted = Lasso_Y.predict, actual = test_Y$maize_yield)
DFlasso_Y$difference <- abs(DFlasso_Y$predict-DFlasso_Y$actual)
write.csv(DFlasso_Y, file="DFlasso_Y.csv")
RMSE_Lasso_Y <- rmse(predicted = DFlasso_Y$predict, actual = DFlasso_Y$actual)

## Training Error
PredT.Lasso_Y<- predict(Lasso_Y_final, training_Y1)
DFlasso_Yt <- data.frame(predicted = PredT.Lasso_Y, actual = training_Y1$maize_yield)
DFlasso_Yt$difference <- abs(DFlasso_Yt$predict-DFlasso_Yt$actual)
RMSE_T_Lasso_Y <- rmse(predicted = DFlasso_Yt$predict, actual = DFlasso_Yt$actual)


# hyperparameters plot
write.csv(lambdaDF_ly,'lambdaDF_ly.csv')


# hyperparameters plot (test)
hplot_lassoY <- as.data.frame(lambda_ly)
for (i in 1:length(lambda_ly)) {
  final <- train(maize_yield~., data=training_Y1, method="glmnet",
                 tuneGrid = expand.grid(alpha = 1, lambda=lambdaDF_ly$lambda_ly[position_lassoy]))
  predict <- predict(final,test_Y)
  df <- data.frame(predicted = predict, actual = test_Y$maize_yield)
  df$difference <- abs(df$predict-df$actual)
  rmse <- rmse(predicted = df$predict, actual = df$actual)
  hplot_lassoY[i,2] <- rmse
}
write.csv(hplot_lassoY, 'hplot_lassoY.csv')


# permutation importance
imp_lassoY <- as.data.frame(names(training_Y1))
for (i in 1:nrow(imp_lassoY)) {
  imp_lassoY[i,2] <- permutationImportance(data=training_Y1,
                                           vars=as.character(imp_lassoY[i,1]),
                                           y='maize_yield',
                                           model=Lasso_Y_final, nperm=10)
}
write.csv(imp_lassoY, 'imp_lassoY.csv')




## ---------- Nitrogen Loss ---------- ##

years_cut <- c(1988,1993,1998,2003,2008,2013)
lambda_ln <- 10^seq(-8, 3, length = 20)
lambdaDF_ln <- as.data.frame(lambda_ln)

for (i in 1:5) {
  valid_N <- training_N %>%
    filter(year %in% c(years_cut[i]:(years_cut[i+1]-1))) %>%
    select(-year, -uniqueid)
  
  train_N <- training_N %>%
    filter(year %in% c((years_cut[i]-5):(years_cut[i]-1))) %>%
    select(-year, -uniqueid)
  
  Lasso_N <- train(NLoss~., data=train_N, method="glmnet",
                   tuneGrid = expand.grid(alpha = 1, lambda = lambda_ln))
  
  lambdaDF_ln <- cbind(lambdaDF_ln, Lasso_N$results$RMSE)
}

lambdaDF_ln$meanError <- rowMeans(lambdaDF_ln[,c(2:ncol(lambdaDF_ln))])
position_lasson = which.min(lambdaDF_ln$meanError)
position_lasson

training_N1 <- training_N %>%
  select(-year, -uniqueid)

Lasso_N_final <- train(NLoss~., data=training_N1, method="glmnet",
                       tuneGrid = expand.grid(
                         alpha = 1,
                         lambda=lambdaDF_ln$lambda_ln[position_lasson]))

saveRDS(Lasso_N_final, 'model_lassoN.rds')

Lasso_N.predict <- predict(Lasso_N_final,test_N)
DFlasso_N <- data.frame(predicted = Lasso_N.predict, actual = test_N$NLoss)
DFlasso_N$difference <- abs(DFlasso_N$predict-DFlasso_N$actual)
write.csv(DFlasso_N, file="DFlasso_N.csv")
RMSE_Lasso_N <- rmse(predicted = DFlasso_N$predict, actual = DFlasso_N$actual)

## Training Error
PredT.Lasso_N <- predict(Lasso_N_final, training_N1)
DFlasso_Nt <- data.frame(predicted = PredT.Lasso_N, actual = training_N1$NLoss)
DFlasso_Nt$difference <- abs(DFlasso_Nt$predict-DFlasso_Nt$actual)
RMSE_T_Lasso_N <- rmse(predicted = DFlasso_Nt$predict, actual = DFlasso_Nt$actual)


# hyperparameters plot
write.csv(lambdaDF_ln,'lambdaDF_ln.csv')


# hyperparameters plot (test)
hplot_lassoN <- as.data.frame(lambda_ln)
for (i in 1:length(lambda_ln)) {
  final <- train(NLoss~., data=training_N1, method="glmnet",
                 tuneGrid = expand.grid(alpha = 1, lambda=lambdaDF_ln$lambda_ln[position_lasson]))
  predict <- predict(final,test_N)
  df <- data.frame(predicted = predict, actual = test_N$NLoss)
  df$difference <- abs(df$predict-df$actual)
  rmse <- rmse(predicted = df$predict, actual = df$actual)
  hplot_lassoN[i,2] <- rmse
}
write.csv(hplot_lassoN, 'hplot_lassoN.csv')


# permutation importance
imp_lassoN <- as.data.frame(names(training_N1))
for (i in 1:nrow(imp_lassoN)) {
  imp_lassoN[i,2] <- permutationImportance(data=training_N1,
                                           vars=as.character(imp_lassoN[i,1]),
                                           y='NLoss',
                                           model=Lasso_N_final, nperm=10)
}
write.csv(imp_lassoN, 'imp_lassoN.csv')




# Linear Regression -------------------------------------------------------

## ---------- Maize Yield ---------- ##

LR_Y <- lm(maize_yield~., data=training_Y1)
print(LR_Y)
summary(LR_Y)

LR_Y.predict <- predict.lm(LR_Y, test_Y)
df_LR_Y <- data.frame(predicted = LR_Y.predict, actual = test_Y$maize_yield)
df_LR_Y$difference <- abs(df_LR_Y$predict-df_LR_Y$actual)
write.csv(df_LR_Y, file="df_LR_Y.csv")
RMSE_LR_Y <- rmse(predicted = df_LR_Y$predict, actual = df_LR_Y$actual)
saveRDS(LR_Y, 'model_lrY.rds')


## Training Error
PredT.LR_Y <- predict.lm(LR_Y, training_Y1)
LLL_Y <- data.frame(predicted = PredT.LR_Y, actual = training_Y1$maize_yield)
LLL_Y$difference <- abs(LLL_Y$predict-LLL_Y$actual)
RMSE_T_LR_Y <- rmse(predicted = LLL_Y$predict, actual = LLL_Y$actual)


# permutation importance
imp_lrY <- as.data.frame(names(training_Y1))
for (i in 1:nrow(imp_lrY)) {
  imp_lrY[i,2] <- permutationImportance(data=training_Y1,
                                        vars=as.character(imp_lrY[i,1]),
                                        y='maize_yield',
                                        model=LR_Y, nperm=10)
}
write.csv(imp_lrY, 'imp_lrY.csv')



## ---------- Nitrogen Loss ---------- ##

LR_N <- lm(NLoss~., data=training_N1)
print(LR_N)
summary(LR_N)

LR_N.predict <- predict.lm(LR_N, test_N)
df_LR_N <- data.frame(predicted = LR_N.predict, actual = test_N$NLoss)
df_LR_N$difference <- abs(df_LR_N$predict-df_LR_N$actual)
write.csv(df_LR_N, file="df_LR_N.csv")
RMSE_LR_N <- rmse(predicted = df_LR_N$predict, actual = df_LR_N$actual)
saveRDS(LR_N, 'model_lrN.rds')


## Training Error
PredT.LR_N <- predict.lm(LR_N, training_N1)
LLL_N <- data.frame(predicted = PredT.LR_N, actual = training_N1$NLoss)
LLL_N$difference <- abs(LLL_N$predict-LLL_N$actual)
RMSE_T_LR_N <- rmse(predicted = LLL_N$predict, actual = LLL_N$actual)


# permutation importance
imp_lrN <- as.data.frame(names(training_N1))
for (i in 1:nrow(imp_lrN)) {
  imp_lrN[i,2] <- permutationImportance(data=training_N1,
                                        vars=as.character(imp_lrN[i,1]),
                                        y='NLoss',
                                        model=LR_N, nperm=10)
}
write.csv(imp_lrN, 'imp_lrN.csv')



### ----------- RESULTS ------------ ###

test_results <- data.frame(model=c('RidgeY','RidgeN','LassoY','LassoN','LrY','LrN'),
                           rmse=c(RMSE_Ridge_Y,RMSE_Ridge_N,RMSE_Lasso_Y,
                                  RMSE_Lasso_N,RMSE_LR_Y,RMSE_LR_N))

train_results <- data.frame(model=c('RidgeY','RidgeN','LassoY','LassoN','LrY','LrN'),
                            rmse=c(RMSE_T_Ridge_Y,RMSE_T_Ridge_N,RMSE_T_Lasso_Y,
                                   RMSE_T_Lasso_N,RMSE_T_LR_Y,RMSE_T_LR_N))

write.csv(test_results, '2results.csv')
write.csv(train_results, '2results_train.csv')



