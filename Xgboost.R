##################################################
## Author: Mohsen
## Maize Yield and Nitrate Loss prediction with ML
## XGBoost
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

set.seed(1369)




# XGBoost -----------------------------------------------------------

## ---------- Maize Yield ---------- ## 

years_cut <- c(1988,1993,1998,2003,2008,2013)
XGB_parameters_Y <- expand.grid(nrounds=c(15,20,30),
                                eta=seq(.1,.7, by=.1),
                                gamma=c(5,10))
errorXGBy = matrix(ncol=5 , nrow = 42)

for (i in 1:5) {
  valid_Y <- training_Y %>%
    filter(year %in% c(years_cut[i]:(years_cut[i+1]-1))) %>%
    select(-year, -uniqueid)

  train_Y <- training_Y %>%
    filter(year %in% c((years_cut[i]-5):(years_cut[i]-1))) %>%
    select(-year, -uniqueid)

  for (j in 1:42) {
    XGB_Y <- xgboost(data=data.matrix(train_Y[,-17]),
                     label=data.matrix(train_Y[,17]),
                     nrounds=XGB_parameters_Y$nrounds[j],
                     eta=XGB_parameters_Y$eta[j],
                     gamma=XGB_parameters_Y$gamma[j],
                     booster="gbtree",
                     objective="reg:linear")
    XGB_Y

    XGB_Y.predictak <- predict(XGB_Y,data.matrix(valid_Y[,-17]))
    errorXGBy[j,i] = rmse(actual = valid_Y$maize_yield,
                          predicted = XGB_Y.predictak)
  }
}

errorXGBy <- as.data.frame(errorXGBy)
errorXGBy$mean <- rowMeans(errorXGBy)
position_XGBy = which.min(errorXGBy$mean)
position_XGBy

training_Y1 <- training_Y %>%
  select(-year, -uniqueid)

XGB_Y_final <- train(x = as.matrix(select(training_Y1, -maize_yield)), 
                     y = training_Y1$maize_yield,
                     method='xgbTree',
                     tuneGrid = expand.grid(nrounds=XGB_parameters_Y$nrounds[position_XGBy],
                                            eta=XGB_parameters_Y$eta[position_XGBy],
                                            gamma=XGB_parameters_Y$gamma[position_XGBy],
                                            max_depth=6,
                                            min_child_weight=1,
                                            subsample=1,
                                            colsample_bytree=1))

saveRDS(XGB_Y_final, 'model_xgbY2.rds')

XGB_Y.predict <- predict(XGB_Y_final,data.matrix(test_Y[,-17]))
df_XGBY <- data.frame(predicted = XGB_Y.predict, actual = test_Y$maize_yield)
df_XGBY$difference <- df_XGBY$predict-df_XGBY$actual
write.csv(df_XGBY, file="df_XGBY2.csv")
RMSE_XGB_Y <- rmse(predicted = df_XGBY$predict, actual = df_XGBY$actual)


## Training Error
PredT.XGB_Y <- predict(XGB_Y_final,data.matrix(training_Y1[,-17]))
df_XGBYt <- data.frame(predicted = PredT.XGB_Y, actual = training_Y1$maize_yield)
df_XGBYt$difference <- df_XGBYt$predict-df_XGBYt$actual
RMSE_T_XGB_Y <- rmse(predicted = df_XGBYt$predict, actual = df_XGBYt$actual)


# hyperparameters plot
write.csv(errorXGBy, 'errorXGBy2.csv')


# hyperparameters plot (test) * nrounds
hplot_xgbY_nrounds <- as.data.frame(unique(XGB_parameters_Y$nrounds))
for (i in 1:length(unique(XGB_parameters_Y$nrounds))) {
  final <- xgboost(data=data.matrix(training_Y1[,-17]),
                   label=data.matrix(training_Y1[,17]),
                   nrounds=unique(XGB_parameters_Y$nrounds)[i],
                   eta=XGB_parameters_Y$eta[position_XGBy],
                   gamma=XGB_parameters_Y$gamma[position_XGBy],
                   booster="gbtree",
                   objective="reg:linear")
  predict <- predict(final,data.matrix(test_Y[,-17]))
  df <- data.frame(predicted = predict, actual = test_Y$maize_yield)
  df$difference <- abs(df$predict-df$actual)
  rmse <- rmse(predicted = df$predict, actual = df$actual)
  hplot_xgbY_nrounds[i,2] <- rmse
}
write.csv(hplot_xgbY_nrounds, 'hplot_xgbY_nrounds2.csv')


# hyperparameters plot (test) * eta
hplot_xgbY_eta <- as.data.frame(unique(XGB_parameters_Y$eta))
for (i in 1:length(unique(XGB_parameters_Y$eta))) {
  final <- xgboost(data=data.matrix(training_Y1[,-17]),
                   label=data.matrix(training_Y1[,17]),
                   nrounds=XGB_parameters_Y$nrounds[position_XGBy],
                   eta=unique(XGB_parameters_Y$eta)[i],
                   gamma=XGB_parameters_Y$gamma[position_XGBy],
                   booster="gbtree",
                   objective="reg:linear")
  predict <- predict(final,data.matrix(test_Y[,-17]))
  df <- data.frame(predicted = predict, actual = test_Y$maize_yield)
  df$difference <- abs(df$predict-df$actual)
  rmse <- rmse(predicted = df$predict, actual = df$actual)
  hplot_xgbY_eta[i,2] <- rmse
}
write.csv(hplot_xgbY_eta, 'hplot_xgbY_eta2.csv')


# hyperparameters plot (test) * gamma
hplot_xgbY_gamma <- as.data.frame(unique(XGB_parameters_Y$gamma))
for (i in 1:length(unique(XGB_parameters_Y$gamma))) {
  final <- xgboost(data=data.matrix(training_Y1[,-17]),
                   label=data.matrix(training_Y1[,17]),
                   nrounds=XGB_parameters_Y$nrounds[position_XGBy],
                   eta=XGB_parameters_Y$eta[position_XGBy],
                   gamma=unique(XGB_parameters_Y$gamma)[i],
                   booster="gbtree",
                   objective="reg:linear")
  predict <- predict(final,data.matrix(test_Y[,-17]))
  df <- data.frame(predicted = predict, actual = test_Y$maize_yield)
  df$difference <- abs(df$predict-df$actual)
  rmse <- rmse(predicted = df$predict, actual = df$actual)
  hplot_xgbY_gamma[i,2] <- rmse
}
write.csv(hplot_xgbY_gamma, 'hplot_xgbY_gamma2.csv')


# pdp plots
x_pdp_Y <- data.matrix(training_Y1[,-17])
pdp_sowTime <- partial(XGB_Y_final, pred.var='sowTime_doy')
pdp_cultivar <- partial(XGB_Y_final, pred.var='cultivar_mg')
pdp_Nrate <- partial(XGB_Y_final, pred.var='Nrate_kg')
pdp_residue <- partial(XGB_Y_final, pred.var='residue_kg')
pdp_initsoilN <- partial(XGB_Y_final, pred.var='initsoilN_kg')
pdp_watertable <- partial(XGB_Y_final, pred.var='waterTable_mm')
pdp_OC <- partial(XGB_Y_final, pred.var='OC')
pdp_PAWC <- partial(XGB_Y_final, pred.var='PAWC')
pdp_mint <- partial(XGB_Y_final, pred.var='mint_avg')
pdp_maxt <- partial(XGB_Y_final, pred.var='maxt_avg')
pdp_gdd <- partial(XGB_Y_final, pred.var='gdd10_sum')
pdp_rain <- partial(XGB_Y_final, pred.var='rain_sum')

write.csv(pdp_sowTime, 'XGBy_pdp_sowTime2.csv')
write.csv(pdp_cultivar, 'XGBy_pdp_cultivar2.csv')
write.csv(pdp_Nrate, 'XGBy_pdp_Nrate2.csv')
write.csv(pdp_residue, 'XGBy_pdp_residue2.csv')
write.csv(pdp_initsoilN, 'XGBy_pdp_initsoilN2.csv')
write.csv(pdp_watertable, 'XGBy_pdp_watertable2.csv')
write.csv(pdp_OC, 'XGBy_pdp_OC2.csv')
write.csv(pdp_PAWC, 'XGBy_pdp_PAWC2.csv')
write.csv(pdp_mint, 'XGBy_pdp_mint2.csv')
write.csv(pdp_maxt, 'XGBy_pdp_maxt2.csv')
write.csv(pdp_gdd, 'XGBy_pdp_gdd2.csv')
write.csv(pdp_rain, 'XGBy_pdp_rain2.csv')


# permutation importance
imp_xgbY <- as.data.frame(names(training_Y1))
for (i in 1:nrow(imp_xgbY)) {
  imp_xgbY[i,2] <- permutationImportance(data=training_Y1,
                                         vars=as.character(imp_xgbY[i,1]),
                                         y='maize_yield',
                                         model=XGB_Y_final, nperm=10)
}
write.csv(imp_xgbY, 'imp_xgbY2.csv')



## ---------- Nitrogen Loss ---------- ##

years_cut <- c(1988,1993,1998,2003,2008,2013)
XGB_parameters_N <- expand.grid(nrounds=c(15,20,30),
                                eta=seq(.1,.7, by=.1),
                                gamma=c(5,10))
errorXGBn = matrix(ncol=5 , nrow = 42)

for (i in 1:5) {
  valid_N <- training_N %>%
    filter(year %in% c(years_cut[i]:(years_cut[i+1]-1))) %>%
    select(-year, -uniqueid)

  train_N <- training_N %>%
    filter(year %in% c((years_cut[i]-5):(years_cut[i]-1))) %>%
    select(-year, -uniqueid)

  for (j in 1:42) {
    XGB_N <- xgboost(data=data.matrix(train_N[,-17]),
                     label=data.matrix(train_N[,17]),
                     nrounds=XGB_parameters_N$nrounds[j],
                     eta=XGB_parameters_N$eta[j],
                     gamma=XGB_parameters_N$gamma[j],
                     booster="gbtree",
                     objective="reg:linear")
    XGB_N

    XGB_N.predictak <- predict(XGB_N,data.matrix(valid_N[,-17]))
    errorXGBn[j,i] = rmse(actual = valid_N$NLoss,
                          predicted = XGB_N.predictak)
  }
}

errorXGBn <- as.data.frame(errorXGBn)
errorXGBn$mean <- rowMeans(errorXGBn)
position_XGBn = which.min(errorXGBn$mean)
position_XGBn

training_N1 <- training_N %>%
  select(-year, -uniqueid)

XGB_N_final <- train(x = as.matrix(select(training_N1, -NLoss)), 
                     y = training_N1$NLoss,
                     method='xgbTree',
                     tuneGrid = expand.grid(nrounds=XGB_parameters_N$nrounds[position_XGBn],
                                            eta=XGB_parameters_N$eta[position_XGBn],
                                            gamma=XGB_parameters_N$gamma[position_XGBn],
                                            max_depth=20,
                                            min_child_weight=1,
                                            subsample=1,
                                            colsample_bytree=1))


saveRDS(XGB_N_final, 'model_xgbN2.rds')

XGB_N.predict <- predict(XGB_N_final,data.matrix(test_N[,-17]))
df_XGBN <- data.frame(predicted = XGB_N.predict, actual = test_N$NLoss)
df_XGBN$difference <- df_XGBN$predict-df_XGBN$actual
write.csv(df_XGBN, file="df_XGBN2.csv")
RMSE_XGB_N <- rmse(predicted = df_XGBN$predict, actual = df_XGBN$actual)


## Training Error
PredT.XGB_N <- predict(XGB_N_final,data.matrix(training_N1[,-17]))
df_XGBNt <- data.frame(predicted = PredT.XGB_N, actual = training_N1$NLoss)
df_XGBNt$difference <- df_XGBNt$predict-df_XGBNt$actual
RMSE_T_XGB_N <- rmse(predicted = df_XGBNt$predict, actual = df_XGBNt$actual)


# hyperparameters plot
write.csv(errorXGBn, 'errorXGBn2.csv')


# hyperparameters plot (test) * nrounds
hplot_xgbN_nrounds <- as.data.frame(unique(XGB_parameters_N$nrounds))
for (i in 1:length(unique(XGB_parameters_N$nrounds))) {
  final <- xgboost(data=data.matrix(training_N1[,-17]),
                   label=data.matrix(training_N1[,17]),
                   nrounds=unique(XGB_parameters_N$nrounds)[i],
                   eta=XGB_parameters_N$eta[position_XGBn],
                   gamma=XGB_parameters_N$gamma[position_XGBn],
                   booster="gbtree",
                   objective="reg:linear")
  predict <- predict(final,data.matrix(test_N[,-17]))
  df <- data.frame(predicted = predict, actual = test_N$NLoss)
  df$difference <- abs(df$predict-df$actual)
  rmse <- rmse(predicted = df$predict, actual = df$actual)
  hplot_xgbN_nrounds[i,2] <- rmse
}
write.csv(hplot_xgbN_nrounds, 'hplot_xgbN_nrounds2.csv')


# hyperparameters plot (test) * eta
hplot_xgbN_eta <- as.data.frame(unique(XGB_parameters_N$eta))
for (i in 1:length(unique(XGB_parameters_N$eta))) {
  final <- xgboost(data=data.matrix(training_N1[,-17]),
                   label=data.matrix(training_N1[,17]),
                   nrounds=XGB_parameters_N$nrounds[position_XGBn],
                   eta=unique(XGB_parameters_N$eta)[i],
                   gamma=XGB_parameters_N$gamma[position_XGBn],
                   booster="gbtree",
                   objective="reg:linear")
  predict <- predict(final,data.matrix(test_N[,-17]))
  df <- data.frame(predicted = predict, actual = test_N$NLoss)
  df$difference <- abs(df$predict-df$actual)
  rmse <- rmse(predicted = df$predict, actual = df$actual)
  hplot_xgbN_eta[i,2] <- rmse
}
write.csv(hplot_xgbN_eta, 'hplot_xgbN_eta2.csv')


# hyperparameters plot (test) * gamma
hplot_xgbN_gamma <- as.data.frame(unique(XGB_parameters_N$gamma))
for (i in 1:length(unique(XGB_parameters_N$gamma))) {
  final <- xgboost(data=data.matrix(training_N1[,-17]),
                   label=data.matrix(training_N1[,17]),
                   nrounds=XGB_parameters_N$nrounds[position_XGBn],
                   eta=XGB_parameters_N$eta[position_XGBn],
                   gamma=unique(XGB_parameters_N$gamma)[i],
                   booster="gbtree",
                   objective="reg:linear")
  predict <- predict(final,data.matrix(test_N[,-17]))
  df <- data.frame(predicted = predict, actual = test_N$NLoss)
  df$difference <- abs(df$predict-df$actual)
  rmse <- rmse(predicted = df$predict, actual = df$actual)
  hplot_xgbN_gamma[i,2] <- rmse
}
write.csv(hplot_xgbN_gamma, 'hplot_xgbN_gamma2.csv')


# pdp plots
x_pdp_N <- data.matrix(training_N1[,-17])
npdp_sowTime <- partial(XGB_N_final, pred.var='sowTime_doy')
npdp_cultivar <- partial(XGB_N_final, pred.var='cultivar_mg')
npdp_Nrate <- partial(XGB_N_final, pred.var='Nrate_kg')
npdp_residue <- partial(XGB_N_final, pred.var='residue_kg')
npdp_initsoilN <- partial(XGB_N_final, pred.var='initsoilN_kg')
npdp_watertable <- partial(XGB_N_final, pred.var='waterTable_mm')
npdp_OC <- partial(XGB_N_final, pred.var='OC')
npdp_PAWC <- partial(XGB_N_final, pred.var='PAWC')
npdp_mint <- partial(XGB_N_final, pred.var='mint_avg')
npdp_maxt <- partial(XGB_N_final, pred.var='maxt_avg')
npdp_gdd <- partial(XGB_N_final, pred.var='gdd10_sum')
npdp_rain <- partial(XGB_N_final, pred.var='rain_sum')

write.csv(npdp_sowTime, 'XGBn_npdp_sowTime2.csv')
write.csv(npdp_cultivar, 'XGBn_npdp_cultivar2.csv')
write.csv(npdp_Nrate, 'XGBn_npdp_Nrate2.csv')
write.csv(npdp_residue, 'XGBn_npdp_residue2.csv')
write.csv(npdp_initsoilN, 'XGBn_npdp_initsoilN2.csv')
write.csv(npdp_watertable, 'XGBn_npdp_watertable2.csv')
write.csv(npdp_OC, 'XGBn_npdp_OC2.csv')
write.csv(npdp_PAWC, 'XGBn_npdp_PAWC2.csv')
write.csv(npdp_mint, 'XGBn_npdp_mint2.csv')
write.csv(npdp_maxt, 'XGBn_npdp_maxt2.csv')
write.csv(npdp_gdd, 'XGBn_npdp_gdd2.csv')
write.csv(npdp_rain, 'XGBn_npdp_rain2.csv')


# permutation importance
imp_xgbN <- as.data.frame(names(training_N1))
for (i in 1:nrow(imp_xgbN)) {
  imp_xgbN[i,2] <- permutationImportance(data=training_N1,
                                         vars=as.character(imp_xgbN[i,1]),
                                         y='NLoss',
                                         model=XGB_N_final, nperm=10)
}
write.csv(imp_xgbN, 'imp_xgbN2.csv')




### ----------- RESULTS ------------ ###

test_results <- data.frame(model=c('xgbY','xgbN'),
                           rmse=c(RMSE_XGB_Y,RMSE_XGB_N))

train_results <- data.frame(model=c('xgbY','xgbN'),
                            rmse=c(RMSE_T_XGB_Y,RMSE_T_XGB_N))

write.csv(test_results, 'xgb_results2.csv')
write.csv(train_results, 'xgb_results_train2.csv')

