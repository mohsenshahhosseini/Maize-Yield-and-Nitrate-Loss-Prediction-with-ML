##################################################
## Author: Mohsen
## Maize Yield and Nitrate Loss prediction with ML
## Random forest
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
  filter(uniqueid != "SEPAC") %>%
  select(uniqueid:PAWC, NLoss, maize_yield) %>%
  filter(maize_yield != 0) %>%
  as.data.frame() 

dmy <- dummyVars(~sowTime+cultivar+Nrate+Nstrategy+residue+residueRemoval+
                   covercrop+initsoilN+waterTable+texture, data=maizeData)
maizeData_oh <- data.frame(predict(dmy, maizeData))

maizeData <- maizeData %>%
  select(uniqueid:sowTime_doy,cultivar_mg,Nrate_kg,residue_kg,initsoilN_kg,
         waterTable_mm,OC,PAWC,maize_yield,NLoss) %>%
  bind_cols(maizeData_oh)

weather <- read.csv("yyy.csv")

weather2 <- weather %>%
  group_by(uniqueid,year) %>%
  summarise_at(c("mint_avg1","mint_avg2","mint_avg3","mint_avg4","mint_avg5",
                 "maxt_avg1","maxt_avg2","maxt_avg3","maxt_avg4","maxt_avg5",
                 "gdd10_sum1","gdd10_sum2","gdd10_sum3","gdd10_sum4","gdd10_sum5",
                 "rain_sum1","rain_sum2","rain_sum3","rain_sum4","rain_sum5",
                 "radn_sum1","radn_sum2","radn_sum3","radn_sum4","radn_sum5",
                 "rainDays_num1","rainDays_num2","rainDays_num3",
                 "rainDays_num4","rainDays_num5",
                 "frostDays_num1","frostDays_num2","frostDays_num3",
                 "frostDays_num4","frostDays_num5"),sum) %>%
  mutate(year = as.factor(year))

maizeData <- maizeData %>%
  left_join(weather2)

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



# Random Forest -----------------------------------------------------------

## ---------- Maize Yield ---------- ## 

years_cut <- c(1988,1993,1998,2003,2008,2013)
RF_parameters_Y <- data.frame(mtry=c(12,14,16,18,20,22,24,26,28,30))
errorRFy = matrix(ncol=4 , nrow = 10)

for (i in 1:5) {
  valid_Y <- training_Y %>%
    filter(year %in% c(years_cut[i]:(years_cut[i+1]-1))) %>%
    select(-year, -uniqueid)
  
  train_Y <- training_Y %>%
    filter(year %in% c((years_cut[i]-5):(years_cut[i]-1))) %>%
    select(-year, -uniqueid)
  
  for (j in 1:10) {
    RF_Y <- ranger(maize_yield~., data=train_Y, mtry=RF_parameters_Y$mtry[j])
    RF_Y
    
    RF_Y.predictak <- predict(RF_Y,valid_Y)
    errorRFy[j,i] = rmse(actual = valid_Y$maize_yield,
                         predicted = RF_Y.predictak$predictions)
  }
}

errorRFy <- as.data.frame(errorRFy)
errorRFy$mean <- rowMeans(errorRFy)
position_rfy = which.min(errorRFy$mean)
position_rfy

training_Y1 <- training_Y %>%
  select(-year, -uniqueid)

RF_Y_final <- ranger(maize_yield~., data=training_Y1, 
                     mtry=RF_parameters_Y$mtry[position_rfy],
                     importance = "impurity")

saveRDS(RF_Y_final, 'model_rfY.rds')

RF_Y.predict <- predict(RF_Y_final,test_Y)
df_RF_Y <- data.frame(predicted = RF_Y.predict$predictions, actual = test_Y$maize_yield)
df_RF_Y$difference <- df_RF_Y$predict-df_RF_Y$actual
RMSE_RF_Y <- rmse(predicted = df_RF_Y$predict, actual = df_RF_Y$actual)


## Training Error
PredT.RF_Y <- predict(RF_Y_final, training_Y1)
DDD_Y <- data.frame(predicted = PredT.RF_Y$predictions, actual = training_Y1$maize_yield)
DDD_Y$difference <- DDD_Y$predict-DDD_Y$actual
RMSE_T_RF_Y <- rmse(predicted = DDD_Y$predict, actual = DDD_Y$actual)



## ---------- Nitrogen Loss ---------- ##

years_cut <- c(1988,1993,1998,2003,2008,2013)
RF_parameters_N <- data.frame(mtry=c(12,14,16,18,20,22,24,26,28,30))
errorRFn = matrix(ncol=4 , nrow = 10)

for (i in 1:5) {
  valid_N <- training_N %>%
    filter(year %in% c(years_cut[i]:(years_cut[i+1]-1))) %>%
    select(-year, -uniqueid)
  
  train_N <- training_N %>%
    filter(year %in% c((years_cut[i]-5):(years_cut[i]-1))) %>%
    select(-year, -uniqueid)
  
  for (j in 1:10) {
    RF_N <- ranger(NLoss~., data=train_N, mtry=RF_parameters_N$mtry[j])
    RF_N
    
    RF_N.predictak <- predict(RF_N,valid_N)
    errorRFn[j,i] = rmse(actual = valid_N$NLoss,
                         predicted = RF_N.predictak$predictions)
  }
}

errorRFn <- as.data.frame(errorRFn)
errorRFn$mean <- rowMeans(errorRFn)
position_rfn = which.min(errorRFn$mean)
position_rfn

training_N1 <- training_N %>%
  select(-year, -uniqueid)

RF_N_final <- ranger(NLoss~., data=training_N1, 
                     mtry=RF_parameters_N$mtry[position_rfn],
                     importance = "impurity")

saveRDS(RF_N_final, 'model_rfN.rds')

RF_N.predict <- predict(RF_N_final,test_N)
df_RF_N <- data.frame(predicted = RF_N.predict$predictions, actual = test_N$NLoss)
df_RF_N$difference <- df_RF_N$predict-df_RF_N$actual
RMSE_RF_N <- rmse(predicted = df_RF_N$predict, actual = df_RF_N$actual)


## Training Error
PredT.RF_N <- predict(RF_N_final, training_N1)
DDD_N <- data.frame(predicted = PredT.RF_N$predictions, actual = training_N1$NLoss)
DDD_N$difference <- DDD_N$predict-DDD_N$actual
RMSE_T_RF_N <- rmse(predicted = DDD_N$predict, actual = DDD_N$actual)




### ----------- RESULTS ------------ ###

test_results <- data.frame(model=c('RfY','RfN'),
                           rmse=c(RMSE_RF_Y,RMSE_RF_N))

train_results <- data.frame(model=c('RfY','RfN'),
                            rmse=c(RMSE_T_RF_Y,RMSE_T_RF_N))

write.csv(test_results, 'rf_results.csv')
write.csv(train_results, 'rf_results_train.csv')


