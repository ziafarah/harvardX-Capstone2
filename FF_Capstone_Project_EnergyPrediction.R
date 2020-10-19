# --------------------------------------------------------------------------------
#
# Title: HarvardX PH125.9x Data Science: Capstone Project - Appliances Energy Prediction
# Author: Farah Fauzia
# Date: October 2020
#
# --------------------------------------------------------------------------------

# Set digit options
options(digits = 3)

# Load necessary package
if(!require(tidyverse)) install.packages("tidyverse", dependencies = TRUE)
library(tidyverse)

# Download dataset
energydata <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"), 
                       header=TRUE, stringsAsFactors=FALSE)

# Renames the column
cnames <- c("Date_Time", "Appliances", "Lights", "T_kitchen", "H_kitchen", 
            "T_living", "H_living", "T_laundry", "H_laundry", "T_office", "H_office", 
            "T_bathroom", "H_bathroom", "T_outNorth", "H_outNorth", "T_ironRoom",
            "H_ironRoom", "T_teenRoom2", "H_teenRoom2", "T_parentRoom", "H_parentRoom",
            "T_outside", "Pressure", "H_outside", "WindSpeed", "Visibility", "T_dewPoint", 
            "RandVar1", "RandVar2")
colnames(energydata) <- cnames

# --------------------------------------------------------------------------------
#
# 2. EXPLANOTARY DATA ANALYSIS
#
# --------------------------------------------------------------------------------
# 2.1. INITIAL EXPLORATION
# --------------------------------------------------------------------------------

# Check any missing data
any(is.na(energydata)) # Result: FALSE

# Display first 6 lines of dataset
head(energydata)
# Structure of dataset

str(energydata)
# Results: 19,753 observations of 29 variables

# --------------------------------------------------------------------------------
# 2.2. VISUALIZATION
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Checking distribution of energy consumption over time
# --------------------------------------------------------------------------------

# Check data class of [Date_Time] variable
class(energydata$Date_Time) # class: character

# Load/install necessary package
if(!require(lubridate)) install.packages("lubridate", dependencies = TRUE)
library(lubridate)

# Convert [Date_Time] into date-time format and put into new column 
dateplot <- energydata %>% mutate(Date = as.Date(energydata$Date_Time)) %>% 
  select(Appliances, Date)

# Recheck the class of converted [Date] column
str(dateplot) # class: Date, format

# Visualize energy consumption distribution over time
dateplot %>% ggplot(aes(x = Date, y = Appliances)) +
  geom_line(color = "#9970AB", size = 0.75) + 
  labs(title = "Appliances Energy Consumption over Time",
       x = "Date", y = "Energy Consumption (Wh)") + theme_bw()

# Result: no particular trend in time series
# This project would deal with regression problem based on other available features

# --------------------------------------------------------------------------------
# Checking frequency of appliance energy consumption
# --------------------------------------------------------------------------------

# Mean of appliance energy consumption
mean <- mean(energydata$Appliances)
# Result: 97.69 Wh

# Visualize energy consumption frequency 
energydata %>% ggplot(aes(x = Appliances)) + 
  geom_histogram(bins = 50, color = "white", fill = "#9970AB") +
  geom_vline(xintercept = mean, col = "#1B7837", linetype = "dashed") +
  labs(title = "Appliances Energy Consumption Frequency",
       x = "Energy Consumption (Wh)", y = "Frequency") + theme_bw()


# --------------------------------------------------------------------------------
# Checking distribution of temperature features
# --------------------------------------------------------------------------------

# Group all temperature features 
tempvar <- energydata %>% 
  select(T_kitchen, T_living, T_laundry, T_office, T_bathroom, T_outNorth, T_ironRoom, 
         T_teenRoom2, T_parentRoom, T_outside, T_dewPoint)


# Visualize distribution of temperature features
ggplot(gather(tempvar), aes(value)) + 
  geom_histogram(aes(y=..density..), bins = 20, color = "white", fill = "#9970AB") + 
  geom_density(alpha = 0.2, fill = "#E7D4E8", colour = "#1B7837") +
  labs(x = "Temperature (Celcius)", y = "Frequency") + theme_bw() +
  facet_wrap(~key, scales = "free", ncol = 3) 


# --------------------------------------------------------------------------------
# Checking distribution of humidity features
# --------------------------------------------------------------------------------

# Group all humidity features
humvar <- energydata %>% 
  select(H_kitchen, H_living, H_laundry, H_office, H_bathroom, H_outNorth, H_ironRoom, 
         H_teenRoom2, H_parentRoom, H_outside) 


# Visualize distribution of humidity features
ggplot(gather(humvar), aes(value)) + 
  geom_histogram(aes(y=..density..), bins = 20, color = "white", fill = "#9970AB") + 
  geom_density(alpha = 0.2, fill = "#E7D4E8", color = "#1B7837") + 
  labs(x = "Humidity (%)", y = "Frequency") + theme_bw() +
  facet_wrap(~key, scales = "free", ncol = 3)

# --------------------------------------------------------------------------------
# Checking distribution of other features
# --------------------------------------------------------------------------------

# Group other features aside from temperature and humidity
othervar <- energydata %>% 
  select(Lights, Pressure, WindSpeed, Visibility, RandVar1, RandVar2) 


# Visualize distribution of other features
othervar_label <- as_labeller(c(Lights = "Lights Consumption (Wh)", 
                                Pressure = "Pressure (mmHg)", 
                                Visibility = "Visibility (km)",
                                RandVar1 = "Random Variable 1",
                                RandVar2 = "Random Variable 2",
                                WindSpeed = "Wind Speed (m/s)"))
ggplot(gather(othervar), aes(value)) + 
  geom_histogram(aes(y=..density..), bins = 20, color = "white", fill = "#9970AB") + 
  geom_density(alpha = 0.2, fill = "#E7D4E8", color = "#1B7837") +
  labs(x = "Feature", y = "Frequency") + theme_bw() +
  facet_wrap(~key, scales = "free", labeller = othervar_label, ncol = 2) 

# --------------------------------------------------------------------------------
# 2.3. FEATURES REDUCTION
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Identifying features with few non-unique values
# --------------------------------------------------------------------------------

# Load/install necessary package
if(!require(caret)) install.packages("caret", dependencies = TRUE)
if(!require(kableExtra)) install.packages("kableExtra", dependencies = TRUE)
library(caret)
library(kableExtra)


# Identify near zero variance of the features
nearZeroVar(energydata[, 3:29], saveMetrics = TRUE)

# Result: no recommended features to be reduced 
# use other approach: variable correlation analysis

# --------------------------------------------------------------------------------
# Generating variable correlation matrix of dataset
# --------------------------------------------------------------------------------

# Convert non-numeric variables to numeric and remove [Date_Time] column
energydata2 <- energydata %>% mutate(Appliances = as.numeric(Appliances), 
                                     Lights = as.numeric(Lights)) %>% select(-Date_Time)

# Check whether all variables already have numeric class
is.numeric(as.matrix(energydata2))
# Result: TRUE

# Generate correlation matrix of the variables
cor <- cor(energydata2) #using Pearson coefficient by default

# Show correlation matrix result table 
# (Remove '#' to run the line if necessary):
# round(cor,2)

# Load/install necessary packages
if(!require(corrplot)) install.packages("corrplot", dependencies = TRUE)
if(!require(RColorBrewer)) install.packages("RColorBrewer", dependencies = TRUE)

library(corrplot)
library(RColorBrewer)


# Visualize correlation matrix between variables
corrplot(cor, method="ellipse", type="lower", order="hclust", 
         tl.col="black", tl.srt=45,
         col=brewer.pal(n=8, name="PRGn")) # using color-blind-friendly palette 

# --------------------------------------------------------------------------------
# Evaluating relationship between certain pairs of features
# --------------------------------------------------------------------------------

# Load/install necessary packages
if(!require(corrr)) install.packages("corrr", dependencies = TRUE)
library(corrr)

# Generate correlation matrix of dataset using corrr package
cor_tar <- correlate(energydata2, diagonal = NA) # using Pearson coefficient by default

# Show correlation table of all features with target [Appliances] variable 
# (Remove '#' to run the line if necessary):
# cor_tar %>% rearrange() %>% focus(Appliances) %>% fashion(leading_zeros = TRUE)

# Visualize correlation between features with target [Appliances] variable  
cor_tar %>%
  focus(Appliances) %>%
  mutate(rowname = reorder(rowname, Appliances)) %>%
  ggplot(aes(rowname, Appliances)) + coord_flip() +
  geom_col(color = "white", fill = "#9970AB") + theme_bw() +
  labs(title = "Correlation of Appliances Energy Consumption with the Features",
       y = "Correlation with Appliances Energy Consumption", x = "Features")

# Show table of highly-correlated variables of dataset 
cor_tar %>%  
  gather(-rowname, key = "colname", value = "cor") %>% 
  filter(abs(cor) > 0.9)

# remove 5 features: [Visibility] [RandVar1] [RandVar2] [T_parentRoom] [T_outNorth]

# --------------------------------------------------------------------------------
# Removing features based on previous analysis
# --------------------------------------------------------------------------------

# Remove 5 features from dataset
endatared <- energydata2 %>% select(-RandVar1, -RandVar2, -Visibility, 
                                    -T_parentRoom, -T_outNorth)
# Check dimension of reduced dataset
dim(endatared)
# Result: final 23 variables (22 features + 1 target) from original 27 variables

# --------------------------------------------------------------------------------
#
# 3. MODELING APPROACHES
#
# --------------------------------------------------------------------------------
# 3.1. PREPARATION
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Creating train and test sub-dataset
# --------------------------------------------------------------------------------

# Create test set with 25% of the dataset 
set.seed(1306, sample.kind = "Rounding")
test_index <- createDataPartition(y = endatared$Appliances, 
                                  times = 1, p = 0.25, list = FALSE)

endata_train <- endatared[-test_index,]
endata_test <- endatared[test_index,]

# Check dimension of train and test set
dimtrain <- dim(endata_train)
dimtrain
# Result: Train set contains 14,799 observation and 23 variables

dimtest <- dim(endata_test)
dimtest
# Result: Test set contains 4,936 observation and 23 variables

# --------------------------------------------------------------------------------
# Defining control parameter on train function of Caret package
# --------------------------------------------------------------------------------

# Create fit control for models
control <- trainControl(method = "cv", # cross-validation method
                        number = 5) # by 5 folds

# --------------------------------------------------------------------------------
# 3.2. EVALUATION METRICS & BENCHMARK
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Defining loss function RMSE & MAE as evaluation metrics
# --------------------------------------------------------------------------------

# Create function to calculate RMSE
RMSE <- function(true, pred){
  sqrt(mean((true - pred)^2))
}

# Create function to calculate MAE
MAE <- function(true, pred) {
  mean(abs((true - pred)))
}

# --------------------------------------------------------------------------------
# Defining benchmark result
# --------------------------------------------------------------------------------

# Benchmark work: (Candanedo, L.M., et al, 2017) 
# http://dx.doi.org/10.1016/j.enbuild.2017.01.083

# Store benchmark result:
benchmark_model <- bind_rows(tibble(model = "Benchmark (GBM)",
                              "RMSE (Train)" = 17.56,
                              "MAE (Train)" = 11.97,
                              "RMSE (Test)" = 66.65, 
                              "MAE (Test)" = 35.22))
benchmark_model %>% 
  kbl(col.names = c("Model", "RMSE", "MAE", "RMSE", "MAE"),
      align = "c", booktabs = TRUE) %>%
  kable_styling() %>%
  add_header_above(c(" " = 1, "Train" = 2, "Test" = 2), bold = TRUE)

# --------------------------------------------------------------------------------
# Enabling parallel computing
# --------------------------------------------------------------------------------

# Load/install necessary package
if(!require(doParallel)) install.packages("doParallel", dependencies = TRUE)
library(doParallel)

# Setup parallel computing before running the models
cl <- makePSOCKcluster(4) # using 4 CPU cores to use
registerDoParallel(cl)

# --------------------------------------------------------------------------------
# 3.3. LINEAR REGRESSION MODEl
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Multiple Linear Regression (MLR) Model
# --------------------------------------------------------------------------------

set.seed(1306, sample.kind = "Rounding")

timeStart <- proc.time() # measure start time

# Train the model
mlr <- train(Appliances ~ .,
             data = endata_train,
             method = "lm",
             metric = "RMSE",
             trControl = control)

time_mlr <- proc.time() - timeStart # measure elapsed run time

# Generate residual plot for MLR model
res_mlr <- resid(mlr)
plot(endata_train$Appliances, res_mlr,
     ylab = "Residuals", xlab = "Appliances Energy Consumption")
abline(0,0)

# Create models' evaluation table
mlr_results <- bind_rows(tibble(Model = "Multiple Linear Regression",
                                "RMSE" = getTrainPerf(mlr)$TrainRMSE,
                                "MAE" = getTrainPerf(mlr)$TrainMAE,
                                "R-squared" = getTrainPerf(mlr)$TrainRsquared,
                                "Train Time (s)" = time_mlr[[3]]))

# Evaluation for MLR model
mlr_results %>% 
  kbl(align = "c", booktabs = TRUE) %>%
  kable_styling()


# --------------------------------------------------------------------------------
# 3.4. DISTANCE-BASED MODEL
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# k-Nearest Neighbor (KNN) Model
# --------------------------------------------------------------------------------

set.seed(1306, sample.kind = "Rounding")

timeStart <- proc.time() # measure start time

# Train the model
knn <- train(Appliances ~ .,
             data = endata_train,
             method = "knn",
             preProcess = "range", # pre-process data by normalization
             metric = "RMSE",
             trControl = control,
             tuneLength = 10) # possible values to test in auto-tuning

time_knn <- proc.time() - timeStart # measure elapsed run time

# Get tuning parameter that give best performance
knn$bestTune 
# Result: [k = 5]


# Store result to models' evaluation table
knn_results <- bind_rows(tibble(Model = "k-Nearest Neighbor (KNN)",
                                "RMSE" = getTrainPerf(knn)$TrainRMSE,
                                "MAE" = getTrainPerf(knn)$TrainMAE,
                                "R-squared" = getTrainPerf(knn)$TrainRsquared,
                                "Train Time (s)" = time_knn[[3]]))

# Evaluation for KNN model
knn_results %>% 
  kbl(align = "c", booktabs = TRUE) %>%
  kable_styling()


# --------------------------------------------------------------------------------
# 3.5. NEURAL NETWORK MODEL
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Artificial Neural Network (ANN) Model
# --------------------------------------------------------------------------------

# Preparation
set.seed(1306, sample.kind = "Rounding")
timeStart <- proc.time() # measure start time

# Train the model
ann <- train(Appliances ~ .,
             data = endata_train,
             method = "nnet",
             metric = "RMSE",
             preProcess = c("center", "scale"), # pre-process data by standardization
             trControl = control,
             tuneLength = 3, # possible values length to test in auto-tuning
             linout = TRUE) # indicating running model for numerical regression problem
time_ann <- proc.time() - timeStart # measure elapsed run time

# Get tuning parameter that give best performance
ann$bestTune
# Result: [size = 5] [decay = 0]


# Store result to models' evaluation table
ann_results <- bind_rows(tibble(Model = "Artificial Neural Network (ANN)",
                                "RMSE" = getTrainPerf(ann)$TrainRMSE,
                                "MAE" = getTrainPerf(ann)$TrainMAE,
                                "R-squared" = getTrainPerf(ann)$TrainRsquared,
                                "Train Time (s)" = time_knn[[3]]))
# evaluation for ANN model
ann_results %>% 
  kbl(align = "c", booktabs = TRUE) %>%
  kable_styling()


# --------------------------------------------------------------------------------
# 3.6. TREE-BASED MODEL
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# 3.6.1. RANDOM FOREST (RF) Model
# --------------------------------------------------------------------------------
# (Warning: training this model may take several minutes)

# Load/install necessary package
if(!require(randomForest)) install.packages("randomForest", dependencies = TRUE)
library(randomForest)

# Define tuning grid
rf_grid <- expand.grid(mtry = 1:5) # define grid search range

# Preparation
set.seed(1306, sample.kind = "Rounding")
timeStart <- proc.time() # measure start time

# Train the model
rf <- train(Appliances ~ .,
            data = endata_train,
            method = "rf",
            metric = "RMSE",
            trControl = control,
            tuneGrid = rf_grid)
time_rf <- proc.time() - timeStart # measure elapsed run time

# Get tuning parameter that give best performance
rf$bestTune
# Result: [mtry = 3]

# Store result to models' evaluation table
rf_results <- bind_rows(tibble(Model = "Random Forest (RF)",
                               "RMSE" = getTrainPerf(rf)$TrainRMSE,
                               "MAE" = getTrainPerf(rf)$TrainMAE,
                               "R-squared" = getTrainPerf(rf)$TrainRsquared,
                               "Train Time (s)" = time_rf[[3]]))

# Evaluation for RF model
rf_results %>% 
  kbl(align = "c", booktabs = TRUE) %>%
  kable_styling()


# --------------------------------------------------------------------------------
# 3.6.2. GRADIENT BOOSTING MACHINE (GBM) Model
# --------------------------------------------------------------------------------

# Load/install necessary package
if(!require(gbm)) install.packages("gbm", dependencies = TRUE)
library(gbm)

# Preparation
set.seed(1306, sample.kind = "Rounding")
timeStart <- proc.time() # measure start time

# Train the model
gbm <- train(Appliances ~ .,
             data = endata_train,
             method = "gbm",
             metric = "RMSE",
             trControl = control,
             tuneLength = 10, # possible values length to test in auto-tuning
             verbose = FALSE) # supress the output
time_gbm <- proc.time() - timeStart # measure elapsed run time

# Get tuning parameter that give best performance
gbm$bestTune
# Result: [n.trees = 500] [interaction.depth = 10]
# [shrinkage = 0.1] [n.minobsinnode = 10]


# Evaluation for GBM model
gbm_results <- bind_rows(tibble(Model = "Gradient Boosting Machine (GBM)",
                                "RMSE" = getTrainPerf(gbm)$TrainRMSE,
                                "MAE" = getTrainPerf(gbm)$TrainMAE,
                                "R-squared" = getTrainPerf(gbm)$TrainRsquared,
                                "Train Time (s)" = time_gbm[[3]]))
gbm_results %>% 
  kbl(align = "c", booktabs = TRUE) %>%
  kable_styling()


# --------------------------------------------------------------------------------
# 3.6.3. eXtreme Gradient Boosting (XGBoost) Model
# --------------------------------------------------------------------------------

# Load/install necessary package
if(!require(xgboost)) install.packages("xgboost", dependencies = TRUE)
library(xgboost)

# --------------------------------------------------------------------------------
# A) Auto-tuned XGBoost
# --------------------------------------------------------------------------------

# Preparation
set.seed(1306, sample.kind = "Rounding")
timeStart <- proc.time() # measure start time

# Train the model
xgb0 <- train(Appliances ~ .,
              data = endata_train,
              method = "xgbTree",
              metric = "RMSE",
              trControl = control,
              tuneLength = 5) # possible values length to test in auto-tuning
time_xgb0 <- proc.time() - timeStart # measure elapsed run time

# Get tuning parameter that give best performance
xgb0$bestTune
# Result: [nrounds  = 250] [max_depth  = 5] [eta = 0.3] [gamma = 0] 
# [colsample_bytree = 0.8] [min_child_weight = 1] [subsample = 1]

# Evaluation for auto-tuned XGBoost model
xgb0_results <- bind_rows(tibble(Model = "XGBoost (Auto-tuned)",
                                 "RMSE" = getTrainPerf(xgb0)$TrainRMSE,
                                 "MAE" = getTrainPerf(xgb0)$TrainMAE,
                                 "R-squared" = getTrainPerf(xgb0)$TrainRsquared,
                                 "Train Time (s)" = time_xgb0[[3]]))

xgb0_results %>%
  kbl(align = "c", booktabs = TRUE) %>%
  kable_styling()

# --------------------------------------------------------------------------------
# B) Optimized XGBoost
# --------------------------------------------------------------------------------
#
# (Notes: this R file only provides the best tune grid after 5-step optimization
# Refer the full detail of tuning optimization steps on report or .Rmd file)
#
# --------------------------------------------------------------------------------

# Define tune grid for Optimized XGBoost model
tune_xgbfinal <- expand.grid(
  nrounds = 20000,
  max_depth = 6,
  eta = 0.025,
  gamma = 0.75,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  subsample = 0.75)

# Preparation
set.seed(1306, sample.kind = "Rounding")
timeStart <- proc.time() # measure start time

# Train the model
xgb_final <- train(Appliances ~ .,
                   data = endata_train,
                   method = "xgbTree",
                   metric = "RMSE",
                   trControl = control,
                   tuneGrid = tune_xgbfinal)
time_xgbf <- proc.time() - timeStart # measure elapsed run time

# Evaluation for optimized XGBoost model
xgbf_results <- bind_rows(tibble(Model = "XGBoost (Optimized)",
                                 "RMSE" = getTrainPerf(xgb_final)$TrainRMSE,
                                 "MAE" = getTrainPerf(xgb_final)$TrainMAE,
                                 "R-squared" = getTrainPerf(xgb_final)$TrainRsquared,
                                 "Train Time (s)" = time_xgbf[[3]]))


# --------------------------------------------------------------------------------
#
# 4. MODEL TESTING & PERFORMANCE
#
# --------------------------------------------------------------------------------
# Summary of all evaluated models on train set
# --------------------------------------------------------------------------------
train_results <- bind_rows(mlr_results, knn_results, ann_results, 
                           rf_results, gbm_results, xgb0_results, xgbf_results)
train_results %>% 
  kbl(align = "c", booktabs = TRUE) %>%
  kable_styling()

# Result: Optimized XGBoost give the best performance, followed by RF model
# Two models (RF and Optimized XGBoost will be tested)

# --------------------------------------------------------------------------------
# Evaluating RF model on test set
# --------------------------------------------------------------------------------

# Test RF model using test set
set.seed(1306, sample.kind = "Rounding")
pred_rf <- predict(rf, newdata = endata_test)

# Calculate evaluation metrics
rmse_rf <- RMSE(endata_test$Appliances, pred_rf)
mae_rf <- MAE(endata_test$Appliances, pred_rf)

# --------------------------------------------------------------------------------
# Evaluating optimized XGBoost model on test set
# --------------------------------------------------------------------------------

# Test Optimized XGBoost model using test set
set.seed(1306, sample.kind = "Rounding")
pred_xgb_final <- predict(xgb_final, newdata = endata_test)

# Calculate evaluation metrics
rmse_xgb_final <- RMSE(endata_test$Appliances, pred_xgb_final)
mae_xgb_final <- MAE(endata_test$Appliances, pred_xgb_final)

# --------------------------------------------------------------------------------
# Summarizing performance & benchmark analysis
# --------------------------------------------------------------------------------

# Summary for RF model performance on train and test set
best_model_rf <- bind_rows(tibble(model = "Random Forest",
                                  "RMSE (Train)" = getTrainPerf(rf)$TrainRMSE,
                                  "MAE (Train)" = getTrainPerf(rf)$TrainMAE,
                                  "RMSE (Test)" = rmse_rf, 
                                  "MAE (Test)" = mae_rf))

# Summary for Optimized XGboost model performance on train and test set
best_model_xgbf <- bind_rows(tibble(model = "Optimized XGBoost",
                               "RMSE (Train)" = getTrainPerf(xgb_final)$TrainRMSE,
                               "MAE (Train)" = getTrainPerf(xgb_final)$TrainMAE,
                               "RMSE (Test)" = rmse_xgb_final, 
                               "MAE (Test)" = mae_xgb_final))

# Benchmarking summary
best_compare <- bind_rows(benchmark_model, best_model_rf, best_model_xgbf)

best_compare %>% 
  kbl(col.names = c("Model", "RMSE", "MAE", "RMSE", "MAE"),
      align = "c", booktabs = TRUE,
      caption = "Benchmarking Summary") %>%
  kable_styling(font_size = 9, latex_options = "hold_position") %>%
  add_header_above(c(" " = 1, "Train" = 2, "Test" = 2), bold = TRUE) %>%
  row_spec(0, bold = TRUE)

# Result: Optimized XGBoost is the best model for this project

# --------------------------------------------------------------------------------
# Visualizing variable importance of best model
# --------------------------------------------------------------------------------
plot(varImp(xgb_final))

