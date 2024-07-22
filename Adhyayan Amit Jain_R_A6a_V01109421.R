
# Load libraries
# Initialize Keras for LSTM
library(keras)
library(tensorflow)
library(reticulate)
library(tidyverse)
library(quantmod)
library(forecast)
library(fable)
library(fabletools)
library(tseries)
library(keras)
library(caret)
library(randomForest)
library(neuralnet)


# Get NVIDIA stock data
ticker <- "NVDA"
data <- getSymbols(ticker, src = "yahoo", from = "2021-04-01", to = "2024-03-31", auto.assign = FALSE)

# Convert to data.frame and select the target variable Adjusted Close
df <- data %>% 
  data.frame() %>% 
  rownames_to_column(var = "Date") %>% 
  mutate(Date = as.Date(Date)) %>% 
  select(Date, Adjusted = NVDA.Adjusted)

# Check and handle missing values
df <- na.omit(df)

# Plot the time series
ggplot(df, aes(x = Date, y = Adjusted)) +
  geom_line() +
  labs(title = "NVIDIA Adj Close Price", x = "Date", y = "Adj Close Price")

# Decompose the time series
# Convert to time series object with daily frequency
ts_data <- ts(df$Adjusted, frequency = 252)  # For daily data
decomp <- stl(ts_data, s.window = "periodic")
autoplot(decomp) + labs(title = "Decomposition of Time Series")

# Aggregate to monthly data
monthly_data <- df %>%
  mutate(YearMonth = floor_date(Date, "month")) %>%
  group_by(YearMonth) %>%
  summarize(Adjusted = mean(Adjusted)) %>%
  as.data.frame()

# Split data into training and test sets
train_size <- floor(0.8 * nrow(monthly_data))
train_data <- monthly_data[1:train_size,]
test_data <- monthly_data[(train_size + 1):nrow(monthly_data),]

# Use Holt-Winters method on monthly data
holt_winters_model <- HoltWinters(ts(train_data$Adjusted, frequency = 12), seasonal = "multiplicative")

# Forecasting
forecast_holt_winters <- forecast(holt_winters_model, h = nrow(test_data))

# Plot forecast
autoplot(forecast_holt_winters) +
  autolayer(test_data$Adjusted, series = "Test Data") +
  labs(title = "Holt-Winters Forecast", x = "Date", y = "Adjusted Close Price")

# Check and handle missing values
missing_values <- sum(is.na(df))
print(paste("Missing values:", missing_values))
df <- na.omit(df)  # Remove rows with NA values
missing_values <- sum(is.na(df))
print(paste("Missing values after interpolation:", missing_values))

# ARIMA Model
arima_model <- auto.arima(train_data$Adjusted)
summary(arima_model)
forecast_data <- forecast(arima_model, h = 12)
autoplot(forecast_data) + labs(title = "Auto ARIMA Forecasting")

# Random Forest Model Preparation
df$Date <- as.numeric(as.Date(df$Date))
X <- df %>% select(Date)
y <- df$Adjusted
train_index <- 1:train_size
test_index <- (train_size + 1):nrow(df)

X_train_rf <- X[train_index, , drop = FALSE]
X_test_rf <- X[test_index, , drop = FALSE]
y_train_rf <- y[train_index]
y_test_rf <- y[test_index]

rf_model <- randomForest(X_train_rf, y_train_rf, ntree = 100)
train_predict_rf <- predict(rf_model, X_train_rf)
test_predict_rf <- predict(rf_model, X_test_rf)

# Plot the predictions
ggplot() +
  geom_line(aes(x = df$Date, y = df$Adjusted), color = 'black') +
  geom_line(aes(x = df$Date[train_index], y = train_predict_rf), color = 'blue') +
  geom_line(aes(x = df$Date[test_index], y = test_predict_rf), color = 'red') +
  labs(title = "Random Forest Model Predictions")

# Performance metrics for Random Forest
train_rmse_rf <- sqrt(mean((y_train_rf - train_predict_rf)^2))
test_rmse_rf <- sqrt(mean((y_test_rf - test_predict_rf)^2))
train_mae_rf <- mean(abs(y_train_rf - train_predict_rf))
test_mae_rf <- mean(abs(y_test_rf - test_predict_rf))
train_r2_rf <- cor(y_train_rf, train_predict_rf)^2
test_r2_rf <- cor(y_test_rf, test_predict_rf)^2

cat('Train RMSE (RF):', train_rmse_rf, '\n')
cat('Test RMSE (RF):', test_rmse_rf, '\n')
cat('Train MAE (RF):', train_mae_rf, '\n')
cat('Test MAE (RF):', test_mae_rf, '\n')
cat('Train R-squared (RF):', train_r2_rf, '\n')
cat('Test R-squared (RF):', test_r2_rf, '\n')

# ANN Model Preparation
scaled_features <- predict(preProcess(df %>% select(Adjusted), method = c("range")), df %>% select(Adjusted))
target <- df$Adjusted
split_index <- floor(0.8 * nrow(df))

X_train_ann <- scaled_features[1:split_index, , drop = FALSE]
y_train_ann <- target[1:split_index]
X_test_ann <- scaled_features[(split_index + 1):nrow(df), , drop = FALSE]
y_test_ann <- target[(split_index + 1):nrow(df)]

ann_model <- neuralnet(Adjusted ~ ., data = as.data.frame(cbind(X_train_ann, Adjusted = y_train_ann)), hidden = c(10, 10), linear.output = TRUE)
ann_predictions <- predict(ann_model, as.data.frame(X_test_ann))

# Performance metrics for ANN
ann_rmse <- sqrt(mean((y_test_ann - ann_predictions)^2))
ann_mae <- mean(abs(y_test_ann - ann_predictions))
ann_r2 <- cor(y_test_ann, ann_predictions)^2

cat('ANN RMSE:', ann_rmse, '\n')
cat('ANN MAE:', ann_mae, '\n')
cat('ANN R-squared:', ann_r2, '\n')

