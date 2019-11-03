library(tidyverse)
library(tensorflow)
library(ggplot2)
library(nnet)
library(keras)
library(kerasR)

mnist <- dataset_mnist(path = "mnist.npz")

# train
train.X <- k_flatten(mnist$train$x/255)
train.y <- to_categorical(mnist$train$y, num_classes = 10)
# test

test.X <- k_flatten(mnist$test$x/255)
test.y <- to_categorical(mnist$test$y, num_classes = 10)

model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  train.X, train.y, 
  epochs = 3, batch_size = 128,
  steps_per_epoch = 1000,
  validation_split = 0.2
)
