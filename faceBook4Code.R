
## Facebook 4 Competition
## Hosted by Kaggle
## The goal of this competition was to determine whether an entity placing an online bid 
## is a human or a robot.
## The data is available on the Kaggle website at the url given below
## https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot
## This is a starter code in R. A derived version placed 9th on the private leaderboard.


setwd("C:/Kaggle/Facebook4/Data")


library(data.table)
require(bit64)
library(plyr)
library(reshape2)
library(caTools)
library(ROCR)

rm(list=ls())



## load train set
train  <-  read.csv("train.csv")

## load bids
bids  <-  fread("bids.csv")

bids$time  <-  log(bids$time)

## merge train set with bids
train <-  merge(train, bids, by="bidder_id", all.x =TRUE)

## transform data
### using melt and dcast functions in the plyr package

measure.var = setdiff(names(train),c("bidder_id","payment_account","address","outcome"))


## melt and aggregate train data
train$bid_id = as.character(train$bid_id)
train$time = as.character(train$time)
train<-  melt(train, id=c("bidder_id","payment_account","address","outcome"),
                     measure = measure.var)
train  <- dcast(train, bidder_id + payment_account + address + outcome ~ variable, 
                      fun.aggregate = function(x) {length(unique(x))})

######

train$outcome  <-  as.factor(train$outcome)

## use only data with no missing values 
ok  <-  complete.cases(train)
train  <-  train[ok, ]

# split into train  and validation sets
set.seed(123)
indx  <-  sample.split(train$outcome, SplitRatio = 0.8)

train.set  <-  train[indx, ]
val.set  <-  train[!indx, ]

features  <-  setdiff(names(train),c("bidder_id","payment_account","address","outcome"))

y  <-  train$outcome
levels(y)  <-  c(0,1) ## change levels to numeric. This is required for the xgboos trainer
y <-  as.numeric(as.character(y))
X <-  train[, features]
y.T  <-  val.set$outcome
X.T  <-  val.set[ , features]

X  <-  model.matrix(~. -1, X)
X.T  <-  model.matrix(~. -1, X.T)

###########  xgboost model ################

ETA  <-  0.1
NROUNDS  <-  100
DEPTH  <-  3
model.xgb <- xgboost(data = X, label= y,
                     objective = "binary:logistic",                    
                     eta = ETA, 
                     max.depth = DEPTH,
                     nrounds = NROUNDS,
                     metric="auc",
                     verbose = 0)               


pred.xgb = predict(model.xgb, newdata=X.T)

## evaluate model on validation set
xgb.perf = prediction(pred.xgb,y.T)
auc.xgb=as.numeric(performance(xgb.perf,"auc")@y.values)
auc.xgb


################## random forest model

model.rf  <-  randomForest(X, as.factor(y), n.trees=200)


# predict on val set
pred.rf = predict(model.rf, newdata=X.T, type="prob")
pred.rf = pred.rf[ , "1"]

rf.perf = prediction(pred.rf,y.T)
auc.rf=as.numeric(performance(rf.perf,"auc")@y.values)
auc.rf
