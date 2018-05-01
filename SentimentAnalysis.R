# Open libraries
library(tidytext)
library(dplyr)
library(stringr) 
library(tidyr) 
library(ggplot2)


# https://www.hvitfeldt.me/2018/03/binary-text-classification-with-tidytext-and-caret/

##################### Naive Sentiment ##################### 


# Read CSV into R
data = read.csv("tweets.csv", stringsAsFactors = FALSE)

# Remove unneeded info
data = data[,-2]
colnames(data)[2] = "emotion"
data = data[data$emotion == "Negative emotion" | data$emotion == "Positive emotion",]
data = data %>% mutate(index = row_number())

# Seperate by word
parsed_tweets = tidytext::unnest_tokens(data, word, tweet_text)

# Afinn sentiment
afinn = parsed_tweets %>%
  inner_join(get_sentiments("afinn")) %>%
  group_by(index = index %/% 1) %>%
  summarise(sentiment = sum(score)) %>%
  mutate(method = "AFINN")

# Put theoretical/actual side-by-side
joined_tweets = inner_join(data, afinn)

# Compare actual vs calculated
tp = 0; fp = 0; tn = 0; fn = 0

for (row in 1:nrow(joined_tweets)){
  if (joined_tweets[row,]$emotion == "Negative emotion"){
    if (joined_tweets[row,]$sentiment < 0){
      tn = tn + 1
    }
    else{
      fn = fn + 1
    }
  }
  else{
    if (joined_tweets[row,]$sentiment > 0){
      tp = tp + 1
    }
    else{
      fp = fp + 1
    }
  }
}

# Compute accuracy stats
accuracy = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp + fn)
f1 = 2*(precision * recall)/(precision + recall)

print(paste("AFINN: Accuracy: ", accuracy, " Precision: ", precision,
            " Recall: ", recall, " F1: ", f1))

# Plot
bind_rows(afinn) %>%
  ggplot(aes(index, sentiment, fill = method)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~method, ncol = 1, scales = "free_y")


# Bing/NRC sentiment
bing_and_nrc = bind_rows(parsed_tweets %>%
        inner_join(get_sentiments("bing")) %>%
        mutate(method = "Bing et al."),
      parsed_tweets %>%
        inner_join(get_sentiments("nrc") %>%
        filter(sentiment %in% c("positive",
                                "negative"))) %>%
        mutate(method = "NRC")) %>%
  count(method, index = index %/% 1, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) 

bing_and_nrc = bing_and_nrc[!duplicated(bing_and_nrc$index),]

joined_tweets = inner_join(bing_and_nrc, joined_tweets, by = "index")
joined_tweets = joined_tweets[,c("sentiment.x", "sentiment.y", "emotion")]

# Compare actual vs calculated
tp = 0; fp = 0; tn = 0; fn = 0

for (row in 1:nrow(joined_tweets)){
  if (joined_tweets[row,]$emotion == "Negative emotion"){
    if (joined_tweets[row,]$sentiment.x < 0){
      tn = tn + 1
    }
    else{
      fn = fn + 1
    }
  }
  else{
    if (joined_tweets[row,]$sentiment.x > 0){
      tp = tp + 1
    }
    else{
      fp = fp + 1
    }
  }
}

# Compute accuracy stats
accuracy = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp + fn)
f1 = 2*(precision * recall)/(precision + recall)

print(paste("Bing/NCR: Accuracy: ", accuracy, " Precision: ", precision,
            " Recall: ", recall, " F1: ", f1))



##################### SVM Sentiment ##################### 


library("e1071")

# Setup data
set.seed(1)
model = joined_tweets[,c("sentiment.x", "sentiment.y", "emotion")]
model$emotion[model$emotion == "Negative emotion"] = 0
model$emotion[model$emotion == "Positive emotion"] = 1

# Choose training set
data_size = dim(model)[1]
train_size = floor(data_size * .7)

dat = data.frame(x=cbind(model$sentiment.x, model$sentiment.y), y=as.factor(model$emotion))
train = sample(data_size,train_size)

# Radial
svmradi = svm(y~.,data=dat[train,], kernel="radial", gamma=1, cost=1)
plot(svmradi, dat[train,])

# Predict with radial
pred = predict(svmradi,dat[-train,])
table(predict = pred, truth = dat[-train,]$y)

# Tune radial model
#tune.out = tune(svm,y~., data=dat[train,], kernel="radial", ranges = list(cost=10^(-1:3),gamma=c(0.5,1:4)))
#best = tune.out$best.model
#plot(best,dat[train,])

# Predict with best radial
#pred = predict(best,dat[-train,])
#table(predict = pred, truth = dat[-train,]$y)


################## TF-IDF Sentiment ##################### 


#library(tidyverse)
#library(caret)
#library(LiblineaR)
library(tm)
library(caret)
#library(plyr)

# Pos/neg ratio of data
counts = table(model$emotion)
colors = c("darkred", "darkgreen")
barplot(counts, main = "Emotion distribution", ylab = "Count",
        col = colors)
# Use tm library to setup TFIDF matrix
train_dtm = DocumentTermMatrix(Corpus(VectorSource(data$tweet_text)))
train_dtm_sparse = removeSparseTerms(train_dtm, .997)

# Apply weights based on TFIDF
unweighted_dtm = weightTfIdf(train_dtm_sparse, normalize = FALSE)
y = as.factor(data$emotion)
weighted_data = data.frame(y,as.matrix(unweighted_dtm))

# Do 10K cross validation in SVM
dat=weighted_data
k = 10
folds = sample(rep(1:k, length.out = nrow(dat)), nrow(dat))
dat_k = lapply(1:k, function(x){
  model = svm(y~., dat[folds != x, ], gamma = .5, cost = 10, kernel = "radial",probability = T)
  pred = predict(model, dat[folds == x, ])
  true = dat$y[folds == x]
  return(data.frame(pred = pred, true = true))
})

# Output confusion matrix
output = do.call(rbind, dat_k)
caret::confusionMatrix(output$pred, output$true)

