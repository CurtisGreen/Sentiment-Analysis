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
data$total = str_count(data$tweet_text, '\\s+')+1

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

# Tune radial model
tune.out = tune(svm,y~., data=dat[train,], kernel="radial", ranges = list(cost=10^(-1:1),gamma=c(0.5,1:4)))
best = tune.out$best.model
plot(best,dat[train,])

# Predict with radial
pred = predict(best,dat[-train,])
table(predict = pred, truth = dat[-train,]$y)


################## TF-IDF Sentiment ##################### 


library(tidyverse)
library(caret)
library(LiblineaR)

# Pos/neg ratio of data
counts = table(model$emotion)
colors = c("darkred", "darkgreen")
barplot(counts, main = "Emotion distribution", ylab = "Count",
        col = colors)

# Get all words in both dictionaries
all_words = parsed_tweets %>%
  inner_join(get_sentiments("afinn"))%>%
  inner_join(get_sentiments("bing"))

# Change text into binary
all_words = all_words[!duplicated(all_words$word),]
all_words$emotion[all_words$emotion == "Negative emotion"] = 0
all_words$emotion[all_words$emotion == "Positive emotion"] = 1
all_words$sentiment[all_words$sentiment == "negative"] = -1
all_words$sentiment[all_words$sentiment == "positive"] = 1

test = inner_join(parsed_tweets, all_words, by = "word")
colnames(test)[1] = "afinn_tfidf"
test$afinn_tfidf = 0
test_table = as.data.frame(table(test$word))
colnames(test_table)[1] = "word"
testerino = right_join(test, test_table, by = "word")

num_words = dim(all_words)[1]

testerino$weight = (1/testerino$total.x)*log2(num_words/testerino$Freq)
testerino$afinn_tfidf = testerino$weight * testerino$score
testerino$bing_tfidf = testerino$weight * as.numeric(testerino$sentiment)

weighted_model = cbind.data.frame(testerino$index.x, testerino$afinn_tfidf, testerino$bing_tfidf, as.numeric(testerino$emotion.y))
colnames(weighted_model) = c("index", "afinn", "bing", "emotion")

weighted_model = weighted_model %>%
  group_by(index = index %/% 1) %>%
  summarise(afinn = sum(afinn), bing = sum(bing))

data$index = as.numeric(data$index)
weighted_model = inner_join(data, weighted_model, by = "index")

weighted_model$emotion[weighted_model$emotion == "Negative emotion"] = 0
weighted_model$emotion[weighted_model$emotion == "Positive emotion"] = 1  

# Choose training set
data_size = dim(weighted_model)[1]
train_size = floor(data_size * .7)

dat = data.frame(x=cbind(weighted_model$afinn, weighted_model$bing), y=as.factor(weighted_model$emotion))
train = sample(data_size,train_size)

# Radial
svmradi = svm(y~.,data=dat[train,], kernel="radial", gamma=1, cost=1)
plot(svmradi, dat[train,])

# Predict with radial
pred = predict(svmradi,dat[-train,])
table(predict = pred, truth = dat[-train,]$y)

# Tune radial model
tune.out = tune(svm,y~., data=dat[train,], kernel="radial", ranges = list(cost=10^(-1:1),gamma=c(0.5,1:4)))
best = tune.out$best.model
plot(best,dat[train,])

# Predict with tuned radial
pred = predict(best,dat[-train,])
table(predict = pred, truth = dat[-train,]$y)
