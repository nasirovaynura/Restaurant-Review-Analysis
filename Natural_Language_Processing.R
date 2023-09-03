library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)
library(inspectdf)


# 1. Import nlpdata and get familiarized with it. ----

df <- read_csv("nlpdata.csv")

df %>% colnames()
df %>% glimpse()
df %>% inspect_na()

colnames(df) <- c("id", "review", "liked")
df$id <- df$id %>% as.character()


# 2. Define preprocessing function and tokenization function. ----

# Split data

set.seed(123)
split <- df$liked %>% sample.split(SplitRatio = 0.85)
train <- df %>% subset(split == T)
test <- df %>% subset(split == F)

it_train <- train$review %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$id,
         progressbar = F) 


vocab <- it_train %>% create_vocabulary()

vocab %>% 
  arrange(desc(term_count)) %>% 
  head(110) %>% 
  tail(10) 

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()
identical(rownames(dtm_train), train$id)


# 3. Model: Normal Nfold GLm. ----

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['liked']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,           
            maxit = 1000)             

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")


it_test <- test$review %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$id,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

preds <- predict(glmnet_classifier, dtm_test, type = 'response')[, 1]
glmnet:::auc(test$liked, preds) %>% round(2)


# 4. Predict model and remove Stopwords. ----

stop_words <- c("i", "you", "we", "they",
                "me", "him", "her", "them",
                "your", "yours", "his", 
                "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "by", "so",
                "from", "about", "to", "for", "of",
                "is", "are")


vocab <- it_train %>% create_vocabulary(stopwords = stop_words)


# 5. Create DTM for Training and Testing with new pruned vocabulary. ----

pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 3, 
                   doc_proportion_max = 0.9,
                   doc_proportion_min = 0.001)

pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10) 


# 6. Apply vectorizer. ----

vectorizer <- pruned_vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()


glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['liked']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 4,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[, 1]
glmnet:::auc(test$liked, preds) %>% round(2)


# 7. Give interpretation for model. ----

# Extract coefficients from the trained glmnet model
coefficients <- coef(glmnet_classifier)

# Identify important terms with non-zero coefficients
important_terms <- vocab$term[which(coefficients != 0)]

# Create a data frame to store important terms and their corresponding coefficients
important_terms_coeff <- data.frame(Term = important_terms, Coefficient = coefficients[coefficients != 0])

# Display the data frame containing important terms and coefficients
important_terms_coeff %>% view()

