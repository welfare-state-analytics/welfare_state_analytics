dtm1 <- readRDS("dtm1.rds")

###############################################################################
###############################################################################
#Metod 3. Perplexity
#From (among others):
#Martin Ponweiser. 2012. Latent dirichlet allocation in r.

#Divide the corpus, build models with different number of topics on a training-set,
#see how well the models predict the held-out set. If the models does not predict well,
#it gets "perplexed" (who wouldn't be?) in terms of high entropy/low redundancy:
#the held-out dataset contains information we can't predict well with our model.
#Hence, high perplexity -> poor model. Low perplexity -> good model.
#The levels of perplexity are essentially
#corpus- but not model-specific.
#Find the model with the number of topics
#that have the lowest perplexity.

###############################################################################
###############################################################################

#Below script is ever so slightly adjusted from..:
#https://www.r-bloggers.com/cross-validation-of-topic-modelling/.
#Hats of to the r-blogger "Peter's stats stuff - R".

library(doParallel)
#ForEach is a smarter version of parallelization, since it does not split the data, but the processes.

burnin = 200
iter = 500
keep = 50 
delta = 0.1 #We need these stated individually, to call them inside the below function... #Delta most often refered to as Beta.

n <- nrow(dtm1) #We need this for splitting the sample

cluster <- makeCluster(detectCores(logical = TRUE) - 1) # leave one CPU to spare for stability...

registerDoParallel(cluster)

clusterEvalQ(cluster, {
  library(topicmodels)
})

folds <- 3 #For the overachievers: =>5.
splitfolds <- sample(1:5, n, replace = TRUE) #1:5 will result in (roughly) a 80/20 split.

#to inspect train and valid data-set
for(i in 1:folds){
  train_set <- dtm1[splitfolds != i , ]
  valid_set <- dtm1[splitfolds == i, ]}

clusterExport(cluster, c("dtm1", "burnin", "iter", "keep", "delta", "splitfolds", "folds", "candidate_k"))
#These are 'uploaded' to the clusters to be included in the calculations

#Below, parallelization below by the different number of topics: a processor is allocated a value
#of k, and does the cross-validation serially. Why (and why not over folds)?
#Because it is assumed there are more candidate values of k than there are cross-validation folds (k>folds),
system.time({
  results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
    k <- candidate_k[j]
    results_1k <- matrix(0, nrow = folds, ncol = 2)
    colnames(results_1k) <- c("k", "perplexity")
    for(i in 1:folds){
      train_set <- dtm1[splitfolds != i , ]
      valid_set <- dtm1[splitfolds == i, ]
      
      fitted <- LDA(train_set, k = k, method = "Gibbs",
                    control = list(burnin = burnin, iter = iter, keep = keep, delta = delta) )
      results_1k[i,] <- c(k, perplexity(fitted, newdata = valid_set))
    }
    return(results_1k)
  }
})
stopCluster(cluster)

results_perplexity <- as.data.frame(results)

library(ggplot2)
library(scales)
p <- ggplot(results_perplexity, aes(x = k, y = perplexity)) +
  geom_point(pch = 21, size = 2, fill = I("orange")) +
  geom_line(color=c("#753633"),size=0.5) +
  ggtitle("3-fold cross-validation of LDA-model with Gobbs sampler",
          "Perplexity when fitting the trained model to the hold-out set.") +
  labs(x = "Candidate number of topics", y = "Perplexity when fitting the trained model to the hold-out set")

p

library(plotly)
ggplotly(p) #The perplexity measure thus indicates a higher no of appropriate topics.