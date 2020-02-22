dtm1 <- readRDS("dtm1.rds")

#How to determine no of topics
###############################################################################
###############################################################################

#Method 1. Density-based method
#From:
#Cao Juan, Xia Tian, Li Jintao, Zhang Yongdong, and Tang Sheng. 2009. A density-based method for adaptive lDA model selection.
#Neurocomputing - 16th European Symposium on Artificial Neural Networks 2008 72, 7-9: 1775-1781.

#"Cluster-like approach: ..."that the similarity will be as large as possible in the intra-
#cluster, but as small as possible between inter-clusters."

#Transfer this idea to topics, maximizing (density-based) similarity intra-topics
#and maximizing difference inter-topics.

###############################################################################

#Metod 2. Harmonic-mean of the log likelihood
#From (among others):
#Thomas L. Griffiths and Mark Steyvers. 2004. Finding scientific topics.
#Proceedings of the National Academy of Sciences 101, suppl 1: 5228-5235.

#Maximizing the likelihood of the observed data by changing the number of topics.
#(I.e. maximizing the probability that the model gives to the observed data,
#i.e. the likelihood P(words|no. of topics))
#This (log-)likelihood is estimated with harmonic means, using the Gibbs sampler.

###############################################################################

#From the package 'ldatuning', we get the result of method 1 and 2 normalized to a 0-1 scale,
#where we will just pick the model with a minimum from method 1, and maximum from method 2. 

###############################################################################
###############################################################################


candidate_k <- c(2, 3, 2:60 * 2, 7:10 * 20) # a proper sampling of models with different no of K.
#Minimal hyperP....:
controlGibbs <- list(seed = 5683, #hrm?
                     burnin = 200,
                     iter = 500)
library(parallel)
library(ldatuning)
library(topicmodels)
cores <- detectCores(logical = TRUE) - 1 #leave one for stability
system.time({
  results_Ka <- FindTopicsNumber(
    dtm1,
    topics = candidate_k,
    metrics = c("Griffiths2004", "CaoJuan2009"),
    method = "Gibbs",
    control = controlGibbs,
    mc.cores = cores, 
    verbose = TRUE
  )
})
results_Ka
FindTopicsNumber_plot(results_Ka)

#"But I'm not guilty," said K. "there's been a mistake.
#How is it even possible for someone to be guilty?
#We're all human beings here, one like the other."
#"That is true" said the priest "but that is how the guilty speak." 


###############################################################################