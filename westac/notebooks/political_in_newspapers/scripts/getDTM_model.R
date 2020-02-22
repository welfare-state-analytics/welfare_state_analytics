require(data.table)

text1 <- as.data.frame(fread("political_1945_1989_meta_quoted.tsv", header = FALSE))

colnames(text1)
names(text1)[1] <- "publication"
names(text1)[2] <- "date"
names(text1)[3] <- "doc_id"
names(text1)[4] <- "text"

text1 <- text1[c('doc_id', 'text', 'publication', 'date')]

library(tm)
corp1 <- VCorpus(DataframeSource(text1)) #create a corpus.
#a corpus is (for the purpose of this script/lab) a data class of tm that holds all of our texts
#and that we can manipulate in various ways. See below.

dtm1 <- DocumentTermMatrix(corp1)
#The dtm is a way to transform words into numbers. The basic structure is:
#rows=docs, columns=terms, cells=fq (of terms in docs).

#remove empty rows
nrow(dtm1)
ui <- unique(dtm1$i) #i for rows...
dtm1 <- dtm1[ui,] #remove empty docs
nrow(dtm1) #So here we trim the dtm a little, but thanks to
#the 'DataFrameSource' described above, we maintain our linkage to the original corpora and text. 

#save dtm..
saveRDS(dtm1, file = "dtm1.rds") #call it like so: dtm1 <- readRDS("dtm1.rds")


k <- 50 #no of topics

#All possible hyperP settings..:
controlGibbs <- list(estimate.alpha = TRUE, # alpha is the numeric prior for document-topic multinomial distribution.
                     #Starting value for alpha is 50/k as suggested by Griffiths and Steyvers (2004).
                     estimate.beta = TRUE, #Save logarithmized parameters of the term distribution over topics.
                     #Not a prior in 'topicmodels'! See 'delta' below.
                     verbose = 0, #no information is printed during the algorithm
                     save = 0, #no intermediate results are saved
                     keep = 0, #the log-likelihood values are stored every 'keep' iteration. For diagnostic purposes.
                     seed = list(5683, 123, 8972, 7, 9999), #seed needs to have the length nstart.
                     nstart = 5, #no of independent runs
                     best = TRUE, #only the best model over all runs with respect to the log-likelihood is returned.
                     #Default is true. But read first:
                     #http://cs.colorado.edu/~jbg/docs/2014_emnlp_howto_gibbs.pdf
                     delta = 0.1, #numeric prior for topic-word multinomial distribution. 
                     #The default 0.1 is suggested in Griffiths and Steyvers (2004).
                     #Also, 'delta', ususally referred to as 'beta'. Yes. Confusing.
                     #Decreasing 'Delta' ('beta'), e.g. from 0.1 to 0.001 will increase the granularity of the model.
                     #If topics appear to be to general and hard to distinguish, manipulating 'delta' (lowering its value) 
                     #could be a strategy. If topcs are too particular, go the opposite way.
                     iter = 2000, #>1000
                     burnin = 500, #>200, to throwaway the first inaccurate samples. Default set to 0.
                     thin = 2000) #Default = iter, that is to say we only need the stationary state.
# setting iter=thin is sometimes disputed, optionally thin to have >10 samples.
#Following Griffiths and Steyvers (2004).

#simplified into...:
controlGibbs <- list(seed = 5683, #what does this mean?
                     burnin = 200,
                     iter = 500 #increase for final version)

library(topicmodels)
model2 <- LDA(dtm2, k, method = "Gibbs", control = controlGibbs) #Model...
terms(model2,10)#Summarize.
