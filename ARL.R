library(arules)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd('FinalPackage') #Directory depends on where the data set can be found.

#This defines the outcomes.
outcomes = c('AnySTHs', 'AnyHelminth', 'AnyProt', "AnyParasite")
for (outcome in outcomes){
  fname = paste(outcome, "_SMOTE_", ".csv", sep ="")
  data = read.csv(fname)
  #Removes the column with indexes
  data = data[,!names(data) %in% c("X")]
  
  #converts 0s and 1s to TRUE and FALSE
  data <- data.frame(lapply(data,as.logical))
  
  #Uses the apriori algorithm to perform association rule learning.
  rules <- apriori(data, parameter = list(supp=0.002, conf=0.5,maxlen=5,target ="rules"),appearance = list(default="lhs",rhs=outcome), control=list(verbose = FALSE, load = T, memopt = F))
  filtered_rules = subset(rules, subset = lift >1.5)
  
  #This code outputs the results to a csv
  f_out =paste("arules_SMOTE", outcome, ".csv", sep ="")
  arules::write(filtered_rules, file = f_out, sep = ",")
}