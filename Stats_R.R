
data = read.csv("stats_copy_merged copy.csv")

#here the columns are grouped if they were one-hot encoded from one categorical
#column. This is necessary for univariate logistic regression analysis.

columns_grouped = list(c('Age_1.0', 'Age_2.0'),
                       c('DEWOR6A'),c('FamilySize_1.0', 'FamilySize_2.0'),
                       c('ADD'),c('Sex'), c('CHBED6A'), c('GCHAR_1.0', 'GCHAR_2.0'),
                       c('GDUNG_1.0', 'GDUNG_2.0'), c('GGAS_1.0', 'GGAS_2.0'),
                       c('GLEAVES_1.0','GLEAVES_2.0'), c("GNAF_1.0", "GNAF_2.0"),
                       c('GWOOD_1.0', 'GWOOD_2.0'), c('GELEC_1.0', 'GELEC_2.0'),
                       c('GFLOOR6A'), c('MaternalEdc'), c('MaternalOcc_1.0',
                                                          'MaternalOcc_2.0'),
                       c('CHMAT6A'), c('GROOF6A'), c('GWALL6A'), c('CHSLP6A'),
                       c('GCPOS'), c('ASTL6A'), c('HAYF6A'), c('HAYFL6A'),
                       c('RASHL6A'), c('WHZL6A'),c('DP'), c('FAAS6A'),c('FAHAY6A'),
                       c('FAWHEZ6A'),c('MOAS6A'),c('MOHAY6A'),c('MOWHZ6A'),
                       c('GAPPD'), c('HCIGR6A'),c('GCOOK6A'), c('GCAT'),
                       c('GCOW'), c('GDOG'), c('GHEN'), c('GHORSE'),
                       c('GPIG'),  c('GSHEEP'), c('GWATER6A_1.0', 'GWATER6A_2.0'),
                       c('GTOLIET6A'),c('GWASTE6A_1.0',
                                        'GWASTE6A_2.0'),
                       c('HCT'), c('HGB'), c('LYM_1.0', 'LYM_2.0'),
                       c('MCH'), c('MCHC'),c('MCV'),c('PLT'),
                       c('RBC'), c('WBC')) 
    
#this is the vector with all the columns, for multivariate analysis.

columns = unlist(columns_grouped)

options(scipen=999)

#This code performs the multivariate logistic regressions for each outcome.
columns_formula = paste(columns, collapse = "+")
model_logi_STH = glm(paste("AnySTHs ~ ", columns_formula), data = data, family = "binomial")
model_logi_Par = glm(paste("AnyParasite ~ ",columns_formula), data = data, family = "binomial")
model_logi_Helm = glm(paste("AnyHelminth ~ " ,columns_formula), data = data, family = "binomial")
model_logi_Prot = glm(paste("AnyProt ~ ",columns_formula), data = data, family = "binomial")

#This code writes the results to multiple csv files.
# write.csv(cbind(attr((model_logi_STH$terms), "term.labels"),
#                 p.adjust(summary(model_logi_STH)$coefficients[1:length(columns)+1,4],
#                          method = "BH"),
#                 exp(summary(model_logi_STH)$coef[1:length(columns)+1]),
#                 exp(confint(model_logi_STH))[1:length(columns)+1, 1:2]),
#           file = "STH_multi.csv", row.names = F)
# write.csv(cbind(attr((model_logi_Par$terms), "term.labels"),
#                 p.adjust(summary(model_logi_Par)$coefficients[1:length(columns)+1,4],
#                          method = "BH"),
#                 exp(summary(model_logi_Par)$coef[1:length(columns)+1]),
#                 exp(confint(model_logi_Par))[1:length(columns)+1, 1:2]),
#           file = "Par_multi.csv", row.names = F)
# write.csv(cbind(attr((model_logi_Prot$terms), "term.labels"),
#                 p.adjust(summary(model_logi_Prot)$coefficients[1:length(columns)+1,4],
#                          method = "BH"),
#                 exp(summary(model_logi_Prot)$coef[1:length(columns)+1]),
#                 exp(confint(model_logi_Prot))[1:length(columns)+1, 1:2]),
#           file = "Prot_multi.csv", row.names = F)
# write.csv(cbind(attr((model_logi_Helm$terms), "term.labels"),
#                 p.adjust(summary(model_logi_Helm)$coefficients[1:length(columns)+1,4],
#                          method = "BH"),
#                 exp(summary(model_logi_Helm)$coef[1:length(columns)+1]),
#                 exp(confint(model_logi_Helm))[1:length(columns)+1, 1:2]),
#           file = "Helm_multi.csv", row.names = F)

write.csv(cbind(attr((model_logi_STH$terms), "term.labels"),
                summary(model_logi_STH)$coefficients[1:length(columns)+1,4],
                exp(summary(model_logi_STH)$coef[1:length(columns)+1]),
                exp(confint(model_logi_STH))[1:length(columns)+1, 1:2]),
          file = "STH_multi.csv", row.names = F)
write.csv(cbind(attr((model_logi_Par$terms), "term.labels"),
                summary(model_logi_Par)$coefficients[1:length(columns)+1,4],
                exp(summary(model_logi_Par)$coef[1:length(columns)+1]),
                exp(confint(model_logi_Par))[1:length(columns)+1, 1:2]),
          file = "Par_multi.csv", row.names = F)
write.csv(cbind(attr((model_logi_Prot$terms), "term.labels"),
                summary(model_logi_Prot)$coefficients[1:length(columns)+1,4],
                exp(summary(model_logi_Prot)$coef[1:length(columns)+1]),
                exp(confint(model_logi_Prot))[1:length(columns)+1, 1:2]),
          file = "Prot_multi.csv", row.names = F)
write.csv(cbind(attr((model_logi_Helm$terms), "term.labels"),
                summary(model_logi_Helm)$coefficients[1:length(columns)+1,4],
                exp(summary(model_logi_Helm)$coef[1:length(columns)+1]),
                exp(confint(model_logi_Helm))[1:length(columns)+1, 1:2]),
          file = "Helm_multi.csv", row.names = F)

#This code creates blank data frames for univariate logistic regression models.
df_uni_STH <- data.frame(Variable_Name=character(), C_Odds_Ratio=numeric(), Conf_025=numeric(),
Conf_975=numeric(), P_Value_uni=numeric())
df_uni_Par <- data.frame(Variable_Name=character(), C_Odds_Ratio=numeric(), Conf_025=numeric(),
Conf_975=numeric(), P_Value_uni=numeric())
df_uni_Prot <- data.frame(Variable_Name=character(), C_Odds_Ratio=numeric(), Conf_025=numeric(),
Conf_975=numeric(), P_Value_uni=numeric())
df_uni_Helm <- data.frame(Variable_Name=character(), C_Odds_Ratio=numeric(), Conf_025=numeric(),
Conf_975=numeric(), P_Value_uni=numeric())

#This code performs univariate analysis (by groups), and then appends the results
#to the data frames made in the above code.
for (i in 1:length(columns_grouped)){
  vec = unlist(columns_grouped[i])
  a = paste(vec, collapse = "+")
  uni_model_STH = glm(paste("AnySTHs ~", a), data = data, family ="binomial")
  uni_model_Par = glm(paste("AnyParasite ~",a),data =data, family = "binomial")
  uni_model_Prot = glm(paste("AnyProt ~",a),data =data, family = "binomial")
  uni_model_Helm = glm(paste("AnyHelminth ~",a),data =data, family = "binomial")
  trm <- attr(uni_model_STH$terms, "term.labels")
  for (j in 1:length(vec)){
    print(j)
    uni_STH <- c(trm[j], (exp(summary(uni_model_STH)$coef[j+1,][1])),
    exp((confint(uni_model_STH)[j+1,])),
    (summary(uni_model_STH)$coef[j+1,][4]))
    uni_Par = c(trm[j], (exp(summary(uni_model_Par)$coef[j+1,][1])),
    exp((confint(uni_model_Par)[j+1,])),
    (summary(uni_model_Par)$coef[j+1,][4]))
    uni_Prot = c(trm[j], (exp(summary(uni_model_Prot)$coef[j+1,][1])),
    exp((confint(uni_model_Prot)[j+1,])),
    (summary(uni_model_Prot)$coef[j+1,][4]))
    uni_Helm = c(trm[j], (exp(summary(uni_model_Helm)$coef[j+1,][1])),
    exp((confint(uni_model_Helm)[j+1,])),
    (summary(uni_model_Helm)$coef[j+1,][4]))
  df_uni_STH[nrow(df_uni_STH) + 1,] = uni_STH
  df_uni_Par[nrow(df_uni_Par) + 1,] = uni_Par
  df_uni_Prot[nrow(df_uni_Prot) + 1,] = uni_Prot
  df_uni_Helm[nrow(df_uni_Helm) + 1,] = uni_Helm
  }
}
#This replaces NAs with 0 (in case the probabilities could not fit to 0 or 1)
df_uni_STH[is.na(df_uni_STH)] <- 0
df_uni_Par[is.na(df_uni_Par)] <- 0
df_uni_Prot[is.na(df_uni_Prot)] <- 0
df_uni_Helm[is.na(df_uni_Helm)] <- 0
#Adjusting the p-values
df_uni_STH$P_Value_uni = p.adjust(df_uni_STH$P_Value_uni)#, method = "BH")
df_uni_Par$P_Value_uni = p.adjust(df_uni_Par$P_Value_uni)#, method = "BH")
df_uni_Prot$P_Value_uni = p.adjust(df_uni_Prot$P_Value_uni)#, method = "BH")
df_uni_Helm$P_Value_uni = p.adjust(df_uni_Helm$P_Value_uni)#, method = "BH")
#This writes the results to csv files.
write.csv(df_uni_STH, file = "STH_uni.csv", row.names = FALSE)
write.csv(df_uni_Par, file = "Par_uni.csv", row.names = FALSE)
write.csv(df_uni_Prot, file = "Prot_uni.csv", row.names = FALSE)
write.csv(df_uni_Helm, file = "Helm_uni.csv", row.names = FALSE)

#This code is optional, for visual ease. 
create_table <- function(dataframe, label) {
  kableExtra::kable_styling(knitr::kable(dataframe[dataframe$P_Value <= 0.05,], booktabs = TRUE, 
                                         caption = label), font_size = 10)
}

create_table(df_uni_Helm, "Helm")

#This code finds the frequencies of positive and negative cases for each risk factor

df_counts = data.frame(Var_name = character(), AnySTHs_true_count = character(),
                       AnySTHs_false_count = character(),
                       AnyPar_true_count = character(),
                       AnyPar_false_count = character(),
                       AnyHelm_true_count = character(),
                       AnyHelm_false_count = character(),
                       AnyProt_true_count = character(),
                       AnyProt_false_count = character())
for (i in 1:length(columns)){
  v = c(columns[i],paste(length(which(data[columns[i]] ==1 & data["AnySTHs"]==1)),
        " (",round(length(which(data[columns[i]] ==1 & data["AnySTHs"]==1))/length(which(data[columns[i]]==1))*100,1),")%",sep =""),
        paste(length(which(data[columns[i]] ==1 & data["AnySTHs"]==0)),
        " (", round(length(which(data[columns[i]] ==1 & data["AnySTHs"]==0))/length(which(data[columns[i]]==1))*100,1),")%", sep = ""),
        paste(length(which(data[columns[i]] ==1 & data["AnyParasite"]==1)),
               " (",round(length(which(data[columns[i]] ==1 & data["AnyParasite"]==1))/length(which(data[columns[i]]==1))*100,1),")%",sep =""),
        paste(length(which(data[columns[i]] ==1 & data["AnyParasite"]==0)),
              " (", round(length(which(data[columns[i]] ==1 & data["AnyParasite"]==0))/length(which(data[columns[i]]==1))*100,1),")%", sep = ""),
        paste(length(which(data[columns[i]] ==1 & data["AnyHelminth"]==1)),
              " (",round(length(which(data[columns[i]] ==1 & data["AnyHelminth"]==1))/length(which(data[columns[i]]==1))*100,1),")%",sep =""),
        paste(length(which(data[columns[i]] ==1 & data["AnyHelminth"]==0)),
              " (", round(length(which(data[columns[i]] ==1 & data["AnyHelminth"]==0))/length(which(data[columns[i]]==1))*100,1),")%", sep = ""),
        paste(length(which(data[columns[i]] ==1 & data["AnyProt"]==1)),
              " (",round(length(which(data[columns[i]] ==1 & data["AnyProt"]==1))/length(which(data[columns[i]]==1))*100,1),")%",sep =""),
        paste(length(which(data[columns[i]] ==1 & data["AnyProt"]==0)),
              " (", round(length(which(data[columns[i]] ==1 & data["AnyProt"]==0))/length(which(data[columns[i]]==1))*100,1),")%", sep = ""))
  df_counts[nrow(df_counts) + 1,] = v
}

#This prints the results to a csv file.
write.csv(df_counts, file = "counts.csv", row.names = FALSE)
