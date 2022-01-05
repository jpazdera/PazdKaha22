library(tidyverse)
library(dplyr)
library(lme4)
library(lmerTest)
library(car)

options(contrasts=c("contr.sum","contr.poly"))
setwd("~/git/PazdKaha22/Data")

all_data = read.csv("recall_data.csv", header = TRUE, sep = ",")

# Filter out practice trials and subjects who reported writing notes
data = filter(all_data, wrote_notes == 0, list_num > 2)
#data = filter(all_data, list_num > 2)

# Categorize start position as Serial Position 1, Last 4, or Other
data$sp1_start = data$start_position == 1
data$l4_start = data$start_position > data$list_length - 4
data$start_type = NA
data$start_type[data$sp1_start] = 'SP1'
data$start_type[data$l4_start] = 'L4'
data$start_type[!(data$sp1_start | data$l4_start)] = 'Other'

# Calculate primacy and recency performance scores
data$primacy = rowMeans(select(data, rec1, rec2, rec3))
data$recency = rowMeans(select(data, rec20, rec21, rec22, rec23, rec24))
data$recency[data$list_length==12] = rowMeans(select(data, rec8, rec9, rec10, rec11, rec12))[data$list_length==12]

# Make sure categorical variables are treated as factors
for (column in c('subject', 'experiment', 'wrote_notes', 'modality', 'list_length', 'pres_rate', 'start_type')) {
  data[[column]] = as.factor(data[[column]])
}
data$modality = relevel(data$modality, 'v')
data$pres_rate = relevel(data$pres_rate, '1600')

# Sort data by experiment
e1_data = filter(data, experiment==1)
e2_data = filter(data, experiment==2)

# Check number of trials per start type
summary(e1_data$start_type)  # 12.0% SP1, 67.0% L4, 21.0% Other / 12.7%, 66.4%, 20.9%  if include wn
summary(e2_data$start_type)  # 15.8% SP1, 63.3% L4, 21.0% Other / 16.6%, 62.3%, 21.1% if include wn

# Stats Notes:
# summary(model) gives estimate of coefficients (log-odds), standard errors of coefficients, Wald Z scores, and p-values
# Anova(model) gives Wald chi-squared values, degrees of freedom, and p-values
# exp(fixef(model)) gives the odds ratios for all fixed effect
# odds ratio is e^(coefficient); chi-squared is Z^2; p-value should be approx. equal between methods

###
# Model Testing - Primacy ~ Modality x List Length x Pres Rate
###

# E1
model = glmer(primacy ~ 1 + modality * list_length * pres_rate + (1|subject), data=e1_data,
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)  # All 3 main effects, no interactions

# E2
model = glmer(primacy ~ 1 + modality * list_length * pres_rate + (1|subject), data=e2_data,
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)  # All 3 main effects, no interactions


###
# Model Testing - Recency ~ Modality x List Length x Pres Rate
###

# E1
model = glmer(recency ~ 1 + modality * list_length * pres_rate + (1|subject), data=e1_data,
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)  # All 3 main effects, pres rate x list length interaction

# E2
model = glmer(recency ~ 1 + modality * list_length * pres_rate + (1|subject), data=e2_data,
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)  # All 3 main effects, pres rate x list length interaction


###
# Model Testing - PFR-SP1 ~ Modality x List Length X Pres Rate
###

# E1
model = glmer(sp1_start ~ 1 + modality * list_length * pres_rate + (1|subject), data=e1_data,
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)  # Modality and list length, no interactions

# E2
model = glmer(sp1_start ~ 1 + modality * list_length * pres_rate + (1|subject), data=e2_data,
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)  # All 3 main effects, no interactions


###
# Model Testing - PFR-L4 ~ Modality x List Length X Pres Rate
###

# E1
model = glmer(l4_start ~ 1 + modality * list_length * pres_rate + (1|subject), data=e1_data,
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)  # All 3 main effects, pres rate x list length interaction (greater effect of pres rate on long lists)

# E2
model = glmer(l4_start ~ 1 + modality * list_length * pres_rate + (1|subject), data=e2_data,
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)  # Modality and 3-way interaction

model = glmer(l4_start ~ 1 + modality * pres_rate + (1|subject), data=e2_data[e2_data$list_length=='12',],
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
model = glmer(l4_start ~ 1 + modality * pres_rate + (1|subject), data=e2_data[e2_data$list_length=='24',],
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)

# 3-way interaction appears to be because the modality effect is marginally larger during slow presentation than fast presentation on 12 item lists,
# but (non-significantly) larger during fast presentation than during slow presentation on 24 item lists. This direction reversal of the 
# non-significant two-way interaction makes the three-way interaction significant.


###
# Model Testing - Primacy ~ Modality x Start Position
###

# E1
model = glmer(primacy ~ 1 + modality * start_type + (1|subject), 
              data=e1_data[e1_data$start_type != 'Other',],
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)

# E2
model = glmer(primacy ~ 1 + modality * start_type + (1|subject), 
              data=e2_data[e2_data$start_type != 'Other',],
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)


###
# Model Testing - Recency ~ Modality x Start Position
###

# E1
model = glmer(recency ~ 1 + modality * start_type + (1|subject), 
              data=e1_data[e1_data$start_type != 'Other',],
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)

# E2
model = glmer(recency ~ 1 + modality * start_type + (1|subject), 
              data=e2_data[e2_data$start_type != 'Other',],
              family='binomial', control=glmerControl(optimizer = "bobyqa"))
summary(model)
Anova(model, type=3)


###
# INTRUSION ANALYSES
###

# Load intrusion data table
idata = read.csv("intrusion_data.csv", header = TRUE, sep = ",")
idata$subject = as.factor(idata$subject)
idata$enc_modality = as.factor(idata$enc_modality)
idata$ret_modality = as.factor(idata$ret_modality)

# Drop data from people who wrote notes
idata = filter(idata, wrote_notes==0)

# Sort data by experiment
e1_idata = filter(idata, experiment==1)
e2_idata = filter(idata, experiment==2)

# Sort Experiment 2 data by modality
e2v_idata = filter(e2_idata, enc_modality=='v')
e2a_idata = filter(e2_idata, enc_modality=='a')

# E1:

# Check number of participants with data in each condition
drop_na(e1_idata) %>% group_by(enc_modality, ret_modality) %>% summarize(N=n())

# Run ANOVA for PLIs per trial
model = aov(plis ~ 1 + enc_modality * ret_modality + Error(subject/(enc_modality*ret_modality)), data=e1_idata)
summary(model)

# Run linear mixed model for PLI recency
model = lmer(pli_recency ~ 1 + enc_modality * ret_modality + (1|subject), data=e1_idata)
summary(model)
Anova(model, type=3)

# E2:

# t-test for PLIs per trial
t.test(e2v_idata$plis, e2a_idata$plis, alternative="two.sided", paired=F, var.equal=T)

# t-test for PLI recency
t.test(e2v_idata$pli_recency, e2a_idata$pli_recency, alternative="two.sided", paired=F, var.equal=T)
