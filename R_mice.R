library(mice)
library(here)

setwd(here('dataset'))

all_R_mice = read.csv('to_mice.csv')
all_R_mice = mice(all_R_mice, m=5, seed=123, method='rf')
all_R_mice1 = complete(all_R_mice, action=1)
write.csv(all_R_mice1, 'miced.csv', row.names=FALSE)
