library(data.table)
train <- fread("./input/train.csv", header = TRUE)
train <- train[,c("ID","target","f190486d6","58e2e02e6","eeb9cd3aa","9fd594eec","6eef030c1","15ace8c9f","fb0f5dbfe","58e056e12","20aa07010","024c577b9","d6bb78916","b43a7cfd5","58232a6fb"),with=F]
train <- train[ c(2072,3493,379,2972,2367,4415,2791,3980,194,1190,3517,811,4444) ]
print("No Magic No Gain")
head(train,16)