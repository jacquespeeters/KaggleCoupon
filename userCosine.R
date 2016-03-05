# Jacques Peeters
# Projet de fin d'études - Polytech Lille
# kaggle.com/c/coupon-purchase-prediction/
# github.com/l4mun/KaggleCoupon

library(dplyr)
library(Matrix)
library(dummies)
library(data.table)

setwd("C:/Users/Jacques/Documents/PFE/jacques")

# Lecture des jeux de données avec le package data.table pour gagner du temps
#####
couponlist= fread("data/couponlist.csv")
userlist= fread("data/userlist.csv") #un warning, non gênant
userlist$user_id = as.integer(userlist$user_id)
weblog= fread("data/weblog.csv")
weblog$user_id = as.integer(weblog$user_id)
weblog$coupon_id = as.integer(weblog$coupon_id)

########################
# quelques Stats descriptives
########################
# nrow(filter(userlist, gender==0)) / nrow(userlist)

# couponlist$coupon_id = as.numeric(couponlist$coupon_id )
# 
# weblog %>% filter(coupon_id %in% unique(couponlist$coupon_id)) -> weblogF
# 
# weblogF %>% filter(purchase_flag ==1) %>% left_join(couponlist) %>% group_by(price_rate) %>% summarise( nb = n())  -> info
# couponlist %>% group_by( price_rate) %>% summarise( div = n()) -> info1
# info2 = left_join(info1, info)
# plot(info2$price_rate, info2$nb / info2$div )
# 
# weblogF %>% filter(purchase_flag ==1) %>% left_join(couponlist) %>% group_by(discount_price) %>% summarise( nb = n())  -> info
# couponlist %>% group_by( discount_price) %>% summarise( div = n()) -> info1
# info2 = left_join(info1, info)
# plot(info2$discount_price, info2$nb / info2$div )
# 
# weblogF %>% filter(purchase_flag ==1) %>% left_join(couponlist) %>% group_by(disp_period) %>% summarise( nb = n())  -> info
# couponlist %>% group_by( disp_period) %>% summarise( div = n()) -> info1
# info2 = left_join(info1, info)
# plot(info2$disp_period, info2$nb / (info2$div+500 ))
# 
# weblogF %>% filter(purchase_flag ==1) %>% left_join(couponlist) %>% group_by(valid_period) %>% summarise( nb = n())  -> info
# couponlist %>% group_by( valid_period) %>% summarise( div = n()) -> info1
# info2 = left_join(info1, info)
# plot(info2$valid_period, info2$nb / (info2$div ))


########################
# Feature engineering
########################
# copie pour éviter d'écraser des informations
feature=couponlist

# Dates
# Je ne tiens pas compte des dates (même si on pourrait tester que plus l'achat est ancien, moins il a d'importance)
# Seul les durée sont importantes
feature %>% select (-c(disp_from, disp_end,valid_from,valid_end)) -> feature

# capsulte_text et genre_name
# capsule_text inclu dans genre_name
# unique(feature$capsule_text)
# unique(couponlist$genre_name)
# Les variables sont très corrélées, je décide de ne garder que genre_name
feature %>% select(-capsule_text) -> feature

# Je passe la variable genre_name en variable dummies (0/1)
feature = dummy.data.frame(feature, names = "genre_name", sep=":")

# valid_period
# J'arrondi à l'unité
feature$valid_period = round(feature$valid_period,0)

# Géographie/Shop :
# shop_small_area_name inclu dans shop_pref_name lui-même inclu dans shop_large_area_name
#     unique(couponlist$shop_large_area_name) #9 différentes
#     unique(couponlist$shop_pref_name) #47 différentes
#     unique(couponlist$shop_small_area_name) #55 différentes

#Je choisi de ne pas garder shop_pref_name
feature %>% select(-c(shop_pref_name)) -> feature

# Je raccourcis le nom des variables
setnames(feature, "shop_large_area_name", "large_area_name")
setnames(feature, "shop_small_area_name", "small_area_name")

# Je passe les variables large_area_name et small_area_name en dummies (1/0)
feature = dummy.data.frame(feature, names = "large_area_name", sep=":")
feature = dummy.data.frame(feature, names = "small_area_name", sep=":")

# Prix
feature$discount_price = 1/(log10(feature$discount_price+1)+1)
feature$price_rate = feature$price_rate/100
feature$catalog_price = 1/(log10(feature$catalog_price+1)+1)

feature$disp_period = feature$disp_period / max(feature$disp_period)
feature$valid_period = feature$valid_period / max(feature$valid_period)


########################   
# Model
######################## 

# Les fonctions
# Création de la matrice sparse en fonction des notes des utilisateurs 
fURM = function(URM){
  sURM = sparseMatrix(URM$user_id, URM$coupon_id, x = URM$rating)
  col = unique(weblogF$coupon_id) #On ne garde que les colonnes intéressantes
  col=sort(col) #Il faut ordonner les colonnes à conserver
  sURM=sURM[,col] # sinon la commande suivante marche mal, je l'ai appris douloureusement
  return(sURM)
}

# Permet la création de la matrice diagonale W à partir d'un vecteur weight.
get.W = function(weight){
  W=c(
    rep(weight[1], length(unique(couponlist$genre_name))), #Il faut répliquer weight[1] autant de fois qu'il de variables dummies pour genre_name
    weight[2], 
    weight[3],
    weight[4],
    weight[5],
    weight[6],
    rep(weight[7], 9), 
    rep(weight[8], length(unique(couponlist$shop_large_area_name))), 
    rep(weight[9], length(unique(couponlist$shop_small_area_name)) ))
  return (W)
}

# Prédiction des notes pour les couponTest pour chaque groupe d'utilisateurs
prediction = function(sURM, couponTest, couponTrain, W, subUsers, coupon_id_hash, user_id_hash){
  div = rowSums(sURM)
  div[which(div ==0)]=1 #Simplement pour éviter les divisions par 0
  userPref = (sURM %*% as.matrix(couponTrain[,-length(couponTrain)])) / div
  # Mettre price_rate, catalog_price, discount_price, disp_period, valid_period à 1
  # En effet, les utilisateurs cherchent toujours des prix bas et des période d'achat et d'utilisation grande. Indépendant de leurs achats précédents.
  userPref[,c(14,15,16,17,18)] = 1
  
  # Calcul du score pour les couponTest pour chaque user en fonction de la matrice de pondération W
  score = as.matrix(userPref %*% diag(W[[1]]) %*% t(as.matrix(couponTest[,-length(couponTrain)])))
  for (i in 1:length(W)){
    score[subUsers[[i]],] = as.matrix(userPref[subUsers[[i]],] %*% diag(W[[i]]) %*% t(as.matrix(couponTest[,-length(couponTrain)])))
  }
  
  coupon_id_hash$coupon_id = as.character(coupon_id_hash$coupon_id) #pour permettre la jointure
  left_join(couponTest, coupon_id_hash) %>% select(coupon_id, coupon_hash) -> name #Les coupon_id_hash ordonnés selon couponTest
  submission = matrix(data = NA, nrow = nrow(score), ncol=2) 
  submission = as.data.frame(submission) #Le data.frame qui va accueillir les prédictions 
  colnames(submission)[1:2] = c("USER_ID_hash","PURCHASED_COUPONS")
  submission$USER_ID_hash = user_id_hash$user_hash #Les user_hash
  # Plutôt qu'itérer, permet un gain de rapidité monstrueux
  submission$PURCHASED_COUPONS <- do.call(rbind, lapply(1:nrow(score),FUN=function(i){
    #Ordonne les 10 indices des meilleurs scores par user i, puis récupère les coupon_hash et ensuite les concatène en une chaîne
    purchased_cp <- paste(name$coupon_hash[order(score[i,], decreasing = TRUE)][1:10],collapse=" ")  
    return(purchased_cp)
  }))
  return (submission)
}

# Le script
#########################################

# Filtrer les coupons qui ne sont pas dans couponlist => aucunes infos sur eux, impossibilité de calculer des similarités
weblog %>% filter(coupon_id %in% unique(couponlist$coupon_id)) -> weblogF

# Créer rating(user_id, coupon_id) en fct de purchase, nombre de visite, ...
weblogF %>% 
  group_by(user_id, coupon_id) %>%
    summarise(rating = ifelse(sum(purchase_flag)>0, 1, 0) +n()*0.2 ) -> URM

# Créer l'User Rating Matrix
sURM = fURM(URM)

# Charger la table qui permet la conversion de hash <=> id pour les coupons puis users
coupon_id_hash = read.csv("data/keys/coupon_id_hash.key")
user_id_hash = read.csv("data/keys/user_id_hash.key")

# On sélectionne les couponTest
couponlist %>% filter(dataset==-1) -> couponTest
couponTest = unique(couponTest$coupon_id)
feature %>% filter(coupon_id %in% couponTest) %>% select(-c(dataset)) -> couponTest
# couponTest$coupon_id = as.integer(couponTest$coupon_id)
couponTest = arrange(couponTest,coupon_id)

# On sélectionne les couponTrain qui sont apparus au moins une fois dans weblogF
couponTrain=unique(weblogF$coupon_id) 
feature %>% filter(coupon_id %in% couponTrain) %>% select(-c(dataset)) -> couponTrain
couponTrain$coupon_id = as.integer(couponTrain$coupon_id)
couponTrain = arrange(couponTrain,coupon_id)

# On paramètre les matrices de poids
weightFemale = c(10,  #genre_name   
           0.1,   #price_rate
           0.05,   #catalog_price
           6.5,  #discount_price
           6.5,   #disp_period
           0.05, #valid_period
           0,   #usable_date_DAY
           1.5,   # large_area_name
           30)  # small_area_name

weightMale = c(10,  #genre_name   
               0.1,   #price_rate
               0.05,   #catalog_price
               6.5,  #discount_price
               5.5,   #disp_period
               0.5, #valid_period
               0,   #usable_date_DAY
               1.5,   # large_area_name
               20)  # small_area_name
           

Wfemale = get.W(weightFemale)
Wmale = get.W(weightMale)
W = list(Wfemale, Wmale)
userlist = arrange(userlist, user_id)
female= filter(userlist, gender == 0)$user_id
male = filter(userlist, gender == 1)$user_id
subUsers = list(female, male)

system.time(test<-prediction(sURM, couponTest, couponTrain, W, subUsers, coupon_id_hash, user_id_hash))
nameFile=paste(c("masubmission",Sys.time(),".csv"),collapse="")
write.table(test, nameFile, col.names = TRUE, row.names = FALSE, sep=",")
rm(nameFile)



# Partie locale
##################################################

# Filtrer les coupons qui ne sont pas dans couponlist => aucunes infos sur eux, impossibilité de calculer des similarités...
weblog %>% filter(coupon_id %in% unique(couponlist$coupon_id)) -> weblogF

# Placer les 7 derniers jours dans weblogTest et le reste dans weblogTrain
limite=sort(unique(weblogF$activity_date), decreasing = T)[7]
weblogTest= filter(weblogF, activity_date >=limite)
weblogTrain = filter(weblogF, activity_date <limite)
rm(limite)

# Supprimmer de weblogTest les coupon qui sont dans weblogTrain
weblogTest <- weblogTest %>% filter(!coupon_id %in% unique(weblogTrain$coupon_id))

# Identifier les achats (utilisateur, coupon) et les placer dans actual
weblogTest %>% 
  group_by(user_id, coupon_id) %>%
  summarise(rating = (ifelse(sum(purchase_flag)>0, 1, 0))) %>%
  filter(rating ==1)  %>% 
  group_by(user_id) %>% 
  summarize(purchased = list(coupon_id))-> actual

# Compter le nombre d'achats par coupon et un peu d'analyse
weblogTest %>% 
  group_by(user_id, coupon_id) %>%
  summarise(rating = (ifelse(sum(purchase_flag)>0, 1, 0))) %>%
  filter(rating ==1) %>%
  group_by(coupon_id) %>%
  summarize(nb = n())-> nbAchat
nbAchat = arrange(nbAchat, desc(nb))
nbAchat
nbAchat$coupon_id[1]
plot(nbAchat$nb/sum(nbAchat$nb), xlab="Coupons ordonnés", ylab="Nombre d'achats", main="Poids des coupons pour le nombre d'achat total de coupons")

sum(nbAchat$nb[1:4]) / sum(nbAchat$nb)
sum(nbAchat$nb) - sum(nbAchat$nb[1:4])

actualp=matrix(data = NA, nrow=nrow(userlist), ncol=1)
actualp = as.data.frame(actualp)
names(actualp) = names(actual)[1]
actualp[,1]=c(1:nrow(userlist))
actual = left_join(actualp, actual, by="user_id")
rm(actualp)

# Sélection des couponTest
feature %>% filter(coupon_id %in% unique(weblogTest$coupon_id), !coupon_id %in% nbAchat$coupon_id[1:10]) %>% select(-c(dataset)) -> couponTest
couponTest$coupon_id = as.integer(couponTest$coupon_id) 
couponTest = arrange(couponTest, coupon_id)

# Sélection des couponTrain
feature %>% filter(coupon_id %in% unique(weblogTrain$coupon_id)) %>% select(-c(dataset)) -> couponTrain
couponTrain$coupon_id = as.integer(couponTrain$coupon_id) 
couponTrain = arrange(couponTrain, coupon_id)

# Mes fonctions
###############
# Fonction proposée par Kaggle pour calculer l'Average Precision
apk <- function(k, actual, predicted){
  score <- 0.0
  cnt <- 0.0
  for (i in 1:min(k,length(predicted)))
  {
    if (predicted[i] %in% actual && !(predicted[i] %in% predicted[0:(i-1)]))
    {
      cnt <- cnt + 1
      score <- score + cnt/i 
    }
  }
  score <- score / min(length(actual), k)
  score
}

# Fonction proposée par Kaggle pour calculer le Mean Average Precision
mapk <- function (k, actual, predicted){
  if( length(actual)==0 || length(predicted)==0 ) 
  {
    return(0.0)
  }
  
  scores <- rep(0, length(actual))
  for (i in 1:length(scores))
  {
    scores[i] <- apk(k, actual[[i]], predicted[[i]])
  }
  score <- mean(scores)
  score
}

# Fonction pour la  matrice sparse locale
fURM = function(visit){
  weblogTrain %>% 
    group_by(user_id, coupon_id) %>%
    summarise(rating = (ifelse(sum(purchase_flag)>0, 1, 0) + n()*visit)) -> URM
  sURM = sparseMatrix(URM$user_id, URM$coupon_id, x = URM$rating)
  col = unique(weblogTrain$coupon_id) #On garde que les colonnes intéressantes
  col=sort(col) 
  sURM=sURM[,col]
  return(sURM)
}

# Permet de réaliser les prédictions en local puis calculer la performance MAP@10
predictionLocal = function(W, users){
  score = as.matrix(userPref %*% diag(W) %*% t(as.matrix(couponTest[-length(couponTrain)])))
  name = couponTest$coupon_id
  # Plutôt qu'itérer, permet un gain de rapidité monstrueux
  actual$predicted <- do.call(rbind, lapply(1:nrow(score),FUN=function(i){
    purchased_cp <- list(name[order(score[i,], decreasing = TRUE)][1:10])
    return(purchased_cp)
  }))
  # Garder les users souhaités
  actual = actual[users,]
  actual = actual[which(lapply(actual$purchased, is.null)==F ),]
  return(mapk(10, actual$purchased, actual$predicted))
}

################
# Script Local
################
visit=0
sURM = fURM(visit)
div = rowSums(sURM)
div[which(div ==0)]=1
userPref = (sURM %*% as.matrix(couponTrain[-length(couponTrain)])) / div

userlist = arrange(userlist, user_id)
users = userlist$user_id
users <- filter(userlist, gender == 1)$user_id

res=c()
# iter = (-10:10)/10+autour
iter = (1:50)
for (i in iter){
  weight = c(10,  #genre_name   
             0,   #price_rate
             0,   #catalog_price
             0,  #discount_price
             0,   #disp_period
             0, #valid_period
             0,   #usable_date_DAY
             0,   # large_area_name
             i)  # small_area_name
  
  W = get.W(weight)
  res =cbind(res, predictionLocal(W, users) )
}
plot(iter,res)
grid(ny=0)
autour=iter[which.max(res)]
autour
# max(res)


