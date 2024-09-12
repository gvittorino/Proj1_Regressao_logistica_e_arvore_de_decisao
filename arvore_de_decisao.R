


# Instalação e carregamento dos pacotes utilizados
pacotes <- c("tidyverse",
             "rpart",
             "rpart.plot",
             "readxl",
             "caTools",
             "fastDummies",
             "caret",
             "pROC",
             "plotly",
             "forcats",
             "randomForest",
             "gbm",
             "xgboost")

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T) 
} else {
  sapply(pacotes, require, character = T) 
}

################################################################################
######################### Árvores de Decisão ###################################
################################################################################

pacientes <- read.table("dados_pacientes.txt", header = TRUE, sep = "\t")

pacientes$classe <- as.factor(pacientes$classe)

# Excluindo a variável incidencia_pelvica devido a multicolinearidade
pacientes$incidencia_pelvica <- NULL

# Vamos analisar algumas estatísticas descritivas da base de dados
summary(pacientes[,1:6])

# Separação da base de dados em bases de treino e teste
set.seed(132)

pacientes$id <- 1:nrow(pacientes)

treino <- pacientes %>% sample_frac(0.75)
teste  <- anti_join(pacientes, treino, by = 'id')

treino$id <- NULL
teste$id <- NULL

# Gerando a árvore de decisão (árvore de classificação)
set.seed(132)

arvore <- rpart(formula = classe ~ versao_pelvica +lordose_lombar + declive_sacral + raio_pelvico + grau_deslizamento,
                data = treino,
                parms = list(split = "gini"),
                method = "class",
                control = rpart.control(minsplit = 25,
                                        maxdepth = 5,
                                        minbucket = 10,
                                        cp = 0))


# Plotando a árvore
rpart.plot(arvore, type = 4, clip.right.labs = F, extra = 101, nn = T, cex = 0.6)
#ggsave(filename = "arvore.jpg", plot = arvore, width = 8, height = 6, dpi = 300)

# Resultados detalhados da árvore
summary(arvore)

# Resultados relacionados ao parâmetro de complexidade
plotcp(arvore)
printcp(arvore)

# Valores preditos pela árvore (base de treino)
preditos_treino <- predict(arvore, treino)


preditos_class <- apply(preditos_treino, 1, function(x) {
  if (x[1] > 0.5) {
    return("DH")
  } else if (x[2] > 0.5) {
    return("NO")
  } else {
    return("SL")
  }
})

# Converter para fator com os níveis corretos
preditos_class <- factor(preditos_class, levels = levels(treino$classe))

# Visualizar as primeiras previsões de classe
head(preditos_class)

# Gerar a matriz de confusão
matriz_confusao <- confusionMatrix(preditos_class, treino$classe)
matriz_confusao

accuracy_treino <- mean(preditos_class == treino$classe)
print(paste("Acurácia do modelo de árvore de decisão na base de treino:", accuracy_treino))

# Plotando a Curva ROC

prob_pred <- predict(arvore, newdata = treino, type = "prob")


# Inicializar uma lista para armazenar os objetos ROC
roc_list <- list()

# Calcular e plotar a curva ROC para cada classe
for (i in 1:3) {
  # Criar a variável de resposta binária (one-vs-all)
  binary_response <- as.numeric(treino$classe == levels(treino$classe)[i])
  
  # Calcular a curva ROC
  roc_obj <- roc(binary_response, prob_pred[, i])
  roc_list[[i]] <- roc_obj
  
  # Plotar a curva ROC
  if (i == 1) {
    plot(roc_obj, main = "Curvas ROC para cada classe", col = i)
  } else {
    plot(roc_obj, add = TRUE, col = i)
  }
}

# Adicionar a legenda
legend("bottomright", legend = levels(treino$classe), col = 1:3, lty = 1)

# fim Curva ROC

# Relevância das variáveis no modelo
imp_arvore <- data.frame(importancia = arvore$variable.importance) %>% 
  rownames_to_column() %>% 
  arrange(importancia) %>% 
  rename(variavel = rowname) %>%
  mutate(variavel = fct_inorder(variavel))

ggplot(imp_arvore) +
  geom_segment(aes(x = variavel, y = 0, xend = variavel, yend = importancia), 
               size = 1.5, alpha = 0.7) +
  geom_point(aes(x = variavel, y = importancia, col = variavel), 
             size = 4, show.legend = F) +
  coord_flip() +
  labs(x = "Variável", y = "Importância") +
  theme_bw()

# Investigando se há overfitting


preditos_teste <- predict(arvore, teste)
preditos_teste

# Classificando os valores preditos com base no cutoff
preditos_class_teste <- apply(preditos_teste, 1, function(x) {
  if (x[1] > 0.5) {
    return("DH")
  } else if (x[2] > 0.5) {
    return("NO")
  } else {
    return("SL")
  }
})



preditos_class_teste <- factor(preditos_class_teste, levels = levels(teste$classe))
head(preditos_class_teste)


matriz_confusao <- confusionMatrix(preditos_class_teste, teste$classe)
matriz_confusao





#curva ROC teste

prob_pred_teste <- predict(arvore, newdata = teste, type = "prob")


# Inicializar uma lista para armazenar os objetos ROC
roc_list_teste <- list()

# Calcular e plotar a curva ROC para cada classe
for (i in 1:3) {
  binary_response_teste <- as.numeric(teste$classe == levels(teste$classe)[i])
  
  # Calcular a curva ROC
  roc_obj_teste <- roc(binary_response_teste, prob_pred_teste[, i])
  roc_list_teste[[i]] <- roc_obj_teste
  
  # Plotar a curva ROC
  if (i == 1) {
    plot(roc_obj_teste, main = "Curvas ROC para cada classe", col = i)
  } else {
    plot(roc_obj_teste, add = TRUE, col = i)
  }
}

# Adicionar a legenda
legend("bottomright", legend = levels(teste$classe), col = 1:3, lty = 1)




# GRID SEARCH GRID SEARCH 

# Parametrizando o grid
grid <- expand.grid(minsplit = seq(from = 10, to = 100, by = 5),
                    maxdepth = seq(from = 3, to = 12, by = 1),
                    minbucket = seq(from = 5, to = 50, by = 5))



# Criando uma lista para armazenar os resultados
modelos <- list()

# Gerando um processo iterativo
for (i in 1:nrow(grid)) {
  
  # Coletando os parâmetros do grid
  minsplit <- grid$minsplit[i]
  maxdepth <- grid$maxdepth[i]
  minbucket <- grid$minbucket[i]
  
  # Estimando os modelos e armazenando os resultados
  set.seed(132)
  modelos[[i]] <- rpart(
    formula = classe ~ versao_pelvica + lordose_lombar + declive_sacral + raio_pelvico + grau_deslizamento,
    data    = treino,
    parms = list(split = 'gini'),
    method  = "class",
    control = rpart.control(minsplit = minsplit, 
                            maxdepth = maxdepth,
                            minbucket = minbucket, 
                            cp = 0))
}

# Função para coletar parâmetro cp dos modelos
coleta_cp <- function(x) {
  min <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}

# Função para coletar o erro mínimo dos modelos
coleta_erro <- function(x) {
  min <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}

# Analisando os resultados
grid %>%
  mutate(
    cp = purrr::map_dbl(modelos, coleta_cp),
    erro = purrr::map_dbl(modelos, coleta_erro)
  ) %>%
  arrange(erro) %>% 
  slice_head(n = 10)

# Modelo final
set.seed(132)

arvore_grid <- rpart(formula = classe ~ versao_pelvica + lordose_lombar + declive_sacral + raio_pelvico + grau_deslizamento,
                     data = treino,
                     parms = list(split = "gini"),
                     method = "class",
                     control = rpart.control(minsplit = 15,
                                             maxdepth = 12,
                                             minbucket = 5,
                                             cp = 0))


preditos_grid <- predict(arvore_grid, teste)

# Classificando os valores preditos com base no cutoff
preditos_class_grid <- apply(preditos_grid, 1, function(x) {
  if (x[1] > 0.5) {
    return("DH")
  } else if (x[2] > 0.5) {
    return("NO")
  } else {
    return("SL")
  }
})


# Converter para fator com os níveis corretos
preditos_class_grid <- factor(preditos_class_grid, levels = levels(teste$classe))

# Visualizar as primeiras previsões de classe
head(preditos_class_grid)

# Gerando a matriz de confusão_grid final
matriz_confusao <- confusionMatrix(preditos_class_grid, teste$classe)
matriz_confusao

# Acurácia do modelo_grid final
accuracy <- mean(preditos_class_grid == teste$classe)
print(paste("Acurácia do modelo de árvore de decisão na base de teste:", accuracy))

# Plotando a árvore_grid final
rpart.plot(arvore_grid, type = 4, clip.right.labs = F, extra = 101, nn = T, cex = 0.6)

# CP árvore_grid
plotcp(arvore_grid)
printcp(arvore_grid)

imp_arvore_grid <- data.frame(importancia = arvore_grid$variable.importance) %>% 
  rownames_to_column() %>% 
  arrange(importancia) %>% 
  rename(variavel = rowname) %>%
  mutate(variavel = fct_inorder(variavel))

ggplot(imp_arvore_grid) +
  geom_segment(aes(x = variavel, y = 0, xend = variavel, yend = importancia), 
               size = 1.5, alpha = 0.7) +
  geom_point(aes(x = variavel, y = importancia, col = variavel), 
             size = 4, show.legend = F) +
  coord_flip() +
  labs(x = "Variável", y = "Importância") +
  theme_bw()

#curva ROC teste_grid

prob_pred_teste_grid <- predict(arvore_grid, newdata = teste, type = "prob")


# Inicializar uma lista para armazenar os objetos ROC
roc_list_teste <- list()

# Calcular e plotar a curva ROC para cada classe
for (i in 1:3) {
  # Criar a variável de resposta binária (one-vs-all)
  binary_response_teste <- as.numeric(teste$classe == levels(teste$classe)[i])
  
  # Calcular a curva ROC
  roc_obj_teste <- roc(binary_response_teste, prob_pred_teste_grid[, i])
  roc_list_teste[[i]] <- roc_obj_teste
  
  # Plotar a curva ROC
  if (i == 1) {
    plot(roc_obj_teste, main = "Curvas ROC para cada classe", col = i)
  } else {
    plot(roc_obj_teste, add = TRUE, col = i)
  }
}

# Adicionar a legenda
legend("bottomright", legend = levels(teste$classe), col = 1:3, lty = 1)
