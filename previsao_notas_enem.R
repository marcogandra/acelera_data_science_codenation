######### AceleraDev - Data Science ##########
####### Predição de notas ENEM 2016 ##########

## pacotes
library(tidyverse)
library(h2o)

## leitura dos dados
treino = read_csv("train.csv")
teste = read_csv("test.csv")

## variáveis do modelo
# dependente
dep_var = "NU_NOTA_MT"

# independentes
ind_vars = c("SG_UF_RESIDENCIA",
             "NU_IDADE",
             "TP_SEXO",
             "TP_COR_RACA",
             "TP_NACIONALIDADE",
             "TP_ST_CONCLUSAO",
             "TP_ANO_CONCLUIU",
             "TP_ESCOLA",
             "TP_ENSINO",
             "IN_TREINEIRO",
             "TP_DEPENDENCIA_ADM_ESC",
             "IN_BAIXA_VISAO",
             "IN_CEGUEIRA",
             "IN_SURDEZ",
             "IN_DISLEXIA",
             "IN_DISCALCULIA",
             "IN_SABATISTA",
             "IN_GESTANTE",
             "IN_IDOSO",
             "TP_PRESENCA_CN",
             "TP_PRESENCA_CH",
             "TP_PRESENCA_LC",
             "NU_NOTA_CN",
             "NU_NOTA_CH",
             "NU_NOTA_LC",
             "TP_LINGUA",
             "TP_STATUS_REDACAO",
             "NU_NOTA_REDACAO",
             "Q001",
             "Q002",
             "Q006",
             "Q024",
             "Q025",
             "Q025",
             "Q027",
             "Q047")


## inicializa o h2o
h2o.init()

## autoML
automl = h2o.automl(y = dep_var,
                    x = ind_vars,
                    training_frame = as.h2o(treino),
                    nfolds = 5,
                    stopping_metric = "RMSE",
                    sort_metric = "RMSE",
                    max_models = 15,
                    seed = 65)

as_tibble(automl@leaderboard)

# perfomance do líder
h2o.performance(automl@leader)

# predições no banco de teste
pred_teste = predict(automl@leader, as.h2o(teste %>% select(all_of(ind_vars)))) %>% as_tibble()

# adiciona na base
teste_pred = bind_cols(teste, pred_teste)

# para os que não fizeram nenhuma prova, definir 0
teste_pred = teste_pred %>%
  mutate(NU_NOTA_MT = case_when(
    ((TP_PRESENCA_CH == 0) & (TP_PRESENCA_CN == 0) & (TP_PRESENCA_LC == 0)) ~ 0,
    ((TP_PRESENCA_CH != 0) | (TP_PRESENCA_CN != 0) | (TP_PRESENCA_LC != 0)) ~ predict
  ))

# separa os dados finais
dados_finais = teste_pred %>%
  select(NU_INSCRICAO, NU_NOTA_MT)

# salva o arquivo
write_csv(dados_finais, "answer.csv")

# encerra o cluster
h2o.shutdown(prompt = FALSE)
