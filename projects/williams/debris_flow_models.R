library(tidyverse)
library(tidymodels)
library(terra)
library(sf)

debris_flow <- read_csv('projects/williams/debris_flow.csv')

aridity_index <- rast('projects/williams/ai_.tif')

df_sf <- debris_flow %>% st_as_sf(coords = c('UTM_X', 'UTM_Y'))

df_sf_11 <- df_sf %>% filter(UTM_Zone == 11) %>% st_set_crs(32611)

df_sf_12 <- df_sf %>% filter(UTM_Zone == 12) %>% st_set_crs(32612)

df_sf_13 <- df_sf %>% filter(UTM_Zone == 13) %>% st_set_crs(32613)

library(rgee)
library(wildlandhydRo)
ee_Initialize()





library(nhdplusTools)
library(furrr)
library(future)

plan(multisession(workers = availableCores()-1))
basins_11 <- df_sf_11 %>%
             group_by(Fire_SegID) %>%
             slice(n = 1) %>%
             ungroup() %>%
             split(.$Fire_SegID) %>%
             future_map(~get_split_catchment(.))

fire_ids <- names(basins_11)

basins_11_final <- basins_11 %>%
                   map2(.x = .,.y = fire_ids,safely(~.x %>% mutate(Fire_SegID = .y))) %>%
                   map(~.x[['result']]) %>% bind_rows() %>%
                   filter(id == 'splitCatchment')
basins_11_final %>% mapview::mapview() + mapview::mapview(df_sf_11)
not_in <- df_sf_11 %>%
  group_by(Fire_SegID) %>%
  slice(n = 1) %>%
  st_drop_geometry() %>%
  filter(!Fire_SegID %in% fire_ids)

catchments_11 <- basins_11_final %>%
                 mutate()


c11_int <- df_sf_11 %>%
  group_by(Fire_SegID) %>%
  slice(n=1) %>%
  ungroup() %>%
  st_intersects(st_transform(catchments_11, st_crs(df_sf_11)))

c11_int <- lengths(c11_int) > 0

basins_11_get <- df_sf_11[]

basins_11_get %>% group_by(Fire_SegID) %>% slice(n = 1)
ggplot() + geom_sf(fill = NA) +
  geom_sf(data = df_sf_11 %>% filter(state != 'MT'))


df_sf <- df_sf

df_sf_11$ai <- as.numeric(extract(aridity_index, vect(st_transform(df_sf_11, 4326)))$b1)
df_sf_12$ai <- as.numeric(extract(aridity_index, vect(st_transform(df_sf_12, 4326)))$b1)
df_sf_13$ai <- as.numeric(extract(aridity_index, vect(st_transform(df_sf_13, 4326)))$b1)

debris_flow <- bind_rows(df_sf_11, df_sf_12, df_sf_13) %>% st_drop_geometry()

mapview::mapview(df_sf)

train <- debris_flow %>% filter(Database == 'Training') %>%
  select(Response, Acc015_mm, KF, `dNBR/1000`, PropHM23, ai) %>%
  mutate(Response = factor(Response)) %>%
  filter(!is.na(`dNBR/1000`), !is.na(Acc015_mm))

test <- debris_flow %>% filter(Database == 'Test') %>%
  select(Response, Acc015_mm, KF, `dNBR/1000`, PropHM23, ai) %>%
  mutate(Response = factor(Response)) %>%
  filter(!is.na(`dNBR/1000`), !is.na(Acc015_mm))

debris_flow %>%
  filter(KF < 1) %>%
  ggplot(aes(Acc030_mm, ai, color = Response)) +
  geom_point()

sp <- make_splits(train, assessment = test)

train <- training(sp)

test <- testing(sp)


xgb_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(), min_n = tune(), loss_reduction = tune(),
  mtry = tune(), learn_rate = tune(), sample_size = tune()
) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  learn_rate(),
  finalize(mtry(), train),
  size = 20
)

xgb_wf <- workflow() %>%
  add_formula(Response~.) %>%
  add_model(xgb_spec)

set.seed(234)

vb_folds <- vfold_cv(data = train, strata = Response)
doParallel::registerDoParallel()

set.seed(2345)

xgb_res <- tune_grid(
  xgb_wf,
  resamples = vb_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE)
)


xgb_res %>% collect_metrics()


autoplot(xgb_res)

best_acc <- select_best(xgb_res, metric = 'accuracy')

xgb_final <- finalize_workflow(xgb_wf, best_acc)

xgb_fit <- last_fit(xgb_final, sp)

xgb_fit %>%
  extract_workflow() %>%
  predict(test) %>%
  bind_cols(tibble(obs = test$Response)) %>%
  mutate(acc = if_else(.pred_class == obs, 1, 0)) %>%
  pull(acc) %>% sum()

273/428
182+20+131+95



xgb_fit %>%
  extract_workflow() %>%
  predict(test) %>%
  bind_cols(tibble(obs = test$Response)) %>%
  conf_mat(truth = 'obs', estimate = '.pred_class')

20/428

(86)/(86+120+29)
(95)/(95+20+131)

#rf

tune_spec <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()
) %>%
  set_mode('classification') %>%
  set_engine('ranger')

tune_workflow <- workflow() %>%
  add_formula(Response~.) %>%
  add_model(tune_spec)

set.seed(234)

vb_folds <- vfold_cv(data = train, strata = Response)
doParallel::registerDoParallel()

set.seed(2345)

rf_res <- tune_grid(
  tune_workflow,
  resamples = vb_folds,
  grid = 20,
  control = control_grid(save_pred = TRUE)
)

rf_res %>% collect_metrics()

autoplot(rf_res)

best_acc_rf <- select_best(rf_res, metric = 'accuracy')

rf_final <- finalize_workflow(tune_workflow, best_acc_rf)

rf_fit <- last_fit(rf_final, sp)

rf_fit %>%
  extract_workflow() %>%
  predict(test) %>%
  bind_cols(tibble(obs = test$Response)) %>%
  mutate(acc = if_else(.pred_class == obs, 1, 0)) %>%
  pull(acc) %>% sum()

266/428

rf_fit %>%
  extract_workflow() %>%
  predict(test) %>%
  bind_cols(tibble(obs = test$Response)) %>%
  conf_mat(truth = 'obs', estimate = '.pred_class')

(92)/(92+175+23)
