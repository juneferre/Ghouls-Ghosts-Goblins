## ---- Libraries ---- ##
library(tidymodels)
library(tidyverse)
library(vroom)
library(doParallel)

# -----------------------------------------------------------------------------
# Read in data sets
# -----------------------------------------------------------------------------
trainData <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/spooky/train.csv") |>
  mutate(type = as.factor(type))

testData  <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/spooky/test.csv")

## ---- Recipe ---- ##
my_recipe <- recipe(type ~ ., data = trainData) %>%
  step_normalize(all_numeric_predictors())


## ---- Model ---- ##
svmRadial <- svm_rbf(
  rbf_sigma = tune(),
  cost = tune()
) |>
  set_mode("classification") |>
  set_engine("kernlab")

## ---- Workflow ---- ##
my_workflow <- workflow() |>
  add_model(svmRadial) |>
  add_recipe(my_recipe)

## ---- Tuning Grid ---- ##
tuning_grid <- grid_regular(
  cost(range = c(-10, 10)),         # log2 scale
  rbf_sigma(range = c(-5, 5)),    # you forgot to grid over sigma too!
  levels = 10
)

## ---- Cross-validation ---- ##
folds <- vfold_cv(trainData, v = 10, repeats = 3)

## ---- Tune Model ---- ##
CV_results <- my_workflow |>
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc, accuracy, sensitivity, specificity)
  )

## ---- Select Best ---- ##
best_tune_svm <- CV_results |>
  select_best(metric = "accuracy")

## ---- Final Workflow ---- ##
final_wf <- finalize_workflow(
  my_workflow,
  best_tune_svm
)

## ---- Fit Final Model ---- ##
final_fit <- final_wf |>
  fit(data = trainData)

## ---- Predict on Test ---- ##
svm_predictions <- predict(final_fit, new_data = testData, type = "class")

# ---- Prepare Submission ---- #
submission <- testData %>%
  bind_cols(svm_predictions) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

# ---- Export CSV ---- #
vroom_write(x = submission, file = "./SVM.csv", delim = ",")