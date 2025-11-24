# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
library(vroom)
library(tidymodels)
library(embed)
library(workflows)
library(themis)
library(ranger)

# -----------------------------------------------------------------------------
# Read in data sets
# -----------------------------------------------------------------------------
train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/spooky/train.csv") |>
  mutate(type = as.factor(type))
test  <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/spooky/test.csv")

table(train$type)
# -----------------------------------------------------------------------------
# Recipe for multiclass classification
# -----------------------------------------------------------------------------
my_recipe <- recipe(type ~ ., data = train) %>% 
  step_zv(all_predictors())  %>%  # remove zero-variance predictors 
  step_normalize(all_numeric_predictors()) %>% # standardize numeric features 
  step_dummy(all_nominal_predictors()) # one-hot encode categorical vars

  

prepped <- prep(my_recipe, verbose = TRUE)
new_data <- bake(prepped, new_data = NULL)

# -----------------------------------------------------------------------------
# Define model
# -----------------------------------------------------------------------------
my_mod <- rand_forest(mtry = 3, trees = 2000, min_n = 2) %>%
  set_engine("ranger",
             sample.fraction = 0.8,
             importance = "impurity",
             class.weights = c(Ghost = 1, Goblin = 1.2, Ghoul = 1.1)) %>%
  set_mode("classification")



# -----------------------------------------------------------------------------
# Workflow
# -----------------------------------------------------------------------------
wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data=train)


# -----------------------------------------------------------------------------
# Predictions
# -----------------------------------------------------------------------------

preds <- predict(wf, new_data = test)

# -----------------------------------------------------------------------------
# Kaggle Submission
# -----------------------------------------------------------------------------

submission <- tibble(
  id = test$id,
  type = preds$.pred_class)

vroom_write(submission, file = "RandomForest9.csv", delim = ",")
