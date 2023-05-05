#' Create small models without mlflow
#'
#'
#'
#' @param
#
#' @return
#'
#' @examples
#'
#' @export
create_multiple_models_lite <- function(
    data, target, experiment_name, n_cores = detectCores() / 4,
    n_prop = 2/3, n_repeats = 50, directory = getwd()
){
  is_Windows <- Sys.info()[["sysname"]] == "Windows"

  directory <- file.path(directory, experiment_name)
  if (!dir.exists(directory)) {
    dir.create(directory)
  } else {
    logger::log_error("Experiment with given name already exists. Please introduce different name")
    stop("Experiment with given name already exists. Please introduce different name")
  }

  last_run <- Sys.Date()

  data <- data %>%
    reduce(full_join, by = c(target$target_variable, target$id_variable)) %>%
    select(-target$id_variable) %>%
    relocate(target$target_variable, .after = last_col())

  chunks <- c(
    combn(ncol(data) - 1, 2, simplify = FALSE),
    combn(ncol(data) - 1, 3, simplify = FALSE)
  ) %>%
    lapply(c, ncol(data)) %>%
    sample()

  # Save
  save(data, directory, chunks, last_run, file = file.path(directory, "raw_data.RData"))

  models_dir <- file.path(directory, "models")
  dir.create(models_dir)

  logger::log_info("Experiment {experiment_name} created")

  if (!is_Windows) {
    cl <- parallel::makeForkCluster(n_cores)
  }
  # Use proper iterator function for different OS
  iterator_func <- if (!is_Windows) {
    function(...) { parallel::parLapplyLB(cl, ...) }
  } else {
    lapply
  }

  logger::log_info("Starting modelling experiment")

  models <- iterator_func(chunks, function(column_indices) {
    set.seed(sample(seq_len(100000), size = 1))

    model_data <- na.omit(select(data, column_indices))
    model_name <- paste(colnames(select(model_data, -last_col())), collapse = " + ")
    # Model names are unique within an analysis
    model_id <- digest::digest(model_name)
    model_dir <- file.path(models_dir, model_id)
    dir.create(model_dir)

    tryCatch({
      n_groups <- model_data %>%
        count(!!rlang::sym(target$target_variable)) %>%
        pivot_wider(
          names_from = all_of(target$target_variable), names_prefix = "n_", values_from = n
        ) %>%
        lapply(identity)

      # Log parameters
      list(
        model_name = model_name,
        groups = n_groups,
        resampling_stategy = "mc_cv",
        prop = n_prop,
        times = n_repeats
      ) %>%
        jsonlite::write_json(file.path(model_dir, "params.json"),
                             pretty = TRUE, auto_unbox = TRUE)

      # Define model
      data_recipe <- recipes::recipe(model_data) %>%
        recipes::update_role(target$target_variable, new_role = "outcome") %>%
        recipes::update_role(recipes::has_role(NA), new_role = "predictor")
      # step_normalize(all_predictors()) #should the data be normalized?

      model_spec <- parsnip::logistic_reg() %>%  # model type
        parsnip::set_engine(engine = "glm") %>%  # model engine
        parsnip::set_mode("classification")   # model mode

      # Subsampling
      resample <- rsample::mc_cv(
        model_data, prop = n_prop, times = n_repeats, strata = target$target_variable
      )

      # Define workflow
      model_wflow <- workflows::workflow() %>% # use workflow function
        workflows::add_recipe(data_recipe) %>%   # use the recipe
        workflows::add_model(model_spec)

      model_res <- model_wflow %>%
        tune::fit_resamples(
          resamples = resample,
          metrics = yardstick::metric_set(
            yardstick::mcc,
            yardstick::recall,
            yardstick::precision,
            yardstick::accuracy,
            yardstick::roc_auc,
            yardstick::sens,
            yardstick::spec,
            yardstick::ppv,
            yardstick::npv,
            yardstick::pr_auc,
            yardstick::f_meas,
          ),
          control = tune::control_resamples(
            save_pred = TRUE, allow_par = F)
        )

      results <- model_res %>%
        tune::collect_metrics(summarize = TRUE) %>%
        select(.metric, mean) %>%
        pivot_wider(names_from = .metric, values_from = mean)

      # Log metrics
      jsonlite::write_json(
        lapply(metrics, identity),
        file.path(model_dir, "metrics.json"), pretty = TRUE, auto_unbox = TRUE
      )

      # create final model on complete data
      fitted_model <- fit(model_wflow, model_data)

      saveRDS(
        carrier::crate(
          function(data_to_fit) {
            dplyr::bind_cols(
              workflows:::predict.workflow(fitted_model, as.data.frame(data_to_fit), type = "class"),
              workflows:::predict.workflow(fitted_model, as.data.frame(data_to_fit), type = "prob")
            )
          },
          fitted_model = fitted_model
        ),
        file.path(model_dir, "model.Rds")
      )

      # explain prediction
      explainer_lr <- DALEXtra::explain_tidymodels(
        fitted_model,
        data = model_data,
        y = model_data[[target$target_variable]] == target$positive_class_indication,
        label = "lr",
        verbose = FALSE
      )

      saveRDS(
        carrier::crate(
          function(data_to_fit) {
            DALEX::predict_parts(explainer_lr, data_to_fit, type = "shap")
          },
          explainer_lr = explainer_lr
        ),
        file.path(model_dir, "explainer.Rds")
      )

      # save raw data
      saveRDS(
        model_data, file.path(model_dir, "data.Rds")
      )

      # return metrics
      results
    }, error = function(error_condition) {
      message(error_condition)
      return(NULL)
    })
  })
  logger::log_info("Modelling experiment ended")

  # save models' stats
  saveRDS(models, file.path(models_dir, "models_stats.Rds"))

  # Multicore off
  if (!is_Windows) {
    parallel::stopCluster(cl)
    rm(cl)
  }

  models
}
