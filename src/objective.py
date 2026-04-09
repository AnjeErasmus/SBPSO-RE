from sklearn.metrics import f1_score

def make_objective(X_train, y_train, X_test, y_test, U):
    """
    Creates and returns an Optuna objective function.

    The returned objective function is called once per Optuna trial. In each
    trial, Optuna suggests a new set of hyperparameters (`c1`, `c2`, `c3`,
    `c4`, `k`, `swarm_size`, and `max_iterations`). These values are passed
    into `separate_and_conquer`, which trains your SBPSO / SeCo rule-learning
    system on the training data and produces a ruleset. That ruleset is then
    evaluated on the test set using `apply_ruleset_v2`, which generates
    predictions. Finally, the macro F1-score between the true labels and the
    predicted labels is returned as the objective value. Optuna uses this score
    to compare trials and search for the hyperparameter combination that gives
    the best classification performance across all classes.
    """
    def objective(trial):
        # Suggest hyperparameters
        param_set = {
            "c1": trial.suggest_float("c1", 0.5, 2.5),
            "c2": trial.suggest_float("c2", 0.5, 2.5),
            "c3": trial.suggest_float("c3", 0.5, 2.5),
            "c4": trial.suggest_float("c4", 0.5, 2.5),
            "k": trial.suggest_int("k", 2, 10),
            "swarm_size": trial.suggest_int("swarm_size", 10, 100),
            "max_iterations": trial.suggest_int("max_iterations", 10, 200),
        }

        # Train SBPSO / SeCo
        ruleset = separate_and_conquer(
            c1=param_set["c1"],
            c2=param_set["c2"],
            c3=param_set["c3"],
            c4=param_set["c4"],
            k=param_set["k"],
            swarm_size=param_set["swarm_size"],
            max_iterations=param_set["max_iterations"],
            data_frame=X_train,
            label_frame=y_train,
            U=U,
            f=single_rule_fitness,
        )

        # Evaluate ruleset
        y_pred, rule_stats, overlaps = apply_ruleset_v2(
            ruleset,
            X_test,
            y_test,
            default_class=y_test.mode()[0],
        )

        # Use macro F1 score as optimization objective
        return f1_score(y_test, y_pred, average="macro")

    return objective