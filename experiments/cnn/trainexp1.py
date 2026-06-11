import train_engine

if __name__ == "__main__":
    # Exp1: Valence binary — StratifiedKFold 5 folds
    config_exp1 = {
        "exp_name":    "exp1_valence_binary",
        "num_classes": 2,
        "label":       "valence",
        "cv":          "stratified_kfold",
        "n_splits":    5,
        "Model" : "CNN"
    }
    train_engine.run_experiment(config=config_exp1)