import train_engine

if __name__ == "__main__":
    config_exp3 = {
        "exp_name":    "exp3_valence_loso",
        "num_classes": 2,
        "label":       "valence",
        "cv":          "loso",
        "patience_es" : 20,
        "Model" : "CNN"
    }
    train_engine.run_experiment(config= config_exp3)