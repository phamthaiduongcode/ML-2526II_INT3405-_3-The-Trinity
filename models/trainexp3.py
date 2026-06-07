import train_engine

if __name__ == "__main__":
    config_exp3 = {
        "exp_name":    "exp3_valence_logo",
        "num_classes": 2,
        "label":       "valence",
        "cv":          "loso",
    }
    train_engine.run_experiment(config= config_exp3)