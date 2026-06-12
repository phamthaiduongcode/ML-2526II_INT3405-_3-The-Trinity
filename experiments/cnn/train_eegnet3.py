import train_engine

if __name__ == "__main__":
    # Exp3 model eegnet
    config_exp3_EEGnet = {
        "exp_name":    "exp3_EEGnet",
        "num_classes": 2,
        "label":       "valence",
        "cv":          "loso",
        "Model" : "EEGnet"
    }
    train_engine.run_experiment(config=config_exp3_EEGnet)