import train_engine

if __name__ == "__main__":
    # exp1 model eegnet
    config_eeg = {
        "exp_name":    "EEGnet_exp1",
        "num_classes": 2,
        "label":       "valence",
        "cv":          "stratified_kfold",
        "n_splits":    5,
        "Model" : "EEGnet"
    }
    train_engine.run_experiment(config=config_eeg)