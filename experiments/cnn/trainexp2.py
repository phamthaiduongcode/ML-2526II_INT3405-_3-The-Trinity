import train_engine

if __name__ == "__main__": 
    config_exp2 = {
        "exp_name":    "exp2_4class_va",
        "num_classes": 4,
        "label":       "4class",
        "cv":          "stratified_kfold",
        "n_splits":    5,
        "Model" : "CNN"
    }
    train_engine.run_experiment(config=config_exp2)