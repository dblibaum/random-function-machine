from constructor import Constructor

constructor = Constructor(datafile="",
                          test_fraction=.25,
                          n_objects=1000,
                          n_types=50,
                          max_attention_depth=100,
                          max_attention_objects=50,
                          computer_depth=100,
                          n_functions=1000,
                          stochastic=True,
                          batch_size=1000,
                          sep_features_targets=True,
                          save_file="best_protein_classifier.pkl")

constructor.evolve(1000)
