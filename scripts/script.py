from turbofan_pkg.models import RNNTrainer

model = RNNTrainer(train_path = '/datadrive/Turbofan_Engine/df_train_cluster.pkl',
                   test_path = '/datadrive/Turbofan_Engine/df_test_cluster.pkl',
                   logger_path = '/home/rneves/temp/temp_logger/',
                   model_name = 'Turbofan_Test6',
                   train_log_interval = 100,
                   valid_log_interval = 100,
                   validation_split = 0.1,
                   use_script=True,
                   lr = 0.05,
                   number_steps_train = 50,
                   hidden_size = 128,
                   num_layers = 2,
                   cell_type = 'GRU',
                   kernel_size=10,
                   batch_size = 256,
                   num_epoch = 10,
                   number_features_input = 31,
                   number_features_output = 1,
                   loss_function = 'MSE',
                   optimizer = 'Adam',
                   normalizer = 'Standardization',
                   use_scheduler = False)

model.train(10)
