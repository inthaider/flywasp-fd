# 2023-11-15

<!-- *************************************************************** -->
<!--                         MPS #2 TRAINING                         -->
<!-- *************************************************************** -->
Training RNN Model (mps)
==============================

mps
Using kwargs for data loaders

Epochs:   0%|          | 0/10 [00:00<?, ?it/s]src.models.train_eval - INFO - Number of batches: 1594

src.models.train_eval - INFO - Batch size: 4096

Epoch 1/10
-------------------------------

Training the model...
Training...:   0%|          | 2/1594 [00:08<1:31:46,  3.46s/it]Loss: 0.765352  [ 4096/6527546]
Training...:  10%|█         | 161/1594 [00:30<03:12,  7.45it/s]Loss: 0.536670  [655360/6527546]
Training...:  20%|██        | 320/1594 [00:56<03:26,  6.16it/s]Loss: 0.512099  [1306624/6527546]
Training...:  30%|███       | 479/1594 [01:24<03:06,  5.97it/s]Loss: 0.500141  [1957888/6527546]
Training...:  40%|███▉      | 637/1594 [01:50<02:41,  5.91it/s]Loss: 0.359428  [2609152/6527546]
Training...:  50%|█████     | 797/1594 [02:18<02:11,  6.07it/s]Loss: 0.335490  [3260416/6527546]
Training...:  60%|█████▉    | 956/1594 [02:46<01:49,  5.82it/s]Loss: 0.327499  [3911680/6527546]
Training...:  70%|██████▉   | 1115/1594 [03:12<01:21,  5.90it/s]Loss: 0.323656  [4562944/6527546]
Training...:  80%|███████▉  | 1274/1594 [03:36<00:45,  7.01it/s]Loss: 0.321425  [5214208/6527546]
Training...:  90%|████████▉ | 1433/1594 [04:00<00:24,  6.52it/s]Loss: 0.319974  [5865472/6527546]
Training...: 100%|█████████▉| 1592/1594 [04:24<00:00,  7.03it/s]Loss: 0.318958  [6516736/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 1:
 Sum of squared gradients :     450.3048
 Sum of squared parameters:   36835.2830

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 75.1%,Avg loss: 0.407116, F1 Score: 0.0047

src.models.train_eval - INFO - Training Epoch 1 took 4.8 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:
 Accuracy: 74.9%,Avg loss: 0.384540, F1 Score: 0.4303

src.models.train_eval - INFO - Evaluating Epoch 1 took 1.1 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2444230, 1.0: 819544}

src.models.train_eval - INFO - Epoch 1 Metrics --
Train Loss: 0.4071,
Train F1 Score: 0.0047,
Train Acc: 75.1%,
Test Loss: 0.3845,
Test Acc: 0.7494,
Test F1: 0.4303,
Test PR AUC: 0.0109

Epochs:  10%|█         | 1/10 [05:54<53:06, 354.03s/it]src.models.train_eval - INFO - Number of batches: 1594

src.models.train_eval - INFO - Batch size: 4096

Epoch 2/10
-------------------------------

Training the model...
Training...:   0%|          | 2/1594 [00:08<1:30:56,  3.43s/it]Loss: 0.494536  [ 4096/6527546]
Training...:  10%|█         | 161/1594 [00:29<02:57,  8.09it/s]Loss: 0.524063  [655360/6527546]
Training...:  20%|██        | 320/1594 [00:53<03:05,  6.86it/s]Loss: 0.503269  [1306624/6527546]
Training...:  30%|███       | 479/1594 [01:21<03:38,  5.11it/s]Loss: 0.318354  [1957888/6527546]
Training...:  40%|████      | 638/1594 [01:49<02:51,  5.58it/s]Loss: 0.317732  [2609152/6527546]
Training...:  50%|████▉     | 796/1594 [02:25<02:53,  4.61it/s]Loss: 0.317245  [3260416/6527546]
Training...:  60%|█████▉    | 956/1594 [02:54<02:01,  5.24it/s]Loss: 0.316854  [3911680/6527546]
Training...:  70%|██████▉   | 1115/1594 [03:22<01:17,  6.18it/s]Loss: 0.316532  [4562944/6527546]
Training...:  80%|███████▉  | 1273/1594 [03:52<01:02,  5.16it/s]Loss: 0.316263  [5214208/6527546]
Training...:  90%|████████▉ | 1433/1594 [04:22<00:26,  6.04it/s]Loss: 0.316034  [5865472/6527546]
Training...: 100%|█████████▉| 1592/1594 [04:49<00:00,  6.82it/s]Loss: 0.315838  [6516736/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 2:
 Sum of squared gradients :     427.6891
 Sum of squared parameters:   40157.4520

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 75.0%,Avg loss: 0.373720, F1 Score: 0.0048

src.models.train_eval - INFO - Training Epoch 2 took 5.2 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:
 Accuracy: 74.9%,Avg loss: 0.380534, F1 Score: 0.4303

src.models.train_eval - INFO - Evaluating Epoch 2 took 1.3 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2444268, 1.0: 819506}

src.models.train_eval - INFO - Epoch 2 Metrics --
Train Loss: 0.3737,
Train F1 Score: 0.0048,
Train Acc: 75.0%,
Test Loss: 0.3805,
Test Acc: 0.7494,
Test F1: 0.4303,
Test PR AUC: 0.0130

Epochs:  20%|██        | 2/10 [12:21<49:49, 373.72s/it]src.models.train_eval - INFO - Number of batches: 1594

src.models.train_eval - INFO - Batch size: 4096

Epoch 3/10
-------------------------------

Training the model...
Training...:   0%|          | 2/1594 [00:08<1:30:30,  3.41s/it]Loss: 0.487404  [ 4096/6527546]
Training...:  10%|█         | 161/1594 [00:30<03:24,  7.02it/s]Loss: 0.520519  [655360/6527546]
Training...:  20%|██        | 320/1594 [00:53<03:32,  5.99it/s]Loss: 0.490114  [1306624/6527546]
Training...:  30%|███       | 479/1594 [01:20<03:02,  6.13it/s]Loss: 0.315651  [1957888/6527546]
Training...:  40%|████      | 638/1594 [01:47<02:36,  6.09it/s]Loss: 0.315501  [2609152/6527546]
Training...:  50%|█████     | 797/1594 [02:15<02:15,  5.90it/s]Loss: 0.315368  [3260416/6527546]
Training...:  60%|█████▉    | 956/1594 [02:47<01:54,  5.59it/s]Loss: 0.315251  [3911680/6527546]
Training...:  70%|██████▉   | 1115/1594 [03:22<01:19,  6.01it/s]Loss: 0.315145  [4562944/6527546]
Training...:  80%|███████▉  | 1274/1594 [03:50<00:56,  5.64it/s]Loss: 0.315051  [5214208/6527546]
Training...:  90%|████████▉ | 1433/1594 [04:16<00:23,  7.00it/s]Loss: 0.314965  [5865472/6527546]
Training...: 100%|█████████▉| 1592/1594 [04:41<00:00,  7.17it/s]Loss: 0.314888  [6516736/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 3:
 Sum of squared gradients :     510.2147
 Sum of squared parameters:   41409.8352

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 75.7%,Avg loss: 0.370905, F1 Score: 0.0049

src.models.train_eval - INFO - Training Epoch 3 took 5.0 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:
 Accuracy: 76.5%,Avg loss: 0.374336, F1 Score: 0.4354

src.models.train_eval - INFO - Evaluating Epoch 3 took 1.0 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2494752, 1.0: 769022}

src.models.train_eval - INFO - Epoch 3 Metrics --
Train Loss: 0.3709,
Train F1 Score: 0.0049,
Train Acc: 75.7%,
Test Loss: 0.3743,
Test Acc: 0.7649,
Test F1: 0.4354,
Test PR AUC: 0.0142

Epochs:  30%|███       | 3/10 [18:24<43:00, 368.60s/it]src.models.train_eval - INFO - Number of batches: 1594

src.models.train_eval - INFO - Batch size: 4096

Epoch 4/10
-------------------------------

Training the model...
Training...:   0%|          | 2/1594 [00:08<1:29:52,  3.39s/it]Loss: 0.493824  [ 4096/6527546]
Training...:  10%|█         | 161/1594 [00:32<03:43,  6.40it/s]Loss: 0.511729  [655360/6527546]
Training...:  20%|██        | 320/1594 [00:56<03:06,  6.82it/s]Loss: 0.494093  [1306624/6527546]
Training...:  30%|███       | 479/1594 [01:23<02:52,  6.47it/s]Loss: 0.314797  [1957888/6527546]
Training...:  40%|████      | 638/1594 [01:51<02:40,  5.95it/s]Loss: 0.314732  [2609152/6527546]
Training...:  50%|█████     | 797/1594 [02:20<02:25,  5.48it/s]Loss: 0.314672  [3260416/6527546]
Training...:  60%|█████▉    | 955/1594 [02:52<03:11,  3.33it/s]Loss: 0.314618  [3911680/6527546]
Training...:  70%|██████▉   | 1115/1594 [03:24<01:25,  5.60it/s]Loss: 0.314567  [4562944/6527546]
Training...:  80%|███████▉  | 1274/1594 [03:52<00:50,  6.29it/s]Loss: 0.314520  [5214208/6527546]
Training...:  90%|████████▉ | 1433/1594 [04:19<00:27,  5.96it/s]Loss: 0.314476  [5865472/6527546]
Training...: 100%|█████████▉| 1592/1594 [04:46<00:00,  6.50it/s]Loss: 0.314436  [6516736/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 4:
 Sum of squared gradients :     786.7433
 Sum of squared parameters:   42328.3376

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 75.1%,Avg loss: 0.369737, F1 Score: 0.0048

src.models.train_eval - INFO - Training Epoch 4 took 5.1 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:
 Accuracy: 75.0%,Avg loss: 0.375452, F1 Score: 0.4305

src.models.train_eval - INFO - Evaluating Epoch 4 took 1.1 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2445864, 1.0: 817910}

src.models.train_eval - INFO - Epoch 4 Metrics --
Train Loss: 0.3697,
Train F1 Score: 0.0048,
Train Acc: 75.1%,
Test Loss: 0.3755,
Test Acc: 0.7499,
Test F1: 0.4305,
Test PR AUC: 0.0131

Epochs:  40%|████      | 4/10 [24:38<37:06, 371.08s/it]src.models.train_eval - INFO - Number of batches: 1594

src.models.train_eval - INFO - Batch size: 4096

Epoch 5/10
-------------------------------

Training the model...
Training...:   0%|          | 2/1594 [00:08<1:32:09,  3.47s/it]Loss: 0.488456  [ 4096/6527546]
Training...:  10%|█         | 161/1594 [00:29<02:49,  8.45it/s]Loss: 0.537666  [655360/6527546]
Training...:  20%|██        | 320/1594 [00:50<02:43,  7.81it/s]Loss: 0.490697  [1306624/6527546]
Training...:  30%|███       | 479/1594 [01:13<02:41,  6.91it/s]Loss: 0.314338  [1957888/6527546]
Training...:  40%|████      | 638/1594 [01:36<02:22,  6.71it/s]Loss: 0.314305  [2609152/6527546]
Training...:  50%|█████     | 797/1594 [02:00<02:12,  6.01it/s]Loss: 0.314275  [3260416/6527546]
Training...:  60%|█████▉    | 955/1594 [02:24<02:21,  4.52it/s]Loss: 0.314246  [3911680/6527546]
Training...:  70%|██████▉   | 1115/1594 [02:48<01:08,  6.98it/s]Loss: 0.314218  [4562944/6527546]
Training...:  80%|███████▉  | 1274/1594 [03:11<00:51,  6.27it/s]Loss: 0.314193  [5214208/6527546]
Training...:  90%|████████▉ | 1433/1594 [03:34<00:23,  6.88it/s]Loss: 0.314168  [5865472/6527546]
Training...: 100%|█████████▉| 1592/1594 [03:56<00:00,  7.33it/s]Loss: 0.314145  [6516736/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 5:
 Sum of squared gradients :     790.1469
 Sum of squared parameters:   43133.5143

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 76.3%,Avg loss: 0.368563, F1 Score: 0.0050

src.models.train_eval - INFO - Training Epoch 5 took 4.2 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:
 Accuracy: 75.3%,Avg loss: 0.376987, F1 Score: 0.4317

src.models.train_eval - INFO - Evaluating Epoch 5 took 0.9 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2457618, 1.0: 806156}

src.models.train_eval - INFO - Epoch 5 Metrics --
Train Loss: 0.3686,
Train F1 Score: 0.0050,
Train Acc: 76.3%,
Test Loss: 0.3770,
Test Acc: 0.7535,
Test F1: 0.4317,
Test PR AUC: 0.0108

Epochs:  50%|█████     | 5/10 [29:44<28:58, 347.61s/it]src.models.train_eval - INFO - Number of batches: 1594

src.models.train_eval - INFO - Batch size: 4096

Epoch 6/10
-------------------------------

Training the model...
Training...:   0%|          | 2/1594 [00:06<1:11:04,  2.68s/it]Loss: 0.480297  [ 4096/6527546]
Training...:  10%|█         | 161/1594 [00:25<02:53,  8.28it/s]Loss: 0.511796  [655360/6527546]
Training...:  20%|██        | 320/1594 [00:44<02:36,  8.13it/s]Loss: 0.486184  [1306624/6527546]
Training...:  30%|███       | 479/1594 [01:05<02:20,  7.93it/s]Loss: 0.314085  [1957888/6527546]
Training...:  40%|████      | 638/1594 [01:26<02:11,  7.28it/s]Loss: 0.314066  [2609152/6527546]
Training...:  50%|█████     | 797/1594 [01:49<01:48,  7.37it/s]Loss: 0.314048  [3260416/6527546]
Training...:  60%|█████▉    | 956/1594 [02:17<01:41,  6.32it/s]Loss: 0.314030  [3911680/6527546]
Training...:  70%|██████▉   | 1115/1594 [02:42<01:10,  6.80it/s]Loss: 0.314013  [4562944/6527546]
Training...:  80%|███████▉  | 1274/1594 [03:07<00:59,  5.36it/s]Loss: 0.313997  [5214208/6527546]
Training...:  90%|████████▉ | 1433/1594 [03:33<00:25,  6.38it/s]Loss: 0.313982  [5865472/6527546]
Training...: 100%|█████████▉| 1592/1594 [04:00<00:00,  6.07it/s]Loss: 0.313967  [6516736/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 6:
 Sum of squared gradients :     789.4031
 Sum of squared parameters:   43806.3279

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 77.1%,Avg loss: 0.367654, F1 Score: 0.0052

src.models.train_eval - INFO - Training Epoch 6 took 4.3 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:
 Accuracy: 77.2%,Avg loss: 0.374029, F1 Score: 0.4379

src.models.train_eval - INFO - Evaluating Epoch 6 took 1.1 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2519225, 1.0: 744549}

src.models.train_eval - INFO - Epoch 6 Metrics --
Train Loss: 0.3677,
Train F1 Score: 0.0052,
Train Acc: 77.1%,
Test Loss: 0.3740,
Test Acc: 0.7724,
Test F1: 0.4379,
Test PR AUC: 0.0126

Epochs:  60%|██████    | 6/10 [35:11<22:41, 340.38s/it]src.models.train_eval - INFO - Number of batches: 1594

src.models.train_eval - INFO - Batch size: 4096

Epoch 7/10
-------------------------------

Training the model...
Training...:   0%|          | 2/1594 [00:07<1:27:24,  3.29s/it]Loss: 0.489893  [ 4096/6527546]
Training...:  10%|█         | 161/1594 [00:28<03:19,  7.17it/s]Loss: 0.531853  [655360/6527546]
Training...:  20%|██        | 320/1594 [00:50<02:40,  7.96it/s]Loss: 0.476242  [1306624/6527546]
Training...:  30%|███       | 479/1594 [01:13<02:33,  7.27it/s]Loss: 0.313902  [1957888/6527546]
Training...:  40%|████      | 638/1594 [01:39<02:27,  6.48it/s]Loss: 0.313890  [2609152/6527546]
Training...:  50%|█████     | 797/1594 [02:07<02:06,  6.32it/s]Loss: 0.313878  [3260416/6527546]
Training...:  60%|█████▉    | 956/1594 [02:32<01:38,  6.46it/s]Loss: 0.313868  [3911680/6527546]
Training...:  70%|██████▉   | 1115/1594 [02:56<01:11,  6.67it/s]Loss: 0.313857  [4562944/6527546]
Training...:  80%|███████▉  | 1274/1594 [03:21<00:45,  7.00it/s]Loss: 0.313847  [5214208/6527546]
Training...:  90%|████████▉ | 1433/1594 [03:43<00:22,  7.19it/s]Loss: 0.313837  [5865472/6527546]
Training...: 100%|█████████▉| 1592/1594 [04:06<00:00,  7.25it/s]Loss: 0.313827  [6516736/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 7:
 Sum of squared gradients :    1249.0923
 Sum of squared parameters:   44520.3879

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 79.4%,Avg loss: 0.366883, F1 Score: 0.0057

src.models.train_eval - INFO - Training Epoch 7 took 4.4 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:
 Accuracy: 78.0%,Avg loss: 0.373860, F1 Score: 0.4405

src.models.train_eval - INFO - Evaluating Epoch 7 took 1.0 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2544947, 1.0: 718827}

src.models.train_eval - INFO - Epoch 7 Metrics --
Train Loss: 0.3669,
Train F1 Score: 0.0057,
Train Acc: 79.4%,
Test Loss: 0.3739,
Test Acc: 0.7803,
Test F1: 0.4405,
Test PR AUC: 0.0079

Epochs:  70%|███████   | 7/10 [40:35<16:45, 335.09s/it]src.models.train_eval - INFO - Number of batches: 1594

src.models.train_eval - INFO - Batch size: 4096

Epoch 8/10
-------------------------------

Training the model...
Training...:   0%|          | 2/1594 [00:07<1:24:32,  3.19s/it]Loss: 0.489494  [ 4096/6527546]
Training...:  10%|█         | 161/1594 [00:27<03:01,  7.92it/s]Loss: 0.527061  [655360/6527546]
Training...:  20%|██        | 320/1594 [00:48<03:01,  7.02it/s]Loss: 0.483453  [1306624/6527546]
Training...:  30%|███       | 479/1594 [01:10<02:41,  6.90it/s]Loss: 0.313775  [1957888/6527546]
Training...:  40%|████      | 638/1594 [01:34<02:22,  6.70it/s]Loss: 0.313767  [2609152/6527546]
Training...:  50%|█████     | 797/1594 [01:58<02:02,  6.51it/s]Loss: 0.313760  [3260416/6527546]
Training...:  60%|█████▉    | 956/1594 [02:22<01:36,  6.61it/s]Loss: 0.313753  [3911680/6527546]
Training...:  70%|██████▉   | 1115/1594 [02:46<01:11,  6.74it/s]Loss: 0.313746  [4562944/6527546]
Training...:  80%|███████▉  | 1274/1594 [03:09<00:41,  7.63it/s]Loss: 0.313739  [5214208/6527546]
Training...:  90%|████████▉ | 1433/1594 [03:31<00:24,  6.63it/s]Loss: 0.313732  [5865472/6527546]
Training...: 100%|█████████▉| 1592/1594 [03:52<00:00,  6.64it/s]Loss: 0.313726  [6516736/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 8:
 Sum of squared gradients :    1025.3968
 Sum of squared parameters:   45160.2247

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 79.3%,Avg loss: 0.366154, F1 Score: 0.0056

src.models.train_eval - INFO - Training Epoch 8 took 4.1 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:
 Accuracy: 90.7%,Avg loss: 0.369892, F1 Score: 0.4799

src.models.train_eval - INFO - Evaluating Epoch 8 took 1.0 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2959133, 1.0: 304641}

src.models.train_eval - INFO - Epoch 8 Metrics --
Train Loss: 0.3662,
Train F1 Score: 0.0056,
Train Acc: 79.3%,
Test Loss: 0.3699,
Test Acc: 0.9070,
Test F1: 0.4799,
Test PR AUC: 0.0135

Epochs:  80%|████████  | 8/10 [45:44<10:53, 326.72s/it]src.models.train_eval - INFO - Number of batches: 1594

src.models.train_eval - INFO - Batch size: 4096

Epoch 9/10
-------------------------------

Training the model...
Training...:   0%|          | 2/1594 [00:07<1:23:29,  3.15s/it]Loss: 0.508861  [ 4096/6527546]
Training...:  10%|█         | 161/1594 [00:27<03:05,  7.73it/s]Loss: 0.523155  [655360/6527546]
Training...:  20%|██        | 320/1594 [00:47<02:57,  7.19it/s]Loss: 0.481906  [1306624/6527546]
Training...:  30%|███       | 479/1594 [01:08<02:33,  7.26it/s]Loss: 0.313679  [1957888/6527546]
Training...:  40%|████      | 638/1594 [01:30<02:04,  7.67it/s]Loss: 0.313674  [2609152/6527546]
Training...:  50%|████▉     | 796/1594 [02:20<05:22,  2.48it/s]Loss: 0.313669  [3260416/6527546]
Training...:  60%|█████▉    | 955/1594 [03:51<04:31,  2.36it/s]Loss: 0.313665  [3911680/6527546]
Training...:  70%|██████▉   | 1115/1594 [04:25<01:15,  6.38it/s]Loss: 0.313660  [4562944/6527546]
Training...:  80%|███████▉  | 1274/1594 [04:50<00:44,  7.12it/s]Loss: 0.313655  [5214208/6527546]
Training...:  90%|████████▉ | 1433/1594 [05:13<00:22,  7.15it/s]Loss: 0.313651  [5865472/6527546]
Training...: 100%|█████████▉| 1592/1594 [05:36<00:00,  7.01it/s]Loss: 0.313647  [6516736/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 9:
 Sum of squared gradients :    1212.2857
 Sum of squared parameters:   45770.8101

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 80.5%,Avg loss: 0.365473, F1 Score: 0.0059

src.models.train_eval - INFO - Training Epoch 9 took 5.9 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:
 Accuracy: 85.0%,Avg loss: 0.370008, F1 Score: 0.4625

src.models.train_eval - INFO - Evaluating Epoch 9 took 1.0 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2772543, 1.0: 491231}

src.models.train_eval - INFO - Epoch 9 Metrics --
Train Loss: 0.3655,
Train F1 Score: 0.0059,
Train Acc: 80.5%,
Test Loss: 0.3700,
Test Acc: 0.8499,
Test F1: 0.4625,
Test PR AUC: 0.0109

Epochs:  90%|█████████ | 9/10 [52:40<05:54, 354.55s/it]src.models.train_eval - INFO - Number of batches: 1594

src.models.train_eval - INFO - Batch size: 4096

Epoch 10/10
-------------------------------

Training the model...
Training...:   0%|          | 2/1594 [00:07<1:21:05,  3.06s/it]Loss: 0.493212  [ 4096/6527546]
Training...:  10%|█         | 161/1594 [00:27<02:54,  8.20it/s]Loss: 0.518487  [655360/6527546]
Training...:  20%|██        | 320/1594 [00:47<02:47,  7.62it/s]Loss: 0.479951  [1306624/6527546]
Training...:  30%|███       | 479/1594 [01:09<02:50,  6.56it/s]Loss: 0.313614  [1957888/6527546]
Training...:  40%|████      | 638/1594 [01:32<02:34,  6.20it/s]Loss: 0.313611  [2609152/6527546]
Training...:  50%|█████     | 797/1594 [01:56<02:01,  6.54it/s]Loss: 0.313607  [3260416/6527546]
Training...:  60%|█████▉    | 956/1594 [02:21<01:29,  7.13it/s]Loss: 0.313604  [3911680/6527546]
Training...:  70%|██████▉   | 1115/1594 [02:45<01:18,  6.07it/s]Loss: 0.313600  [4562944/6527546]
Training...:  80%|███████▉  | 1274/1594 [03:07<00:43,  7.40it/s]Loss: 0.313597  [5214208/6527546]
Training...:  90%|████████▉ | 1433/1594 [03:29<00:22,  7.14it/s]Loss: 0.313594  [5865472/6527546]
Training...: 100%|█████████▉| 1592/1594 [03:52<00:00,  6.35it/s]Loss: 0.313591  [6516736/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 10:
 Sum of squared gradients :    1000.4693
 Sum of squared parameters:   46294.3588

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 80.3%,Avg loss: 0.365079, F1 Score: 0.0059

src.models.train_eval - INFO - Training Epoch 10 took 4.1 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:
 Accuracy: 89.0%,Avg loss: 0.370985, F1 Score: 0.4747

src.models.train_eval - INFO - Evaluating Epoch 10 took 1.0 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2903061, 1.0: 360713}

src.models.train_eval - INFO - Epoch 10 Metrics --
Train Loss: 0.3651,
Train F1 Score: 0.0059,
Train Acc: 80.3%,
Test Loss: 0.3710,
Test Acc: 0.8898,
Test F1: 0.4747,
Test PR AUC: 0.0114

Epochs: 100%|██████████| 10/10 [57:48<00:00, 346.87s/it]
src.models.train_eval - INFO - The entire train+eval code took 57.8
minutes to run.
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- *************************************************************** -->
<!--                         CPU #2 TRAINING                         -->
<!-- *************************************************************** -->
Training RNN Model (cpu)
==============================

cpu
src.models.train_eval - INFO - Number of batches: 12750

src.models.train_eval - INFO - Batch size: 512

Epoch 1/10
-------------------------------

Training the model...
Training...:   0%|          | 0/12750 [00:00<?, ?it/s]
Loss: 0.865520  [  512/6527546]
Loss: 0.660187  [653312/6527546]
Loss: 0.585828  [1306112/6527546]
Loss: 0.322028  [1958912/6527546]
Loss: 0.316312  [2611712/6527546]
Loss: 0.315161  [3264512/6527546]
Loss: 0.314652  [3917312/6527546]
Loss: 0.314361  [4570112/6527546]
Loss: 0.314172  [5222912/6527546]
Loss: 0.314039  [5875712/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 1:
 Sum of squared gradients :   18013.8540
 Sum of squared parameters:  343496.8286

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 86.0%,Avg loss: 0.400990, F1 Score: 0.0067

src.models.train_eval - INFO - Training Epoch 1 took 3.4 minutes.

Evaluating the model...
Testing...:   0%|          | 0/6375 [00:00<?, ?it/s]
src.models.train_eval - INFO - Test Performance:
 Accuracy: 96.1%,Avg loss: 0.396932, F1 Score: 0.4969

src.models.train_eval - INFO - Evaluating Epoch 1 took 0.9 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 3135985, 1.0: 127789}

src.models.train_eval - INFO - Epoch 1 Metrics --
Train Loss: 0.4010,
Train F1 Score: 0.0067,
Train Acc: 86.0%,
Test Loss: 0.3969,
Test Acc: 0.9609,
Test F1: 0.4969,
Test PR AUC: 0.0064

src.models.train_eval - INFO - Number of batches: 12750

src.models.train_eval - INFO - Batch size: 512

Epoch 2/10
-------------------------------

Training the model...
Training...:   0%|          | 0/12750 [00:00<?, ?it/s]
Loss: 0.502255  [  512/6527546]
Loss: 0.802605  [653312/6527546]
Loss: 0.583640  [1306112/6527546]
Loss: 0.313821  [1958912/6527546]
Loss: 0.313749  [2611712/6527546]
Loss: 0.313695  [3264512/6527546]
Loss: 0.313652  [3917312/6527546]
Loss: 0.313617  [4570112/6527546]
Loss: 0.313588  [5222912/6527546]
Loss: 0.313564  [5875712/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 2:
 Sum of squared gradients :   18647.8676
 Sum of squared parameters:  413516.7084

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 84.3%,Avg loss: 0.398606, F1 Score: 0.0061

src.models.train_eval - INFO - Training Epoch 2 took 2.3 minutes.

Evaluating the model...
Testing...:   0%|          | 0/6375 [00:00<?, ?it/s]
src.models.train_eval - INFO - Test Performance:
 Accuracy: 75.5%,Avg loss: 0.409932, F1 Score: 0.4322

src.models.train_eval - INFO - Evaluating Epoch 2 took 0.8 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2462629, 1.0: 801145}

src.models.train_eval - INFO - Epoch 2 Metrics --
Train Loss: 0.3986,
Train F1 Score: 0.0061,
Train Acc: 84.3%,
Test Loss: 0.4099,
Test Acc: 0.7550,
Test F1: 0.4322,
Test PR AUC: 0.0048

src.models.train_eval - INFO - Number of batches: 12750

src.models.train_eval - INFO - Batch size: 512

Epoch 3/10
-------------------------------

Training the model...
Training...:   0%|          | 0/12750 [00:00<?, ?it/s]
Loss: 0.756901  [  512/6527546]
Loss: 0.807625  [653312/6527546]
Loss: 0.737066  [1306112/6527546]
Loss: 0.313426  [1958912/6527546]
Loss: 0.313419  [2611712/6527546]
Loss: 0.313413  [3264512/6527546]
Loss: 0.313408  [3917312/6527546]
Loss: 0.313403  [4570112/6527546]
Loss: 0.313398  [5222912/6527546]
Loss: 0.313394  [5875712/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 3:
 Sum of squared gradients :   17431.0829
 Sum of squared parameters:  460813.0396

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 87.9%,Avg loss: 0.395361, F1 Score: 0.0074

src.models.train_eval - INFO - Training Epoch 3 took 2.2 minutes.

Evaluating the model...
Testing...:   0%|          | 0/6375 [00:00<?, ?it/s]
src.models.train_eval - INFO - Test Performance:
 Accuracy: 92.0%,Avg loss: 0.384122, F1 Score: 0.4841

src.models.train_eval - INFO - Evaluating Epoch 3 took 0.8 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 3002736, 1.0: 261038}

src.models.train_eval - INFO - Epoch 3 Metrics --
Train Loss: 0.3954,
Train F1 Score: 0.0074,
Train Acc: 87.9%,
Test Loss: 0.3841,
Test Acc: 0.9203,
Test F1: 0.4841,
Test PR AUC: 0.0065

src.models.train_eval - INFO - Number of batches: 12750

src.models.train_eval - INFO - Batch size: 512

Epoch 4/10
-------------------------------

Training the model...
Training...:   0%|          | 0/12750 [00:00<?, ?it/s]
Loss: 0.553824  [  512/6527546]
Loss: 0.625850  [653312/6527546]
Loss: 0.574247  [1306112/6527546]
Loss: 0.313390  [1958912/6527546]
Loss: 0.313386  [2611712/6527546]
Loss: 0.313382  [3264512/6527546]
Loss: 0.313378  [3917312/6527546]
Loss: 0.313375  [4570112/6527546]
Loss: 0.313372  [5222912/6527546]
Loss: 0.313368  [5875712/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 4:
 Sum of squared gradients :   18497.7284
 Sum of squared parameters:  505432.6301

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 84.7%,Avg loss: 0.395758, F1 Score: 0.0066

src.models.train_eval - INFO - Training Epoch 4 took 2.2 minutes.

Evaluating the model...
Testing...:   0%|          | 0/6375 [00:00<?, ?it/s]
src.models.train_eval - INFO - Test Performance:
 Accuracy: 75.8%,Avg loss: 0.404030, F1 Score: 0.4331

src.models.train_eval - INFO - Evaluating Epoch 4 took 1.2 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2471535, 1.0: 792239}

src.models.train_eval - INFO - Epoch 4 Metrics --
Train Loss: 0.3958,
Train F1 Score: 0.0066,
Train Acc: 84.7%,
Test Loss: 0.4040,
Test Acc: 0.7578,
Test F1: 0.4331,
Test PR AUC: 0.0074

src.models.train_eval - INFO - Number of batches: 12750

src.models.train_eval - INFO - Batch size: 512

Epoch 5/10
-------------------------------

Training the model...
Training...:   0%|          | 0/12750 [00:00<?, ?it/s]
Loss: 0.685638  [  512/6527546]
Loss: 0.859011  [653312/6527546]
Loss: 0.588841  [1306112/6527546]
Loss: 0.313356  [1958912/6527546]
Loss: 0.313354  [2611712/6527546]
Loss: 0.313352  [3264512/6527546]
Loss: 0.313350  [3917312/6527546]
Loss: 0.313348  [4570112/6527546]
Loss: 0.313346  [5222912/6527546]
Loss: 0.313344  [5875712/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 5:
 Sum of squared gradients :   47648.6705
 Sum of squared parameters:  570780.2515

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 83.8%,Avg loss: 0.399412, F1 Score: 0.0060

src.models.train_eval - INFO - Training Epoch 5 took 9.8 minutes.

Evaluating the model...
Testing...:   0%|          | 0/6375 [00:00<?, ?it/s]
src.models.train_eval - INFO - Test Performance:
 Accuracy: 75.1%,Avg loss: 0.413672, F1 Score: 0.4310

src.models.train_eval - INFO - Evaluating Epoch 5 took 0.7 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2450478, 1.0: 813296}

src.models.train_eval - INFO - Epoch 5 Metrics --
Train Loss: 0.3994,
Train F1 Score: 0.0060,
Train Acc: 83.8%,
Test Loss: 0.4137,
Test Acc: 0.7513,
Test F1: 0.4310,
Test PR AUC: 0.0022

src.models.train_eval - INFO - Number of batches: 12750

src.models.train_eval - INFO - Batch size: 512

Epoch 6/10
-------------------------------

Training the model...
Training...:   0%|          | 0/12750 [00:00<?, ?it/s]
Loss: 0.873056  [  512/6527546]
Loss: 0.722624  [653312/6527546]
Loss: 0.705623  [1306112/6527546]
Loss: 0.313364  [1958912/6527546]
Loss: 0.313361  [2611712/6527546]
Loss: 0.313358  [3264512/6527546]
Loss: 0.313355  [3917312/6527546]
Loss: 0.313352  [4570112/6527546]
Loss: 0.313349  [5222912/6527546]
Loss: 0.313347  [5875712/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 6:
 Sum of squared gradients :   12758.2943
 Sum of squared parameters:  604194.6611

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 86.3%,Avg loss: 0.399116, F1 Score: 0.0065

src.models.train_eval - INFO - Training Epoch 6 took 1.6 minutes.

Evaluating the model...
Testing...:   0%|          | 0/6375 [00:00<?, ?it/s]
src.models.train_eval - INFO - Test Performance:
 Accuracy: 75.1%,Avg loss: 0.414187, F1 Score: 0.4310

src.models.train_eval - INFO - Evaluating Epoch 6 took 0.6 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2450916, 1.0: 812858}

src.models.train_eval - INFO - Epoch 6 Metrics --
Train Loss: 0.3991,
Train F1 Score: 0.0065,
Train Acc: 86.3%,
Test Loss: 0.4142,
Test Acc: 0.7514,
Test F1: 0.4310,
Test PR AUC: 0.0017

src.models.train_eval - INFO - Number of batches: 12750

src.models.train_eval - INFO - Batch size: 512

Epoch 7/10
-------------------------------

Training the model...
Training...:   0%|          | 0/12750 [00:00<?, ?it/s]
Loss: 0.889815  [  512/6527546]
Loss: 0.725035  [653312/6527546]
Loss: 0.589062  [1306112/6527546]
Loss: 0.313347  [1958912/6527546]
Loss: 0.313344  [2611712/6527546]
Loss: 0.313342  [3264512/6527546]
Loss: 0.313340  [3917312/6527546]
Loss: 0.313338  [4570112/6527546]
Loss: 0.313336  [5222912/6527546]
Loss: 0.313334  [5875712/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 7:
 Sum of squared gradients :    4566.8771
 Sum of squared parameters:  614656.8802

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 80.8%,Avg loss: 0.406767, F1 Score: 0.0049

src.models.train_eval - INFO - Training Epoch 7 took 1.5 minutes.

Evaluating the model...
Testing...:   0%|          | 0/6375 [00:00<?, ?it/s]
src.models.train_eval - INFO - Test Performance:
 Accuracy: 75.8%,Avg loss: 0.409293, F1 Score: 0.4331

src.models.train_eval - INFO - Evaluating Epoch 7 took 0.6 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2471640, 1.0: 792134}

src.models.train_eval - INFO - Epoch 7 Metrics --
Train Loss: 0.4068,
Train F1 Score: 0.0049,
Train Acc: 80.8%,
Test Loss: 0.4093,
Test Acc: 0.7578,
Test F1: 0.4331,
Test PR AUC: 0.0024

src.models.train_eval - INFO - Number of batches: 12750

src.models.train_eval - INFO - Batch size: 512

Epoch 8/10
-------------------------------

Training the model...
Training...:   0%|          | 0/12750 [00:00<?, ?it/s]
Loss: 0.763677  [  512/6527546]
Loss: 0.634148  [653312/6527546]
Loss: 0.502825  [1306112/6527546]
Loss: 0.313310  [1958912/6527546]
Loss: 0.313309  [2611712/6527546]
Loss: 0.313308  [3264512/6527546]
Loss: 0.313307  [3917312/6527546]
Loss: 0.313306  [4570112/6527546]
Loss: 0.313306  [5222912/6527546]
Loss: 0.313305  [5875712/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 8:
 Sum of squared gradients :   13664.7350
 Sum of squared parameters:  639716.7964

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 86.5%,Avg loss: 0.399731, F1 Score: 0.0066

src.models.train_eval - INFO - Training Epoch 8 took 1.6 minutes.

Evaluating the model...
Testing...:   0%|          | 0/6375 [00:00<?, ?it/s]
src.models.train_eval - INFO - Test Performance:
 Accuracy: 87.4%,Avg loss: 0.397094, F1 Score: 0.4697

src.models.train_eval - INFO - Evaluating Epoch 8 took 0.6 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2851401, 1.0: 412373}

src.models.train_eval - INFO - Epoch 8 Metrics --
Train Loss: 0.3997,
Train F1 Score: 0.0066,
Train Acc: 86.5%,
Test Loss: 0.3971,
Test Acc: 0.8740,
Test F1: 0.4697,
Test PR AUC: 0.0028

src.models.train_eval - INFO - Number of batches: 12750

src.models.train_eval - INFO - Batch size: 512

Epoch 9/10
-------------------------------

Training the model...
Training...:   0%|          | 0/12750 [00:00<?, ?it/s]
Loss: 0.678922  [  512/6527546]
Loss: 0.859196  [653312/6527546]
Loss: 0.563462  [1306112/6527546]
Loss: 0.313292  [1958912/6527546]
Loss: 0.313291  [2611712/6527546]
Loss: 0.313291  [3264512/6527546]
Loss: 0.313291  [3917312/6527546]
Loss: 0.313291  [4570112/6527546]
Loss: 0.313290  [5222912/6527546]
Loss: 0.313290  [5875712/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 9:
 Sum of squared gradients :   25425.7985
 Sum of squared parameters:  690962.4397

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 89.2%,Avg loss: 0.390401, F1 Score: 0.0084

src.models.train_eval - INFO - Training Epoch 9 took 1.5 minutes.

Evaluating the model...
Testing...:   0%|          | 0/6375 [00:00<?, ?it/s]
src.models.train_eval - INFO - Test Performance:
 Accuracy: 75.3%,Avg loss: 0.410184, F1 Score: 0.4315

src.models.train_eval - INFO - Evaluating Epoch 9 took 0.6 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2456549, 1.0: 807225}

src.models.train_eval - INFO - Epoch 9 Metrics --
Train Loss: 0.3904,
Train F1 Score: 0.0084,
Train Acc: 89.2%,
Test Loss: 0.4102,
Test Acc: 0.7532,
Test F1: 0.4315,
Test PR AUC: 0.0023

src.models.train_eval - INFO - Number of batches: 12750

src.models.train_eval - INFO - Batch size: 512

Epoch 10/10
-------------------------------

Training the model...
Training...:   0%|          | 0/12750 [00:00<?, ?it/s]
Loss: 0.782740  [  512/6527546]
Loss: 0.713513  [653312/6527546]
Loss: 0.604717  [1306112/6527546]
Loss: 0.313279  [1958912/6527546]
Loss: 0.313279  [2611712/6527546]
Loss: 0.313279  [3264512/6527546]
Loss: 0.313279  [3917312/6527546]
Loss: 0.313279  [4570112/6527546]
Loss: 0.313278  [5222912/6527546]
Loss: 0.313278  [5875712/6527546]
src.models.train_eval - INFO -
Sum squared grads/params in Epoch 10:
 Sum of squared gradients :   12293.8694
 Sum of squared parameters:  716464.3804

src.models.train_eval - INFO -
Train Performance:
 Accuracy: 84.0%,Avg loss: 0.400726, F1 Score: 0.0059

src.models.train_eval - INFO - Training Epoch 10 took 1.6 minutes.

Evaluating the model...
Testing...:   0%|          | 0/6375 [00:00<?, ?it/s]
src.models.train_eval - INFO - Test Performance:
 Accuracy: 94.4%,Avg loss: 0.391815, F1 Score: 0.4918

src.models.train_eval - INFO - Evaluating Epoch 10 took 0.6 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 3080572, 1.0: 183202}

src.models.train_eval - INFO - Epoch 10 Metrics --
Train Loss: 0.4007,
Train F1 Score: 0.0059,
Train Acc: 84.0%,
Test Loss: 0.3918,
Test Acc: 0.9441,
Test F1: 0.4918,
Test PR AUC: 0.0055

src.models.train_eval - INFO - The entire train+eval code took 34.9 minutes to run.
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- *************************************************************** -->
<!--                         MPS #2 TRAINING                         -->
<!-- *************************************************************** -->
