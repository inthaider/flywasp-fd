# 2023-11-15

## MPS runs

### 1
============================================================
Training RNN Model...
===============================

Epochs:   0%|          | 0/1 [00:00<?, ?it/s]Epoch 1/1
-------------------------------
Training the model...
Number of batches: 12750
Batch size: 512

Training...:   0%|          | 2/12750 [00:01<2:12:12,  1.61it/s]Loss: 0.837322  [  512/6527546]
Training...:  10%|█         | 1277/12750 [01:48<18:23, 10.40it/s]Loss: 0.783530  [653312/6527546]
Training...:  20%|██        | 2552/12750 [04:04<15:08, 11.23it/s]Loss: 0.628416  [1306112/6527546]
Training...:  30%|███       | 3828/12750 [05:57<12:02, 12.35it/s]Loss: 0.325131  [1958912/6527546]
Training...:  40%|████      | 5102/12750 [07:45<10:23, 12.26it/s]Loss: 0.316819  [2611712/6527546]
Training...:  50%|█████     | 6378/12750 [09:29<08:29, 12.49it/s]Loss: 0.315333  [3264512/6527546]
Training...:  60%|██████    | 7652/12750 [11:10<06:34, 12.92it/s]Loss: 0.314719  [3917312/6527546]
Training...:  70%|███████   | 8928/12750 [12:54<04:59, 12.76it/s]Loss: 0.314384  [4570112/6527546]
Training...:  80%|████████  | 10202/12750 [14:36<03:15, 13.02it/s]Loss: 0.314174  [5222912/6527546]
Training...:  90%|█████████ | 11478/12750 [16:20<01:44, 12.20it/s]Loss: 0.314030  [5875712/6527546]
src.models.train_eval - INFO -                                    
Sum squared grads/params in Epoch 1:
	Sum of squared gradients :   17456.5570
	Sum of squared parameters:  345283.7079

src.models.train_eval - INFO - Training Epoch 1 took 1285.0741701126099 seconds.




Train Performance: 
 Avg loss: 0.401797, F1 Score: 0.0064 

Evaluating the model...
src.models.train_eval - INFO - Evaluating Epoch 1 took 407.5231990814209 seconds.



Epochs: 100%|██████████| 1/1 [28:12<00:00, 1692.65s/it]
src.models.train_eval - INFO - The entire train+eval code took 1692.7201869487762 seconds to run.



Test Performance: 
 Accuracy: 75.0%, Avg loss: 0.399750, F1 Score: 0.4306 

Training Data Distribution:
{0.0: 3262149, 1.0: 1625}
Predicted Data Distribution:
{0.0: 2447275, 1.0: 816499}

 Epoch 1 Metrics -- Train Loss: 0.4018, Train F1 Score: 0.0064, Test Loss: 0.3998, Test Acc: 0.7503, Test F1: 0.4306, Test PR AUC: 0.0067
============================================================




<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
### 2
============================================================
<!-- 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
batch_size = (512 * 8) if device == torch.device("mps") else 512
num_epochs = 1
hidden_size = 32
output_size = 2
learning_rate = 0.01
batch_first = True
input_size = X_train.shape[2]
kwargs = {"num_workers": 2, "pin_memory": True}  # For GPU training 
-->
============================================================
Training RNN Model (mps)...
==============================



mps
Using kwargs for data loaders


Epochs:   0%|          | 0/1 [00:00<?, ?it/s]src.models.train_eval - INFO - Number of batches: 1594

src.models.train_eval - INFO - Batch size: 4096


Epoch 1/1
-------------------------------
Training the model...
Training...:   0%|          | 2/1594 [00:05<56:12,  2.12s/it]  
Loss: 0.631040  [ 4096/6527546]
Training...:  10%|█         | 162/1594 [00:15<01:29, 15.97it/s]
Loss: 0.524200  [655360/6527546]
Training...:  20%|██        | 320/1594 [00:25<01:22, 15.45it/s]
Loss: 0.506328  [1306624/6527546]
Training...:  30%|███       | 480/1594 [00:36<01:15, 14.83it/s]
Loss: 0.446141  [1957888/6527546]
Training...:  40%|████      | 638/1594 [00:47<01:14, 12.91it/s]
Loss: 0.349506  [2609152/6527546]
Training...:  50%|█████     | 798/1594 [01:00<01:10, 11.31it/s]
Loss: 0.332805  [3260416/6527546]
Training...:  60%|█████▉    | 956/1594 [01:14<00:57, 11.13it/s]
Loss: 0.326536  [3911680/6527546]
Training...:  70%|███████   | 1116/1594 [01:28<00:42, 11.24it/s]
Loss: 0.323296  [4562944/6527546]
Training...:  80%|███████▉  | 1274/1594 [01:42<00:26, 12.00it/s]
Loss: 0.321325  [5214208/6527546]
Training...:  90%|████████▉ | 1434/1594 [01:56<00:13, 11.68it/s]
Loss: 0.320000  [5865472/6527546]
Training...: 100%|█████████▉| 1592/1594 [02:10<00:00, 12.12it/s]
Loss: 0.319049  [6516736/6527546]
src.models.train_eval - INFO -                                  
Sum squared grads/params in Epoch 1:
	Sum of squared gradients :     376.0920
	Sum of squared parameters:   35672.5908

src.models.train_eval - INFO - 
Train Performance: 
 Avg loss: 0.400721, F1 Score: 0.0047 


src.models.train_eval - INFO - Training Epoch 1 took 2.4 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:             
 Accuracy: 75.0%, Avg loss: 0.382569, F1 Score: 0.4307 


src.models.train_eval - INFO - Evaluating Epoch 1 took 0.7 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 2447757, 1.0: 816017}

src.models.train_eval - INFO - Epoch 1 Metrics --
Train Loss: 0.4007,
Train F1 Score: 0.0047, Test Loss: 0.3826,
Test Acc: 0.7505, Test F1: 0.4307,
Test PR AUC: 0.0087



Epochs: 100%|██████████| 1/1 [03:06<00:00, 186.64s/it]
src.models.train_eval - INFO - The entire train+eval code took 3.1 minutes to run.
============================================================



<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->
<!-- --------------------------------------------------------------- -->

## CPU runs
### 1
============================================================
<!-- 
hidden_size = 32
output_size = 2
num_epochs = 1
batch_size = (512 * 4) if device == torch.device("mps") else 512
learning_rate = 0.01
batch_first = True
kwargs = {"num_workers": 2, "pin_memory": True}  # For GPU training 
-->
============================================================
Training RNN Model (cpu)...
==============================



cpu
Epochs:   0%|          | 0/1 [00:00<?, ?it/s]src.models.train_eval - INFO - Number of batches: 12750

src.models.train_eval - INFO - Batch size: 512


Epoch 1/1
-------------------------------
Training the model...
Training...:   0%|          | 2/12750 [00:00<19:35, 10.85it/s]
Loss: 0.698561  [  512/6527546]
Training...:  10%|█         | 1292/12750 [00:17<02:09, 88.62it/s]
Loss: 0.687180  [653312/6527546]
Training...:  20%|██        | 2563/12750 [00:31<01:46, 95.33it/s] 
Loss: 0.579098  [1306112/6527546]
Training...:  30%|███       | 3830/12750 [00:46<02:05, 71.25it/s] 
Loss: 0.322511  [1958912/6527546]
Training...:  40%|████      | 5107/12750 [01:07<02:13, 57.07it/s]
Loss: 0.316215  [2611712/6527546]
Training...:  50%|█████     | 6383/12750 [01:37<02:55, 36.37it/s]
Loss: 0.315057  [3264512/6527546]
Training...:  60%|██████    | 7655/12750 [03:02<02:22, 35.80it/s]
Loss: 0.314560  [3917312/6527546]
Training...:  70%|███████   | 8929/12750 [04:26<02:12, 28.80it/s]
Loss: 0.314281  [4570112/6527546]
Training...:  80%|████████  | 10206/12750 [06:12<01:28, 28.80it/s]
Loss: 0.314103  [5222912/6527546]
Training...:  90%|█████████ | 11484/12750 [07:20<00:16, 74.76it/s]
Loss: 0.313978  [5875712/6527546]
src.models.train_eval - INFO -                                    
Sum squared grads/params in Epoch 1:
	Sum of squared gradients :   21480.3940
	Sum of squared parameters:  366379.5568

src.models.train_eval - INFO - 
Train Performance: 
 Avg loss: 0.395037, F1 Score: 0.0074 


src.models.train_eval - INFO - Training Epoch 1 took 7.9 minutes.

Evaluating the model...
src.models.train_eval - INFO - Test Performance:                
 Accuracy: 99.3%, Avg loss: 0.397765, F1 Score: 0.5143 


src.models.train_eval - INFO - Evaluating Epoch 1 took 1.2 minutes.

src.models.train_eval - INFO - Training Data Distribution:

src.models.train_eval - INFO - {0.0: 3262149, 1.0: 1625}

src.models.train_eval - INFO - Predicted Data Distribution:

src.models.train_eval - INFO - {0.0: 3241464, 1.0: 22310}

src.models.train_eval - INFO - Epoch 1 Metrics -- Train Loss: 0.3950,
Train F1 Score: 0.0074, Test Loss: 0.3978,
Test Acc: 0.9929, Test F1: 0.5143,
Test PR AUC: 0.0100



Epochs: 100%|██████████| 1/1 [09:02<00:00, 542.45s/it]
src.models.train_eval - INFO - The entire train+eval code took 9.0
minutes to run.

### 2
