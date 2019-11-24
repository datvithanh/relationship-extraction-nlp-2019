import numpy as np 
from dataset import LoadDataset
from model import Attention_bilstm_let
import torch

#training sida
learning_rate = 1e-3
num_epochs = 501
batch_size = 16

model = Attention_bilstm_let(1024, 300, 50, 50, 19)
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)
best_acc = 0
epoch_loss = []
epoch_acc = []

loss_fn = torch.nn.CrossEntropyLoss(reduce='mean')
train_set = LoadDataset('train', 'data/SemEval2010_task8_training/TRAIN_FILE.TXT')

for epoch in range(num_epochs):
    print(f"epoch: {epoch}")
    num_steps = len(train_set)
    labels = []
    preds = []
    all_loss = []
    step = 0
    for token, label, e1, e2, p1, p2 in train_set:
        if step % 50 == 0:
        print(f"{step}/{num_steps}")
        output = model(token, e1, e2, p1, p2)

        loss = loss_fn(output, label)

        pred = torch.max(output, 1)[1]
        
        labels = labels + label.tolist()
        preds = preds + pred.tolist()
        all_loss.append(loss.tolist())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

    acc = sum([tmp1 == tmp2 for tmp1, tmp2 in zip(preds, labels)])/len(preds)
    epoch_loss.append(np.mean(all_loss))
    epoch_acc.append(acc)
    print(preds)
    print(f'accuracy: {acc}, loss: {np.mean(all_loss)}')
    if best_acc < acc:
        acc = best_acc
        # torch.save(M, save_path)