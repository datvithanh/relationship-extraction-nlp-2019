import numpy as np
from dataset import LoadDataset
import torch
from utils import load_data_and_labels
import pickle
import IPython.display as ipd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

embedding = torch.load('train_embedding.ts')
tokens, labels, e1, e2, pos1, pos2 = load_data_and_labels('data/SemEval2010_task8_training/TRAIN_FILE.TXT')
embedding_np = np.array(embedding)

total = list(zip(embedding_np, tokens, labels, e1, e2, pos1, pos2))
np.random.shuffle(total)

valid_size = len(total)//10
total_train = total[valid_size:]
total_valid = total[:valid_size]


train_loader = LoadDataset('train', total_train)
valid_loader = LoadDataset('valid', total_valid)


exp = 'exp-3-adagrad-l2reg'

learning_rate = 1e-3
num_epochs = 60


lamb = 1e-5
model = Attention_bilstm_let(1024, 300, 50, 50, 19)
# optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)
optimizer = torch.optim.Adagrad(model.parameters(), learning_rate)

best_acc = 0
valid_loss = []
valid_acc = []
train_loss = []
train_acc = []

loss_fn = torch.nn.CrossEntropyLoss(reduce='mean')

dataloaders = {'train': train_loader, 'valid': valid_loader}
os.makedirs(os.path.join('checkpoint', exp), exist_ok=True)
for epoch in range(num_epochs):
  print(f"epoch: {epoch}")
  for phase in ['train', 'valid']:
    dataloader = dataloaders[phase]
    print(phase.upper())
    num_steps = len(dataloader)
    labels = []
    preds = []
    all_loss = []

    tdata = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (token, label, e1, e2, p1, p2) in tdata:
      output = model(token, e1, e2, p1, p2)
      loss = loss_fn(output, label)
      pred = torch.max(output, 1)[1]

      tdata.set_postfix({
          'loss': loss.item()
      })
      
      labels = labels + label.tolist()
      preds = preds + pred.tolist()
      all_loss.append(loss.tolist())

      l2_reg = torch.tensor(0.)
      for param in model.parameters():
        l2_reg += torch.norm(param)
      loss += lamb * l2_reg

      if phase == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      else:
        if np.random.randint(1,100) <= 35 + epoch * 1.5:
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

    acc = sum([tmp1 == tmp2 for tmp1, tmp2 in zip(preds, labels)])/len(preds)
    if phase == 'train':
      train_loss.append(np.mean(all_loss))
      train_acc.append(acc)
    else:
      valid_loss.append(np.mean(all_loss))
      valid_acc.append(acc)
    print(f'accuracy: {acc}, loss: {np.mean(all_loss)}')
    if phase == 'valid' and best_acc < acc:
      acc = best_acc
      # torch.save(M, save_path)
      torch.save(model.state_dict(), f'checkpoint/{exp}/model-epoch-{epoch}.t')

di = {
  'valid_loss': valid_loss,
  'valid_acc': valid_acc,
  'train_loss': train_loss, 
  'train_acc': train_acc}

with open(f'checkpoint/{exp}.pickle', 'wb') as f:
    pickle.dump(di, f)