import numpy as np
import pandas as pd
import sys
sys.path.append('/home/a053h213/PycharmProjects/pythonProject/')

from Models.SimilarMetaVisitUserSequence import Transformer
import getuserembed as ds
import Uitls.evaluation as ev
import torch
from torch.utils.tensorboard import SummaryWriter
#from Uitls.CustomTrainerMeta import Trainer
from Uitls.CustomTrainerMeta import Trainer

from Uitls import Parser
timestep = 5
percentile_values = [14, 22, 28, 33, 38, 44, 50, 61, 82]
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
argprs = Parser.get_parser()
print(argprs)
torch.autograd.set_detect_anomaly(True)
seed = argprs.seed
print('SEED', seed)
dim = argprs.dim
title = argprs.title
dataset, train_loader, val_loader, test_loader = ds.get_loader_retain(value=None, timestep=timestep, cross=seed)

writer_normal = SummaryWriter(log_dir=f'Results_7/visit_user_{dim}_{seed}_{title}_normal')
writer_percentile = []
for i in range(0, len(percentiles)):
    writer_percentile.append(SummaryWriter(log_dir=f'Results_7/visit_user_{dim}_{seed}_{title}_{percentiles[i]}'))

print('inputs', argprs.visit, argprs.user)
model = Transformer(
    dataset=dataset,
    #attn_heads = 4,
    feature_keys=["conditions", "procedures", "drugs_hist"],
    label_key= "drugs",
    embedding_dim = dim,
    timestep=timestep,
    visit_adaptation=argprs.visit,
    user_adaptation=argprs.user,
    mode='multilabel'
)

print('amade train')

trainer = Trainer(model=model, device=f'cuda:2')
'''y_true, y_prob, loss, y_true_cold, y_prob_cold, n_codes = trainer.inference(test_loader)
ev.evaluate(y_prob, y_true, writer_normal, model, epoch=0)'''

output = trainer.train(
    train_dataloader=train_loader,
    val_dataloader=test_loader,
    epochs=20,
    writer_normal=writer_normal,
    writers=writer_percentile,
    monitor="pr_auc_samples",
    optimizer_params={"lr": 1e-3}
)
#y_true, y_prob, loss, y_true_cold, y_prob_cold, n_codes = trainer.inference(test_loader)

'''ev.evaluate(y_prob, y_true, writer_normal, model)

for i in range(0, len(percentiles)):
    #print(np.sum(n_codes < percentile_values[i]))
    ev.evaluate(y_prob[n_codes < percentile_values[i]], y_true[n_codes < percentile_values[i]], writer_percentile[i], model)
#ev.evaluate(y_prob_cold, y_true_cold, writer_cold, model)

writer_normal.close()
for i in range(0, len(percentiles)):
    writer_percentile[i].close()
#writer_cold.close()
'''

