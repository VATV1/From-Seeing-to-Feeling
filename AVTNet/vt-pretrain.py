import torch
from torch.utils.data import random_split
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
import wandb
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import logging
from utils import accuracy, save_checkpoint
from model import MultiModalMoCo

config={
    # "train_dataset_name": 'calandra_label_train',
    "train_dataset_name": 'tag_train',
    # "data_folder": "objects_split_object_wise/",
    "data_folder": "/media/lab404/dataset2/AVTNet/DOVT",
    "model_name": "TAG",
    # "num_channels": 6, # should be 6 for calandra_label and 3 for tag
    "num_channels": 3, # should be 6 for calandra_label and 3 for tag
    "epochs": 240,
    "log_every_n_epochs": 10,
    "batch_size": 32,
    "num_workers": 16,
    "momentum": 0.99,
    "temperature": 0.07,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "nn_model": 'resnet18',
    "intra_dim": 128,
    "inter_dim": 128,
    "weight_inter_tv": 1,
    "weight_inter_vt": 1,
    "weight_intra_vision": 1,
    "weight_intra_tactile": 1,
    "pretrained_encoder": True,
    "use_wandb": False
}

dataset = ContrastiveLearningDataset(root_folder=config['data_folder'])
train_dataset = dataset.get_dataset(config['train_dataset_name'], 2)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                           num_workers=config['num_workers'], drop_last=False, pin_memory=True)

model = MultiModalMoCo(n_channels=config['num_channels'], m=config['momentum'], T=config['temperature'],
                       intra_dim=config['intra_dim'], inter_dim=config['inter_dim'], nn_model=config['nn_model'],
                       weight_inter_tv=config['weight_inter_tv'], weight_inter_vt=config['weight_inter_vt'],
                       weight_intra_vision=config['weight_intra_vision'], weight_intra_tactile=config['weight_intra_tactile'],
                       pretrained_encoder=config['pretrained_encoder'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training with gpu: {device}.")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
writer = SummaryWriter()
logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)
criterion = torch.nn.CrossEntropyLoss().to(device)
# Set number of training epochs
logging.info(f"Start MViTaC training for {config['epochs']} epochs.")
logging.info(f"Training with gpu: {device}.")
best_acc = 0
if config['use_wandb']:
    wandb.init(project="mvitac_pretraining", config=config)
    # name the model
    wandb.run.name = f"{config['nn_model']}_lr_{config['lr']}_batch_{config['batch_size']}_epochs_{config['epochs']}"

for epoch in range(config['epochs']):
    loss_epoch, vis_loss_intra_epoch, tac_loss_intra_epoch, vis_tac_inter_epoch, tac_vis_inter_epoch = 0, 0, 0, 0, 0
    pbar = tqdm(train_loader)  # Wrap train_loader with tqdm
    for idx, values in enumerate(pbar):  # Use enumerate to get idx
        x_vision_q, x_vision_k, x_tactile_q, x_tactile_k, label_p, label_s = values
        model.train()
        # send to device
        x_vision_q = x_vision_q.to(device, non_blocking=True)
        x_vision_k = x_vision_k.to(device, non_blocking=True)

        x_tactile_q = x_tactile_q.to(device, non_blocking=True)
        x_tactile_k = x_tactile_k.to(device, non_blocking=True)

        # Forward pass to get the loss
        # loss, vis_loss_intra, tac_loss_intra, vis_tac_inter, tac_vis_inter, logits, labels = model(x_vision_q,
        #                                                                                            x_vision_k,
        #                                                                                            x_tactile_q,
        #                                                                                            x_tactile_k)
        loss, vis_tac_inter, tac_vis_inter, logits, labels = model(x_vision_q,
                                                                                                   x_vision_k,
                                                                                                   x_tactile_q,
                                                                                                   x_tactile_k)
        loss_epoch += loss.item()
        # vis_loss_intra_epoch += vis_loss_intra.item()
        # tac_loss_intra_epoch += tac_loss_intra.item()
        vis_tac_inter_epoch += vis_tac_inter.item()
        tac_vis_inter_epoch += tac_vis_inter.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update pbar message
        pbar.set_description(f"Epoch {epoch}:\tTrain loss: {loss_epoch / (idx + 1):.2f}")

    if epoch % config['log_every_n_epochs'] == 0:
        top1, top5 = accuracy(logits, labels, topk=(1, 5))
        writer.add_scalar('loss', loss_epoch / len(train_loader), global_step=epoch)
        # writer.add_scalar('loss/vis_loss_intra', vis_loss_intra_epoch / len(train_loader), global_step=epoch)
        # writer.add_scalar('loss/tac_loss_intra', tac_loss_intra_epoch / len(train_loader), global_step=epoch)
        writer.add_scalar('loss/vis_tac_inter', vis_tac_inter_epoch / len(train_loader), global_step=epoch)
        writer.add_scalar('loss/tac_vis_inter', tac_vis_inter_epoch / len(train_loader), global_step=epoch)
        writer.add_scalar('acc/top1', top1[0], global_step=epoch)
        writer.add_scalar('acc/top5', top5[0], global_step=epoch)
        writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
        if top1[0] > best_acc:
            best_acc = top1[0]
            # save both the vision and tactile models

            save_checkpoint({
                'epoch': epoch,
                'arch': 'resnet18',
                'state_dict_vis': model.vision_base_q.state_dict(),
                'state_dict_tac': model.tactile_base_q.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(writer.log_dir, 'model_{}_best_{}.pth'.format(epoch, config["model_name"])))
        # torch.save(state, f'models/calandra/model_{args.task}_{epoch}_{args.batch_size}_best_object_wise_05_t05.pth')
        if config['use_wandb']:
            wandb.log({"epoch": epoch, "loss": loss_epoch / len(train_loader),
                       # "vis_loss_intra": vis_loss_intra_epoch / len(train_loader),
                       # "tac_loss_intra": tac_loss_intra_epoch / len(train_loader),
                       "vis_tac_inter": vis_tac_inter_epoch / len(train_loader),
                       "tac_vis_inter": tac_vis_inter_epoch / len(train_loader), "top1": top1[0], "top5": top5[0],
                       "learning_rate": scheduler.get_last_lr()[0]})
            wandb.save('models/{}/model_{}_best.pth'.format(config["model_name"], epoch))

    # warmup for the first 10 epochs
    if epoch >= 10:
        scheduler.step()
    logging.debug(f"Epoch: {epoch}\tLoss: {loss_epoch / len(train_loader)}\tTop1: {top1[0]}\tTop5: {top5[0]}")

    logging.info("Training has finished.")
    # save model checkpoints
    checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(config['epochs'])
    save_checkpoint({
        'epoch': config['epochs'],
        'arch': config['nn_model'],
        'state_dict_vis': model.vision_base_q.state_dict(),
        'state_dict_tac': model.tactile_base_q.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=os.path.join(writer.log_dir, checkpoint_name))
    logging.info(f"Model checkpoint and metadata has been saved at {writer.log_dir}.")






