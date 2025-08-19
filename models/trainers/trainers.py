from models.networks import *


class SynthTab(pl.LightningModule):
    def __init__(self, input_dim, output_dim, temperature=1.0, lr=1e-3):
        super().__init__()
        self.model = DeepMLP(input_dim, output_dim, temperature)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", val_loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for a batch of data.
        :param batch:
        A tuple containing the input data `x`, target labels `y`, and feedback `h`.
        :param batch_idx:
        Index of the batch in the current epoch.
        :param dataloader_idx:
        Index of the dataloader (default is 0).
        :return:
            A dictionary containing the predictions, probabilities, feedback, true labels, rejection score, and selection status.
        """
        x, y = batch  # assuming batch = (x, y, h)

        logits = self(x)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).view(-1,1)
        target = y
        return {
            "preds": preds,
            "probs": probs,
            "true": target,
            "logits": logits,
        }

    



  