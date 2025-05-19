import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
                )
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class DenseNetSeg(nn.Module):
    def __init__(self):
        super(DenseNetSeg, self).__init__()
        self.init_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # Dense Blocks
        self.db1 = DenseBlock(32, growth_rate=16, num_layers=4)  # 32 + 4*16 = 96
        self.trans1 = nn.Conv2d(96, 64, kernel_size=1)
        self.pool1 = nn.MaxPool2d(2)

        self.db2 = DenseBlock(64, 16, 4)  # 64 + 64 = 128
        self.trans2 = nn.Conv2d(128, 64, 1)
        self.pool2 = nn.MaxPool2d(2)

        self.db3 = DenseBlock(64, 16, 4)
        self.trans3 = nn.Conv2d(128, 64, 1)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up1 = nn.Conv2d(64, 64, 3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up2 = nn.Conv2d(64, 32, 3, padding=1)

        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.init_conv(x)

        x = self.db1(x)
        x = self.trans1(x)
        x = self.pool1(x)

        x = self.db2(x)
        x = self.trans2(x)
        x = self.pool2(x)

        x = self.db3(x)
        x = self.trans3(x)

        x = self.up1(x)
        x = self.conv_up1(x)
        x = self.up2(x)
        x = self.conv_up2(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)



class MyDenseSeg:
    def __init__(self, batch_size=2, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" 当前使用设备: {self.device}")
        self.model = DenseNetSeg().to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def load_data(self, train_path="npydata/imgs_train.npy",
                  mask_path="npydata/imgs_mask_train.npy",
                  test_path="npydata/imgs_test.npy"):

        imgs_train = np.load(train_path).astype("float32")
        masks_train = np.load(mask_path).astype("float32")
        imgs_test = np.load(test_path).astype("float32")

        X = torch.tensor(imgs_train.transpose(0, 3, 1, 2))
        Y = torch.tensor(masks_train.transpose(0, 3, 1, 2))
        Y = (Y > 0.5).float()
        T = torch.tensor(imgs_test.transpose(0, 3, 1, 2))

        self.train_loader = DataLoader(TensorDataset(X, Y), batch_size=self.batch_size, shuffle=True)
        self.test_tensor = T.to(self.device)
        print(f" 已加载训练样本 {len(X)}，测试样本 {len(T)}")

    def train(self, epochs=5, save_path="my_densenet_seg.pth"):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            count = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = (outputs > 0.5).float()
                acc = (preds == labels).float().mean().item()
                total_acc += acc
                count += 1

            print(f" Epoch {epoch+1}/{epochs} - Loss: {total_loss/count:.4f} - Acc: {total_acc/count:.4f}")
        torch.save(self.model.state_dict(), save_path)
        print(f" 模型已保存至: {save_path}")

    def predict_and_save(self, output_dir="results_dense"):
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            preds = self.model(self.test_tensor)
            for i, pred in enumerate(preds):
                save_image(pred, os.path.join(output_dir, f"{i}.png"))
        print(f" 预测结果保存至: {output_dir}")

if __name__ == "__main__":
    model = MyDenseSeg()
    model.load_data(
        train_path="../npydata/imgs_train.npy",
        mask_path="../npydata/imgs_mask_train.npy",
        test_path="../npydata/imgs_test.npy"
    )
    model.train(epochs=10)
    model.predict_and_save("results_dense")