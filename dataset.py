import binascii

import torch
import pytorch_lightning as pl


def fake_crc(val):
    # return 32bit crc checksum
    ret = binascii.crc32(val)
    return ret >> 31


class MyDataSet(torch.utils.data.Dataset):

    def __init__(self):
        self.batch_size = 512

        with open('./data/2022-01-19.bin', 'rb') as f:
            bits = f.read()

        self.data = []
        for i in range(0, len(bits), 32):
            # rv = os.urandom(32) # 伪随机
            rv = bits[i:i + 32]  # www.random.org 真随机
            v = int(binascii.hexlify(rv), 16)
            result = fake_crc(rv)
            self.data.append((rv, result))

    def __getitem__(self, index):
        rv, result = self.data[index]

        x = torch.tensor([b for b in rv], dtype=torch.float32)
        y = torch.tensor(result, dtype=torch.int64)

        return (x, y)

    def __len__(self):
        return len(self.data)


class MyDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=256):
        super().__init__()

        self.batch_size = batch_size

    def prepare_data(self):
        dataset = MyDataSet()

        # train:val:test = 6:2:2
        self.train = torch.utils.data.Subset(
            dataset, range(0, int(len(dataset)*0.6)))
        self.val = torch.utils.data.Subset(
            dataset, range(int(len(dataset)*0.6), int(len(dataset)*0.8)))
        self.test = torch.utils.data.Subset(
            dataset, range(int(len(dataset)*0.8), int(len(dataset))))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':
    dataset = MyDataSet()
