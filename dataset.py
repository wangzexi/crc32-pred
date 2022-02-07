import os
import binascii

import torch
import pytorch_lightning as pl


def fake_crc(val):
    ret = binascii.crc32(val)
    return ret >> 31


def crc32(crc, p, len):
    crc = 0xffffffff & ~crc
    for i in range(len):
        crc = crc ^ p[i]
        for j in range(8):
            crc = (crc >> 1) ^ (0xedb88320 & -(crc & 1))
    return 0xffffffff & ~crc


rv = os.urandom(32)  # 伪随机

ret1 = crc32(0, rv, 32)
ret2 = binascii.crc32(rv)
print(ret1, ret2)

exit()


class MyDataSet(torch.utils.data.Dataset):

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
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

    def __init__(self, data_path, batch_size=512):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size

    def prepare_data(self):
        dataset = MyDataSet(self.data_path)

        # train:val:test = 6:2:2
        self.train = torch.utils.data.Subset(
            dataset, range(0, int(len(dataset)*0.6)))
        self.valid = torch.utils.data.Subset(
            dataset, range(int(len(dataset)*0.6), int(len(dataset)*0.8)))
        self.test = torch.utils.data.Subset(
            dataset, range(int(len(dataset)*0.8), int(len(dataset))))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':
    # dataset = MyDataSet(data_path='data/urandom.bin')
    dataset = MyDataSet(data_path='data/2022-01-19.bin')
    print(len(dataset))
    print(dataset[0])

    exit()

    # 生成伪随机数文件
    data = os.urandom(1024*1024)  # 1MB
    with open('./data/urandom.bin', 'wb') as f:
        f.write(data)
