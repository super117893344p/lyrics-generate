import datetime
import glob  # 查找符合特定规则的路径文本
import numpy as np
import torch
import torch.nn.functional as F  # 实现神经网络的函数
import torch.optim as optim  # 用于优化神经网络的参数
from torch import nn  # 从torch模块中导入nn子模块，用于构建神经网络。
import torch.utils.data as data
# from  torch.utils.data import  DataLoader, Dataset, random_split
import torch.utils.tensorboard as tb  #

# from torch.utils.tensorboard import SummaryWriter
debug = False  # 用于控制程序的调试模式。
embed_size = 128  # 初始值为128，表示嵌入层的维度
hidden_size = 1024  # ，初始值为1024，表示隐藏层的大小
lr = 0.001  # ，初始值为0.001，表示学习率
lstm_layers = 2  # 初始值为2，表示LSTM网络的层数
batch_size = 32  # 初始值为32，表示每个批次的数据量
epochs = 15  # 初始值为15，表示训练的总轮数。
seq_len = 48  # 初始值为48，表示序列的长度。
from torch.nn import Transformer
import torch
import torch.nn.utils.rnn as rnn_utils

# 修改超参数
embed_size = 256  # 嵌入层维度
hidden_size = 512  # 隐藏层大小
lr = 0.0001  # 学习率
num_layers = 4  # 模型层数
batch_size = 64  # 批次大小
epochs = 20  # 训练轮次
dropout = 0.1  # Dropout率，根据需要设置


class LyricsDataset(data.Dataset):
    def __init__(self, seq_len, file="data/lyrics.txt"):
        SOS = 0
        EOS = 1
        self.seq_len = seq_len
        with open(file, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.word2index = {"<SOS>": SOS, "<EOS>": EOS}
        indices = []
        num_words = 2
        for line in lines:
            indices.append(SOS)
            for word in line:
                if word not in self.word2index:
                    self.word2index[word] = num_words
                    num_words += 1
                indices.append(self.word2index[word])
            indices.append(EOS)
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.data = np.array(indices, dtype=np.int64)

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, i):
        start = i * self.seq_len
        end = start + self.seq_len
        return (
            torch.as_tensor(self.data[start:end]),
            torch.as_tensor(self.data[start + 1: end + 1]),
        )


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = src.permute(1, 0, 2)  # 将形状变为 (seq_len, batch_size, d_model)
        tgt = tgt.permute(1, 0, 2)  # 将形状变为 (seq_len, batch_size, d_model)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.fc(output)
        return output
import torch
import torch.nn.functional as F


def training_step(model, optimizer, criterion, train_loader, device, epoch, vocab_size, debug=False):
    for i, (input_, target) in enumerate(train_loader):
        model.train()
        input_, target = input_.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input_,target)
        # 计算损失
        loss = criterion(output.view(-1, vocab_size), target.view(-1))  # 将输出和目标展平后计算损失
        loss.backward()
        optimizer.step()
        # 计算准确率
        pred = output.argmax(dim=-1)  # 获取预测值
        correct = torch.sum(pred == target.view(-1))  # 计算正确预测的数量
        acc = correct.item() / target.numel()  # 计算准确率
        print("Training: Epoch=%d, Batch=%d/%d, Loss=%.4f, Accuracy=%.4f"
              % (epoch, i, len(train_loader), loss.item(), acc))
        if not debug:
            step = epoch * len(train_loader) + i
            writer.add_scalar("loss/training", loss.item(), step)
            writer.add_scalar("accuracy/training", acc, step)
            if i % 50 == 0:
                generated_lyrics = generate("机/器/学/习")
                writer.add_text("generated_lyrics", generated_lyrics, i)
                writer.flush()
def evaluation_step(model, test_loader, device, epoch, writer=None, debug=False):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # 在评估步骤中不需要计算梯度，使用torch.no_grad()上下文管理器
        for data in test_loader:  # 便利测试数据集
            input_, target = data[0].to(device), data[1].to(device)  # 将输入和目标数据移动到指定的设备

            # 前向传播
            output = model(input_)

            # 计算损失
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
            total_loss += loss.item()

            # 统计准确率
            pred = output.argmax(dim=-1)
            correct = torch.sum(pred == target.view(-1))
            total_correct += correct.item()
            total_samples += target.numel()

    # 计算平均损失和平均准确率
    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_correct / total_samples
    # 打印当前评估的轮次、损失和准确率。
    print(
        "Validation: Epoch={}, Loss={:.4f}, Accuracy={:.4f}".format(
            epoch, avg_loss, avg_accuracy
        )
    )
    # 如果不是调试模式且 writer 不为空，则将当前轮次的损失和准确率写入到 TensorBoard 中。
    if not debug and writer:
        writer.add_scalar("loss/validation", avg_loss, epoch)
        writer.add_scalar("accuracy/validation", avg_accuracy, epoch)
        writer.flush()


def save_checkpoint():  # 保存当前以训练轮次epoch到检查点 ,下次训练从检查点开始
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),  # 模型状态字典
            "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态字典
        },
        "transformer-checkpoint-%s.pth" % datetime.datetime.now().strftime("%y%m%d-%H%M%S"),  # 保存文件的路径
    )


def load_checkpoint(file):
    global epoch  # 声明为全局变量
    ckpt = torch.load(file)  # 加载保存文件路径
    print("Loading checkpoint from %s." % file)  # 打印路径信息
    model.load_state_dict(ckpt["model_state_dict"])  # 将模型的状态字典加载到模型中
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # 将优化器状态字典加载到模型
    epoch = ckpt["epoch"]


def generate(start_phrases, model, dataset, device, seq_len=100):
    # 将输入的起始短语按照斜杠分割成一个列表
    start_phrases = start_phrases.split("/")

    def next_word(input_word, hidden):
        input_word_index = dataset.word2index[input_word]  # 将输入的单词转换为对应的索引
        input_ = torch.Tensor([[input_word_index]]).long().to(device)  # 将单词索引转换为张量，并移动到指定的设备上。
        output, hidden = model(input_, hidden)  # 将输入的单词索引和隐藏状态输入到模型中进行前向传播，得到模型的输出和更新后的隐藏状态
        top_word_index = output[0].topk(1).indices.item()  # 找到输出中概率最大的单词的索引
        return dataset.index2word[top_word_index], hidden  # 将索引转换为对应的单词，并返回单词和更新后的隐藏状态

    result = []  # 创建一个空列表，用于存储生成的单词
    cur_word = "/"  # 将当前单词初始化为斜杠
    hidden = None  # 初始化隐藏状态为 None

    for i in range(seq_len):  # 循环生成指定长度的单词
        if cur_word == "/":  # 如果当前单词是斜杠，表示一个句子的结束
            result.append(cur_word)  # 将当前单词添加到生成的单词列表中
            _, hidden = next_word(cur_word, hidden)  # 调用next_word函数生成下一个单词，并更新隐藏状态
            if len(start_phrases) == 0:  # 如果起始短语列表为空，则跳出循环
                break
            for w in start_phrases.pop(0):  # 遍历起始短语列表中的每个单词
                result.append(w)  # 将当前单词添加到生成的单词列表中
                cur_word, hidden = next_word(w, hidden)  # 更新当前单词为生成的下一个单词，并更新隐藏状态
        else:  # 如果当前单词不是斜杠，表示一个句子的中间部分
            result.append(cur_word)  # 将当前单词添加到生成的单词列表
            cur_word, hidden = next_word(cur_word, hidden)  # 更新当前单词为生成的下一个单词，并更新隐藏状态

    result = "".join(result)  # 将生成的单词列表转换为一个字符串
    result = result.strip("/")  # 移除末尾的斜杠
    return result  # 返回生成的字符串


vocab_size = 10000
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1

if __name__ == '__main__':
    # 创建 cuda 设备以在 GPU 上训练模型 创建CUDA设备对象 代码创建一个CUDA设备对象，如果可用的话就选择GPU 0，否则选择CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define dataset 创建数据集 使用LyricsDataset类定义了一个数据集，seq_len参数指定了每个序列的长度。然后使用random_split函数将数据集分割为训练集和验证集，其中训练集长度为data_length - 1000，验证集长度为1000
    dataset = LyricsDataset(seq_len=seq_len)  #
    data_length = len(dataset)  # 获取数据集长度
    lengths = [int(data_length - 1000), 1000]
    train_data, test_data = data.random_split(dataset, lengths)
    # Create data loader  使用data.DataLoader类创建了训练集和验证集的数据加载器，batch_size参数指定了每个批次的大小，shuffle参数指定是否随机打乱数据，num_workers参数指定了使用多少个进程来加载数据
    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    if debug:
        train_loader = [next(iter(train_loader))]
        test_loader = [next(iter(test_loader))]
    # Sanity check: view training data
    if debug:
        i = 0
        for data in train_loader:
            if i >= 10:
                break
            input_batch, _ = data
            first_sample = input_batch[0]
            print("".join([dataset.index2word[x.item()] for x in first_sample]))
            i += 1

    # Create NN model 使用LyricsNet类创建了一个神经网络模型，其中vocab_size参数指定了词汇表的大小，embed_size参数指定了嵌入层的维度，hidden_size参数指定了LSTM隐藏层的大小，lstm_layers参数指定了LSTM层数。然后将模型移动到指定的设备上。
    vocab_size = len(dataset.word2index)
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    model = model.to(device)
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 自适应调整梯度参数
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 随机梯度下降
    # Load checkpoint 将其绑定到模型参数上。然后使用glob.glob函数查找所有的检查点文件，并加载最新的检查点（如果用户选择）。
    criterion = nn.CrossEntropyLoss()
    checkpoint_files = glob.glob("transformer-checkpoint-*.pth")
    # 根据用户输入选择进入推理模式还是训练模式
    if (
            not debug
            and len(checkpoint_files) > 0
            and input("Enter y to load %s: " % checkpoint_files[-1]) == "y"
    ):
        load_checkpoint(checkpoint_files[-1])
    else:
        epoch = 0
    if (
            input("输入 y 进入推理模式, 进入训练模式的按其他键: ")
            == "y"
    ):
        # 推理循环开始
        while True:
            start_words = input("输入起始词用 '/'分割 (e.g. '深/度/学/习'): ")
            if not start_words:
                break
            print(generate(start_words,model,dataset,device))
    else:  # 训练模式开始
        if not debug:
            writer = tb.SummaryWriter()  # SummaryWriter对象，用于记录训练过程中的指标和可视化
        # 优化循环
        while epoch < epochs:  # 一次最多连续训练15次
            training_step(model,optimizer,criterion,train_loader, device, epoch, vocab_size,debug=False)  # 执行一次训练步骤，更新模型的参数
            evaluation_step()  # 执行一次评估步骤，计算模型在验证集上的性能指标
            if not debug:
                save_checkpoint()  # 保存当前的模型参数到检查点文件
            epoch += 1  # 进入下一轮次
