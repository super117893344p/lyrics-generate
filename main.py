#!/usr/bin/env python3
import datetime
import glob  # 查找符合特定规则的路径文本
import numpy as np
import torch
import torch.nn.functional as F  # 实现神经网络的函数
import torch.optim as optim  # 用于优化神经网络的参数
from torch import nn  # 从torch模块中导入nn子模块，用于构建神经网络。
import torch.utils.data as data
#from  torch.utils.data import  DataLoader, Dataset, random_split
import torch.utils.tensorboard as tb #
#from torch.utils.tensorboard import SummaryWriter
debug = False  # 用于控制程序的调试模式。
embed_size = 128  # 初始值为128，表示嵌入层的维度
hidden_size = 1024  # ，初始值为1024，表示隐藏层的大小
lr = 0.001  # ，初始值为0.001，表示学习率
lstm_layers = 2  # 初始值为2，表示LSTM网络的层数
batch_size = 32  # 初始值为32，表示每个批次的数据量
epochs = 15  # 初始值为15，表示训练的总轮数。
seq_len = 48  # 初始值为48，表示序列的长度。

if debug:  # notbug 就是在训练模式
    batch_size = 2
    epochs = 1000
class LyricsDataset(data.Dataset):    # 歌词数据集 自定义的数据集类，用于处理歌词数据
    def __init__(self, seq_len, file="data/lyrics.txt"):  # 样本长度与歌词路径
        SOS = 0  # start of song 表示歌曲的开始和结束标记。
        EOS = 1  # end of song
        self.seq_len = seq_len  # 将输入的seq_len赋值给类的seq_len属性。
        with open(file, encoding="utf-8") as f:
            lines = f.read().splitlines()  # 并使用splitlines方法将内容按行分割，得到一个包含每行歌词的列表
        self.word2index = {"<SOS>": SOS, "<EOS>": EOS}  # 初始化一个字典，用于将单词映射为索引。其中，"<SOS>"表示开始标记，"<EOS>"表示结束标记。
        # Convert words to indices
        indices = []  # 初始化一个空列表，用于存储歌词的索引
        num_words = 0  # 始化一个变量，用于记录不同单词的数量
        for line in lines:
            indices.append(SOS)  # 将开始标记添加到索引列表中
            for word in line:
                if word not in self.word2index:
                    self.word2index[word] = num_words  # 将单词添加到字典word2index中，并将其映射为当前的num_words
                    num_words += 1  # ：num_words加1，表示新增了一个单词。
                indices.append(self.word2index[word])  # 将单词的索引添加到索引列表中
            indices.append(EOS)  # 将结束标记添加到索引列表中
        self.index2word = {v: k for k, v in self.word2index.items()}  # 根据字典word2index，创建一个新的字典index2word，用于将索引映射回单词。
        self.data = np.array(indices, dtype=np.int64)  # 将索引列表转换为numpy数组，并将数据类型设置为int64，存储在类的data属性中。

    #   返回数据集的长度，即样本的数量。这里使用了整除运算符，确保返回的是整数
    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, i):  # 根据给定的索引i，i为样本,返回对应的输入和输出。start和end表示样本的起始和结束位置。返回的是两个torch tensor，分别表示输入和输出。
        start = i * self.seq_len
        end = start + self.seq_len
        return (
            torch.as_tensor(self.data[start:end]),  # input 将从起始位置到结束位置的数据切片转换为torch tensor，作为输入序列
            torch.as_tensor(self.data[start + 1: end + 1]),
            # output 将从起始位置+1到结束位置+1的数据切片转换为torch tensor，作为输出序列。 # tensor(张量)
            # 返回的是一个元组，包含了输入和输出序列，可以直接用于模型的训练和评估。

        )

class LyricsNet(nn.Module):  # 定义了一个名为LyricsNet的神经网络模型，用于生成歌词
    def __init__(self, vocab_size, embed_size, hidden_size,
                 lstm_layers):  # 定义了模型的初始化方法，接收四个参数：vocab_size表示词汇表的大小，embed_size表示词嵌入的维度，hidden_size表示LSTM隐藏层的大小，lstm_layers表示LSTM的层数。
        super().__init__()  # 调用父类的初始化方法。
        self.embedding = nn.Embedding(vocab_size, embed_size)  # 创建一个Embedding层，用于将输入的单词索引转换为词嵌入向量。
        self.lstm = nn.LSTM(embed_size, hidden_size, lstm_layers,
                            batch_first=True)  # 创建一个LSTM层，用于处理词嵌入序列。embed_size表示输入的特征维度，hidden_size表示隐藏层的大小，lstm_layers表示LSTM的层数，batch_first=True表示输入的维度顺序为(batch_size, seq_len, input_size)。
        self.h2h = nn.Linear(hidden_size, hidden_size)  # 创建一个线性层，用于将LSTM隐藏状态映射到隐藏状态。
        self.h2o = nn.Linear(hidden_size, vocab_size)  # 创建一个线性层，用于将隐藏状态映射到输出的词汇表大小。

    def forward(self, word_ids,
                lstm_hidden=None):  # 定义了模型的前向传播方法，接收两个参数：word_ids表示输入的单词索引序列，lstm_hidden表示LSTM的初始隐藏状态，默认为None。
        # 将输入的单词索引序列通过词嵌入层进行词嵌入，得到词嵌入向量。将输入的单词索引序列通过词嵌入层进行词嵌入，得到词嵌入向量。
        embedded = self.embedding(word_ids)
        # 将词嵌入向量输入到LSTM层中进行前向传播，得到LSTM的输出和最终的隐藏状态。
        lstm_out, lstm_hidden = self.lstm(embedded, lstm_hidden)
        # 将LSTM的输出通过线性层进行映射，得到隐藏状态
        out = self.h2h(lstm_out)
        # 将隐藏状态通过线性层进行映射，得到输出的词汇表大小
        out = self.h2o(out)  # 返回输出和最终的隐藏状态。
        return out, lstm_hidden

def accuracy(output, target):  # 用于计算模型输出和真实标签之间的准确率。
    """Compute the accuracy between model output and ground truth.
    Args:
        output: (batch_size, seq_len, vocab_size)
        target: (batch_size, seq_len)
    Returns:
        float: accuracy value between 0 and 1
    """
    output = output.reshape(-1, vocab_size)  # 将模型的输出展平为二维张量，大小为(batch_size * seq_len, vocab_size)
    target = target.flatten()  # 将真实标签展平为一维张量，大小为(batch_size * seq_len,
    a = output.topk(1).indices.flatten()  # 找到每个时间步的最大值的索引，即预测的单词，返回大小为(batch_size * seq_len,)的一维张量。
    b = target  # 将真实标签赋值给变量b。
    return a.eq(b).sum().item() / len(
        a)  # 计算预测正确的数量，除以总数量得到准确率，返回准确率。其中，a.eq(b)返回一个大小为(batch_size * seq_len,)的一维张量，表示预测正确的位置。.sum().item()将这个一维张量中所有元素相加，得到预测正确的数量。最后除以len(a)得到准确率。
def generate(start_phrases):  # generate的函数，用于生成文本。
    # Convert to a list of start words.
    # i.e. '宁可/无法' => ['宁可', '无法']
    start_phrases = start_phrases.split("/")  # 将输入的起始短语按照斜杠分割成一个列表，例如将'宁可/无法'分割成['宁可', '无法']。
    hidden = None  # 初始化隐藏状态为None

    def next_word(input_word):  # 定义了一个内部函数next_word，接收一个输入的单词作为参数。
        nonlocal hidden  # 声明hidden为非局部变量，使得next_word函数可以访问并修改hidden的值。
        input_word_index = dataset.word2index[input_word]  # 将输入的单词转换为对应的索引
        input_ = torch.Tensor([[input_word_index]]).long().to(device)  # 将单词索引转换为张量，并移动到指定的设备上。
        output, hidden = model(input_, hidden)  # 将输入的单词索引和隐藏状态输入到模型中进行前向传播，得到模型的输出和更新后的隐藏状态
        top_word_index = output[0].topk(1).indices.item()  # 找到输出中概率最大的单词的索引
        return dataset.index2word[top_word_index]  # 将索引转换为对应的单词，并返回

    result = []  # a list of output words 创建一个空列表，用于存储生成的单词`
    cur_word = "/"  # 将当前单词初始化为斜杠
    for i in range(seq_len):  # 循环生成指定长度的单词
        if cur_word == "/":  # end of a sentence  如果当前单词是斜杠，表示一个句子的结束
            result.append(cur_word)  # 将当前单词添加到生成的单词列表中
            next_word(cur_word)  # 调用next_word函数生成下一个单词
            if len(start_phrases) == 0:  # 如果起始短语列表为空，则跳出循环
                break
            for w in start_phrases.pop(0):  # 遍历起始短语列表中的每个单词
                result.append(w)  # 将当前单词添加到生成的单词列表中
                cur_word = next_word(w)  # 更新当前单词为生成的下一个单词
        else:  # 如果当前单词不是斜杠，表示一个句子的中间部
            result.append(cur_word)  # 将当前单词添加到生成的单词列表
            cur_word = next_word(cur_word)  # 更新当前单词为生成的下一个单词
    # Convert a list of generated words to a string
    result = "".join(result)  # 将生成的单词列表转换为一个字符串
    result = result.strip("/")  # remove trailing slashes 移除末尾/
    return result  # 返回字符串
def training_step():  # 名为training_step的函数，用于执行一次训练步骤
    for i, (input_, target) in enumerate(train_loader):  # 使用enumerate函数遍历训练数据集，获取每个批次的输入和目标
        model.train()  # 将模型设置为训练模式
        input_, target = input_.to(device), target.to(device)  # 将输入和目标数据移动到指定的设备上
        output, _ = model(input_)  # 将输入数据输入到模型中进行前向传播，得到模型的输出和隐藏状态
        loss = F.cross_entropy(output.reshape(-1, vocab_size), target.flatten())  # 计算输出和目标之间的交叉熵损失
        optimizer.zero_grad()  #  清空优化器中的梯度信息，避免梯度累加
        loss.backward()  # Compute gradient ：计算损失函数对模型参数的梯度
        optimizer.step()  # Update NN weights 使用优化器更新模型参数
        acc = accuracy(output, target)  # 计算模型的准确率
        print(
            "Training: Epoch=%d, Batch=%d/%d, Loss=%.4f, Accuracy=%.4f"  # 打印当前训练信息包括当前的训练轮次、批次编号、总批次数、损失和准确率
            % (epoch, i, len(train_loader), loss.item(), acc)
        )
        if not debug:
            step = epoch * len(train_loader) + i  # 计算当前步骤的总步数
            writer.add_scalar("loss/training", loss.item(), step)  # 将当前批次的损失写入到TensorBoard中
            writer.add_scalar("accuracy/training", acc, step)  # 将当前批次的准确率写入到TensorBoard中
            if i % 50 == 0:
                generated_lyrics = generate("机/器/学/习")  # 每50批次生成信息生成歌词
                writer.add_text("generated_lyrics", generated_lyrics, i)  # 将生成的歌词写入到TensorBoard中
                writer.flush()  # 将写入的数据刷新到TensorBoard中，确保数据及时可见
def evaluation_step():  # 名为evaluation_step的函数，用于执行一次评估
    model.eval()  # 将模型设置为评估模式
    epoch_loss = 0  # 初始化总损失和总准确率
    epoch_acc = 0
    with torch.no_grad():  # 在评估步骤中不需要计算梯度，使用torch.no_grad()上下文管理器
        for data in test_loader:  # 便利测试数据集
            input_, target = data[0].to(device), data[1].to(device)  # 将输入和目标数据移动到指定的设备
            output, _ = model(input_)  # 将输入数据输入到模型中进行前向传播，得到模型的输出和隐藏状态
            loss = F.cross_entropy(output.reshape(-1, vocab_size), target.flatten())  # 计算输出和目标之间的交叉熵损失
            epoch_acc += accuracy(output, target)  # 累加当前批次的准确率
            epoch_loss += loss.item()  # 累加当前批次损失率
    epoch_loss /= len(test_loader)  # 计算平均损失和平均准确
    epoch_acc /= len(test_loader)
    print(  # 打印当前评估的轮次、损失和准确率。
        "Validation: Epoch=%d, Loss=%.4f, Accuracy=%.4f"
        % (epoch, epoch_loss, epoch_acc)
    )
    if not debug:
        writer.add_scalar("loss/validation", epoch_loss, epoch)  # 将当前轮次的损失写入到TensorBoard中。
        writer.add_scalar("accuracy/validation", epoch_acc, epoch)  # 将当前轮次的损失写入到TensorBoard中
        writer.flush()  # 将写入的数据刷新到TensorBoard中，确保数据及时可见
def save_checkpoint():  # 保存当前以训练轮次epoch到检查点 ,下次训练从检查点开始
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),  # 模型状态字典
            "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态字典
        },
        "checkpoint-%s.pth" % datetime.datetime.now().strftime("%y%m%d-%H%M%S"),  # 保存文件的路径
    )
def load_checkpoint(file):
    global epoch  # 声明为全局变量
    ckpt = torch.load(file)  # 加载保存文件路径
    print("Loading checkpoint from %s." % file)  # 打印路径信息
    model.load_state_dict(ckpt["model_state_dict"])  # 将模型的状态字典加载到模型中
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # 将优化器状态字典加载到模型
    epoch = ckpt["epoch"]

import tkinter as tk
from tkinter import messagebox

def enter_mode():
    if entry.get() == "y":
        inference_mode()
    else:
        writer = tb.SummaryWriter()
        training_mode()

def inference_mode():
    start_words = entry_start_words.get()
    if not start_words:
        messagebox.showinfo("提示", "请输入起始词！")
    else:
        generated_lyrics = generate(start_words)
        result_text.config(state=tk.NORMAL)
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, generated_lyrics)
        result_text.config(state=tk.DISABLED)

def training_mode():
    epoch = 0
    writer = tb.SummaryWriter()
    while epoch < epochs:

        training_step()
        evaluation_step()
        save_checkpoint()
        epoch += 1



if __name__ == "__main__":
    # 创建 cuda 设备以在 GPU 上训练模型 创建CUDA设备对象 代码创建一个CUDA设备对象，如果可用的话就选择GPU 0，否则选择CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define dataset 创建数据集 使用LyricsDataset类定义了一个数据集，seq_len参数指定了每个序列的长度。然后使用random_split函数将数据集分割为训练集和验证集，其中训练集长度为data_length - 1000，验证集长度为1000
    dataset = LyricsDataset(seq_len=seq_len)  #
    data_length = len(dataset)  # 获取数据集长度
    lengths = [int(data_length - 1000), 1000]
    train_data, test_data =data.random_split(dataset, lengths)
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
    model = LyricsNet(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
    )
    model = model.to(device)
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 自适应调整梯度参数
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 随机梯度下降
    # Load checkpoint 将其绑定到模型参数上。然后使用glob.glob函数查找所有的检查点文件，并加载最新的检查点（如果用户选择）。
    checkpoint_files = glob.glob("checkpoint-*.pth")
    # 根据用户输入选择进入推理模式还是训练模式
    if (
            not debug
            and len(checkpoint_files) > 0
            and input("Enter y to load %s: " % checkpoint_files[-1]) == "y"
    ):
        load_checkpoint(checkpoint_files[-1])
    else:
        epoch = 0
    # 创建主窗口
    root = tk.Tk()
    root.title("模式选择")

    # 创建标签和输入框用于选择模式
    label = tk.Label(root, text="输入 y 进入推理模式，进入训练模式的按其他键：")
    label.grid(row=0, column=0, padx=10, pady=10)
    entry = tk.Entry(root)
    entry.grid(row=0, column=1, padx=10, pady=10)

    # 创建推理模式的起始词输入框和按钮
    label_start_words = tk.Label(root, text="输入起始词用 '/' 分割（例如 '深/度/学/习'）：")
    label_start_words.grid(row=1, column=0, padx=10, pady=10)
    entry_start_words = tk.Entry(root)
    entry_start_words.grid(row=1, column=1, padx=10, pady=10)
    inference_button = tk.Button(root, text="开始推理", command=inference_mode)
    inference_button.grid(row=1, column=2, padx=10, pady=10)

    # 创建显示结果的文本框
    result_text = tk.Text(root, height=10, width=50)
    result_text.grid(row=2, columnspan=3, padx=10, pady=10)
    result_text.config(state=tk.DISABLED)

    # 运行主循环
    root.mainloop()
    enter_mode()
    # if (
    #         input("输入 y 进入推理模式, 进入训练模式的按其他键: ")
    #         == "y"
    # ):
    #     # 推理循环开始
    #     while True:
    #         start_words = input("输入起始词用 '/'分割 (e.g. '深/度/学/习'): ")
    #         if not start_words:
    #             break
    #         print(generate(start_words))
    #         #GUI(start_words)
    # else:  # 训练模式开始
    #     if not debug:
    #         writer = tb.SummaryWriter()  # SummaryWriter对象，用于记录训练过程中的指标和可视化
    #     # 优化循环
    #     while epoch < epochs:  # 一次最多连续训练15次
    #         training_step()  # 执行一次训练步骤，更新模型的参数
    #         evaluation_step()  # 执行一次评估步骤，计算模型在验证集上的性能指标
    #         if not debug:
    #             save_checkpoint()  # 保存当前的模型参数到检查点文件
    #         epoch += 1  # 进入下一轮次
