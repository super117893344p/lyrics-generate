import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 假设你有一个包含歌词的文本文件，每行一个歌词
# 这里使用一个简单的例子，你需要替换成你的实际数据加载方式
with open("data/lyrics.txt", "r", encoding="utf-8") as file:
    lyrics_data = [line.strip() for line in file.readlines()]

# 数据预处理
word_to_index = {word: idx for idx, word in enumerate(set(" ".join(lyrics_data).split()))}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# 将歌词转换为索引序列
lyrics_indices = [[word_to_index[word] for word in lyric.split()] for lyric in lyrics_data]

# 构建 PyTorch DataLoader
X_train = torch.LongTensor(lyrics_indices)
y_train = torch.zeros(X_train.shape[0])  # 这里简化，假设标签都是 0

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义词嵌入模型
class LyricsEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(LyricsEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded_mean = embedded.mean(dim=1)  # 这里简化，取每个歌词的平均嵌入
        output = torch.sigmoid(self.fc(embedded_mean))
        return output

# 初始化模型和优化器
embedding_dim = 50  # 可根据实际情况调整
model = LyricsEmbeddingModel(vocab_size=len(word_to_index), embedding_dim=embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # 二分类交叉熵损失

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 获取训练好的词嵌入
trained_embedding = model.embedding.weight.data.numpy()

# 在这之后，你可以使用 trained_embedding 来获取歌词中每个词的嵌入向量
