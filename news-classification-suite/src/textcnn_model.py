import torch
import torch.nn as nn
import torch.nn.functional as functional


class StudentTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int = 21128,
        embed_dim: int = 128,
        num_labels: int = 10,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv2d(1, 128, (3, embed_dim))
        self.conv2 = nn.Conv2d(1, 128, (4, embed_dim))
        self.conv3 = nn.Conv2d(1, 128, (5, embed_dim))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 3, num_labels)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids).unsqueeze(1)
        pooled = []
        for kernel_size, convolution in (
            (3, self.conv1),
            (4, self.conv2),
            (5, self.conv3),
        ):
            feature = functional.relu(convolution(embedded)).squeeze(3)
            pooled.append(
                functional.max_pool1d(
                    feature, embedded.size(2) - kernel_size + 1
                ).squeeze(2)
            )
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))
