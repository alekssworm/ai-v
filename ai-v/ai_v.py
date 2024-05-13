import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ���������� ����� ��������� ���� � ����������� �������� ������
class ComplexNeuralNetwork(nn.Module):
    def __init__(self):
        super(ComplexNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # ������ ������� ����
        self.fc2 = nn.Linear(10, 10)  # ������ ������� ����
        self.fc3 = nn.Linear(10, 10)  # ������ ������� ����
        self.fc4 = nn.Linear(10, 1)  # �������� ����

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # ��������� ���������� ������� ��������� � ������� �������� ����
        x = torch.sigmoid(self.fc2(x))  # ��������� ���������� ������� ��������� �� ������� �������� ����
        x = torch.sigmoid(self.fc3(x))  # ��������� ���������� ������� ��������� � �������� �������� ����
        x = self.fc4(x)  # �������� ���� ��� ���������
        return x

# ������� ��������� ����
model = ComplexNeuralNetwork()


input_data = torch.tensor([[0.5], [1.0], [1.5]])  # ������� ������

plt.figure(figsize=(10, 8))

# �������� ������ �� �������� ���� � ������� �������� ����
output_data = torch.sigmoid(model.fc1(input_data))
plt.subplot(1, 4, 1)
plt.plot(input_data.numpy(), output_data.detach().numpy(), 'ro-', label='Input to Hidden Layer 1')
plt.xlabel('Input Data')
plt.ylabel('Output Data')
plt.title('Input -> Hidden Layer 1')
plt.grid(True)

# �������� ������ �� ������� �������� ���� �� ������� �������� ����
output_data = torch.sigmoid(model.fc2(output_data))
plt.subplot(1, 4, 2)
plt.plot(input_data.numpy(), output_data.detach().numpy(), 'go-', label='Hidden Layer 1 to Hidden Layer 2')
plt.xlabel('Input Data')
plt.ylabel('Output Data')
plt.title('Hidden Layer 1 -> Hidden Layer 2')
plt.grid(True)

# �������� ������ �� ������� �������� ���� � �������� �������� ����
output_data = torch.sigmoid(model.fc3(output_data))
plt.subplot(1, 4, 3)
plt.plot(input_data.numpy(), output_data.detach().numpy(), 'bo-', label='Hidden Layer 2 to Hidden Layer 3')
plt.xlabel('Input Data')
plt.ylabel('Output Data')
plt.title('Hidden Layer 2 -> Hidden Layer 3')
plt.grid(True)

# �������� ������ �� �������� �������� ���� � ��������� ����
output_data = model.fc4(output_data)
plt.subplot(1, 4, 4)
plt.plot(input_data.numpy(), output_data.detach().numpy(), 'mo-', label='Hidden Layer 3 to Output')
plt.xlabel('Input Data')
plt.ylabel('Output Data')
plt.title('Hidden Layer 3 -> Output')
plt.grid(True)

plt.tight_layout()
plt.show()
