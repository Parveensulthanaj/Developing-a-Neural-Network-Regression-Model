# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="940" height="633" alt="Neural Network Model" src="https://github.com/user-attachments/assets/7da66cb1-90f2-4b4f-b659-629b2b61908a" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Parveen Sulthana J

### Register Number:212224040233

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/exp1 - Sheet1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

dataset1.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#Name:Parveen Sulthana J
#Register Number:212224040233
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Include your code here
        self.fc1=nn.Linear(1,8) #fc=fully connected,nn=neural network,1.input
        self.fc2=nn.Linear(8,10)#2.hidden layer
        self.fc3=nn.Linear(10,1)#3.output
        self.relu=nn.ReLU() #activation part,Rectified Linear Unit
        self.history={'loss':[]}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
# Write your code here
ai_brain=NeuralNet()#class name
criterion=nn.MSELoss()#Mean square
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)#learning rate,Root means quare

# Name:Parveen Sulthana J
# Register Number: 212224040233
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```

### Dataset Information
<img width="193" height="459" alt="Excel Sheet" src="https://github.com/user-attachments/assets/a74a21ea-a5ca-47e2-a7dc-0992c4db194f" />


### OUTPUT
# InPut and Ouput:
<img width="308" height="273" alt="output" src="https://github.com/user-attachments/assets/d8c27d49-053e-4290-a9ee-8b38099b5b50" />

# Epoch And Loss
<img width="439" height="252" alt="Epochs" src="https://github.com/user-attachments/assets/9197e8f7-f1d0-4804-9580-6f4eb3b2e354" />

# Test Loss
<img width="298" height="35" alt="Test loss" src="https://github.com/user-attachments/assets/2d123d34-9a2f-4ba9-b2c1-ec4663556155" />



### Training Loss Vs Iteration Plot
<img width="833" height="611" alt="Graph" src="https://github.com/user-attachments/assets/0672aec6-69a0-41a3-9874-e423a3b61ab6" />


### New Sample Data Prediction
<img width="428" height="44" alt="Predictions" src="https://github.com/user-attachments/assets/ebfb1671-39a6-4b24-9601-53028b22c92b" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
