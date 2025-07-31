import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('C:/Users/dipan/Human Machine Teaming/data_2.xlsx')

x = df.drop(columns=['label'])
y = df['label']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

import torch
x_tensor = torch.tensor(scaler.transform(x), dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

new_data = pd.concat([
    df.iloc[0:71, :],
    df.iloc[279:350, :],
    df.iloc[570:641, :],
    df.iloc[861:932, :],
    df.iloc[1152:1223, :],
], ignore_index=True)

features = new_data.iloc[:, :-1] 
target = new_data.iloc[:, -1]  
col_stds = features.std() 
noise = pd.DataFrame(
    np.random.normal(loc=0, scale=0.01 * col_stds.values, size=features.shape),
    columns=features.columns
)
noisy_features = features + noise

new_data_noisy = pd.concat([noisy_features, target.reset_index(drop=True)], axis=1)

import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(24, 15)
        self.fc2 = nn.Linear(15, 6)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
model = MLP()

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

for epoch in range(1600):
    model.train()
    optimizer.zero_grad()
    output = model(x_tensor)
    loss = loss_func(output, y_tensor)
    loss.backward()
    optimizer.step()
    print(f'for epoch {epoch+1}: {loss.item()}')
    
with torch.no_grad():
    predictions = model(torch.tensor(scaler.transform(new_data_noisy.drop(columns=['label'])), dtype=torch.float32))
    probs = torch.softmax(predictions, dim=1)

plt.figure(figsize=(16,8))
plt.plot(probs, label=['Normal State','Decrease in coolant flow rate','Increase in feed flow rate','Increase in coolant flow rate','Decrease in feed flowrate','Catalyst Poisoning'])
plt.legend()
plt.title('For Noisy Dataset')
plt.grid()
plt.show()

with torch.no_grad():
    predictions = model(torch.tensor(scaler.transform(new_data.drop(columns=['label'])), dtype=torch.float32))
    probs = torch.softmax(predictions, dim=1)

plt.figure(figsize=(16,8))
plt.plot(probs, label=['Normal State','Decrease in coolant flow rate','Increase in feed flow rate','Increase in coolant flow rate','Decrease in feed flowrate','Catalyst Poisoning'])
plt.legend()
plt.title('For original Dataset')
plt.grid()
plt.show()

scaler1 = MinMaxScaler()

for col in new_data.columns[0:8]:
    plt.plot(scaler1.fit_transform(new_data[[col]]), label=col, color='brown')
    plt.legend()
    plt.show()
    
for col in new_data.columns[0:8]:
    plt.plot(scaler1.fit_transform(new_data_noisy[[col]]), label=col, color='red')
    plt.legend()
    plt.show()

def shap_model(data):
    return model(torch.tensor(scaler.transform(data), dtype=torch.float32))

data = new_data_noisy.drop(columns=['label'])

import shap
explainer = shap.Explainer(shap_model, data)
shap_values = explainer(data)

shap0 = {
    'T103': np.mean(sum(np.abs(shap_values[:,j,0].values) for j in [0, 8, 16])),
    'T104': np.mean(sum(np.abs(shap_values[:,j,0].values) for j in [0+1, 8+1, 16+1])),
    'T105': np.mean(sum(np.abs(shap_values[:,j,0].values) for j in [0+2, 8+2, 16+2])),
    'T106': np.mean(sum(np.abs(shap_values[:,j,0].values) for j in [0+3, 8+3, 16+3])),
    'C101': np.mean(sum(np.abs(shap_values[:,j,0].values) for j in [0+4, 8+4, 16+4])),
    'F101': np.mean(sum(np.abs(shap_values[:,j,0].values) for j in [0+5, 8+5, 16+5])),
    'F102': np.mean(sum(np.abs(shap_values[:,j,0].values) for j in [0+6, 8+6, 16+6])),
    'F105': np.mean(sum(np.abs(shap_values[:,j,0].values) for j in [7+0, 8+7, 16+7]))
}
sns.barplot(shap0)
plt.title('Normal State')
plt.show()

shap1 = {
    'T103': np.mean(sum(np.abs(shap_values[:,j,1].values) for j in [0, 8, 16])),
    'T104': np.mean(sum(np.abs(shap_values[:,j,1].values) for j in [0+1, 8+1, 16+1])),
    'T105': np.mean(sum(np.abs(shap_values[:,j,1].values) for j in [0+2, 8+2, 16+2])),
    'T106': np.mean(sum(np.abs(shap_values[:,j,1].values) for j in [0+3, 8+3, 16+3])),
    'C101': np.mean(sum(np.abs(shap_values[:,j,1].values) for j in [0+4, 8+4, 16+4])),
    'F101': np.mean(sum(np.abs(shap_values[:,j,1].values) for j in [0+5, 8+5, 16+5])),
    'F102': np.mean(sum(np.abs(shap_values[:,j,1].values) for j in [0+6, 8+6, 16+6])),
    'F105': np.mean(sum(np.abs(shap_values[:,j,1].values) for j in [7+0, 8+7, 16+7]))
}
sns.barplot(shap1)
plt.title('Decrease in coolant flowrate')
plt.show()

shap2 = {
    'T103': np.mean(sum(np.abs(shap_values[:,j,2].values) for j in [0, 8, 16])),
    'T104': np.mean(sum(np.abs(shap_values[:,j,2].values) for j in [0+1, 8+1, 16+1])),
    'T105': np.mean(sum(np.abs(shap_values[:,j,2].values) for j in [0+2, 8+2, 16+2])),
    'T106': np.mean(sum(np.abs(shap_values[:,j,2].values) for j in [0+3, 8+3, 16+3])),
    'C101': np.mean(sum(np.abs(shap_values[:,j,2].values) for j in [0+4, 8+4, 16+4])),
    'F101': np.mean(sum(np.abs(shap_values[:,j,2].values) for j in [0+5, 8+5, 16+5])),
    'F102': np.mean(sum(np.abs(shap_values[:,j,2].values) for j in [0+6, 8+6, 16+6])),
    'F105': np.mean(sum(np.abs(shap_values[:,j,2].values) for j in [7+0, 8+7, 16+7]))
}
sns.barplot(shap2)
plt.title('Increase in feed flowrate')
plt.show()

shap3 = {
    'T103': np.mean(sum(np.abs(shap_values[:,j,3].values) for j in [0, 8, 16])),
    'T104': np.mean(sum(np.abs(shap_values[:,j,3].values) for j in [0+1, 8+1, 16+1])),
    'T105': np.mean(sum(np.abs(shap_values[:,j,3].values) for j in [0+2, 8+2, 16+2])),
    'T106': np.mean(sum(np.abs(shap_values[:,j,3].values) for j in [0+3, 8+3, 16+3])),
    'C101': np.mean(sum(np.abs(shap_values[:,j,3].values) for j in [0+4, 8+4, 16+4])),
    'F101': np.mean(sum(np.abs(shap_values[:,j,3].values) for j in [0+5, 8+5, 16+5])),
    'F102': np.mean(sum(np.abs(shap_values[:,j,3].values) for j in [0+6, 8+6, 16+6])),
    'F105': np.mean(sum(np.abs(shap_values[:,j,3].values) for j in [7+0, 8+7, 16+7]))
}
sns.barplot(shap3)
plt.title('Increase in coolant flowrate')
plt.show()

shap4 = {
    'T103': np.mean(sum(np.abs(shap_values[:,j,4].values) for j in [0, 8, 16])),
    'T104': np.mean(sum(np.abs(shap_values[:,j,4].values) for j in [0+1, 8+1, 16+1])),
    'T105': np.mean(sum(np.abs(shap_values[:,j,4].values) for j in [0+2, 8+2, 16+2])),
    'T106': np.mean(sum(np.abs(shap_values[:,j,4].values) for j in [0+3, 8+3, 16+3])),
    'C101': np.mean(sum(np.abs(shap_values[:,j,4].values) for j in [0+4, 8+4, 16+4])),
    'F101': np.mean(sum(np.abs(shap_values[:,j,4].values) for j in [0+5, 8+5, 16+5])),
    'F102': np.mean(sum(np.abs(shap_values[:,j,4].values) for j in [0+6, 8+6, 16+6])),
    'F105': np.mean(sum(np.abs(shap_values[:,j,4].values) for j in [7+0, 8+7, 16+7]))
}
sns.barplot(shap4)
plt.title('Decrease in feed flowrate')
plt.show()

shap5 = {
    'T103': np.mean(sum(np.abs(shap_values[:,j,5].values) for j in [0, 8, 16])),
    'T104': np.mean(sum(np.abs(shap_values[:,j,5].values) for j in [0+1, 8+1, 16+1])),
    'T105': np.mean(sum(np.abs(shap_values[:,j,5].values) for j in [0+2, 8+2, 16+2])),
    'T106': np.mean(sum(np.abs(shap_values[:,j,5].values) for j in [0+3, 8+3, 16+3])),
    'C101': np.mean(sum(np.abs(shap_values[:,j,5].values) for j in [0+4, 8+4, 16+4])),
    'F101': np.mean(sum(np.abs(shap_values[:,j,5].values) for j in [0+5, 8+5, 16+5])),
    'F102': np.mean(sum(np.abs(shap_values[:,j,5].values) for j in [0+6, 8+6, 16+6])),
    'F105': np.mean(sum(np.abs(shap_values[:,j,5].values) for j in [7+0, 8+7, 16+7]))
}
sns.barplot(shap5)
plt.title('Catalytic poisoning')
plt.show()

new_data_noisy[new_data_noisy['label'] == 5]

arr01 = shap_values[0:21, :, 0].values
arr1 = shap_values[21:71, :, 1].values
arr02 = shap_values[71:92, :, 0].values
arr2 = shap_values[92:142,:,2].values
arr03 = shap_values[142:163,:,0].values
arr3 = shap_values[163:213,:,3].values
arr04 = shap_values[213:234, :, 0].values
arr4 = shap_values[234:284, :, 0].values
arr05 = shap_values[284:305, :, 0].values
arr5 = shap_values[305:355, :, 0].values

full = np.vstack((arr01, arr1, arr02, arr2, arr03, arr3, arr04, arr4, arr05, arr5))

full = np.transpose(np.vstack((full[:,0] + full[:,8] + full[:,16],
                     full[:,1] + full[:,9] + full[:,17],
                     full[:,2] + full[:,10] + full[:,18],
                     full[:,3] + full[:,11] + full[:,19],
                     full[:,4] + full[:,12] + full[:,20],
                     full[:,5] + full[:,13] + full[:,21],
                     full[:,6] + full[:,14] + full[:,22],
                     full[:,7] + full[:,15] + full[:,23])))



plt.plot(np.abs(full[0:71]), label=new_data_noisy.iloc[:,0:8].columns)
plt.legend()
plt.title('Attribution of Decrease in coolant flow')
plt.show()

plt.plot(np.abs(full[71:142]), label=new_data_noisy.iloc[:,0:8].columns)
plt.legend()
plt.title('Attribution of Increase in feed flow')
plt.show()

plt.plot(np.abs(full[142:213]), label=new_data_noisy.iloc[:,0:8].columns)
plt.legend()
plt.title('Attribution of Increase in coolant flow')
plt.show()

plt.plot(np.abs(full[213:284]), label=new_data_noisy.iloc[:,0:8].columns)
plt.legend()
plt.title('Attribution of Decrease in feed flow')
plt.show()

plt.plot(np.abs(full[284:355]), label=new_data_noisy.iloc[:,0:8].columns)
plt.legend()
plt.title('Attribution of Catalyst Poisoning')
plt.show()


shap.summary_plot(shap_values[:,:,0], features=data, feature_names=new_data_noisy.columns)
shap.summary_plot(shap_values[:,:,1], features=data, feature_names=new_data_noisy.columns)
shap.summary_plot(shap_values[:,:,2], features=data, feature_names=new_data_noisy.columns)
shap.summary_plot(shap_values[:,:,3], features=data, feature_names=new_data_noisy.columns)
shap.summary_plot(shap_values[:,:,4], features=data, feature_names=new_data_noisy.columns)
shap.summary_plot(shap_values[:,:,5], features=data, feature_names=new_data_noisy.columns)


with torch.no_grad():
    predictions = model(torch.tensor(scaler.transform(new_data_noisy.drop(columns=['label'])), dtype=torch.float32))
    predicted_probs = torch.softmax(predictions, dim=1)

predicted_probs = np.array(predicted_probs)
actual_fault_indices = {
    1: 21,
    2: 92,
    3: 163,
    4: 234,
    5: 305,
    6: 355
}

def evaluation(fault_id):      
    index = None
    for i in range(actual_fault_indices[fault_id], actual_fault_indices[fault_id+1]-5):
        if all((np.argmax(predicted_probs[i+j]) == fault_id and 
                np.max(predicted_probs[i+j]) >= 0.95) for j in range(5)):
            index = i+4
            break
    
    if index is not None:
        print(index - actual_fault_indices[fault_id])
    else:
        print("No 5-sequence found")

for fault in range(1,6):
    evaluation(fault)



