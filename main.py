import dataset
from model import CharRNN, CharLSTM
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import dataset as ds
from model import CharRNN, CharLSTM
import torch.optim as optim
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    model.train()
    trn_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = model.init_hidden(inputs.size(0))
        hidden = tuple([h.to(device) for h in hidden]) if isinstance(hidden, tuple) else hidden.to(device)
        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    return trn_loss / len(trn_loader)

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))
            hidden = tuple([h.to(device) for h in hidden]) if isinstance(hidden, tuple) else hidden.to(device)
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

def plot_loss(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss per Epoch')
    plt.xticks(range(1, len(train_losses) + 1))
    plt.legend()
    plt.savefig(f'{model_name.lower()}_loss.png')
    plt.show()

if __name__ == '__main__':
    input_file = './shakespeare_train.txt'
    
    print("Training CharRNN")
    
    shakespeare_dataset = ds.Shakespeare(input_file)
    
    train_idx, val_idx = train_test_split(list(range(len(shakespeare_dataset))), test_size=0.2)
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(shakespeare_dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(shakespeare_dataset, batch_size=64, sampler=val_sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CharRNN(len(shakespeare_dataset.chars), hidden_size=256, n_layers=2).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    n_epochs = 20
    
    train_losses = []
    val_losses = []
    best_val_loss_rnn = float('inf')  # Initialize with infinity
    best_epoch_rnn = -1

    for epoch in range(1, n_epochs + 1):
        trn_loss = train(model, train_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)
        train_losses.append(trn_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}, Training Loss: {trn_loss}, Validation Loss: {val_loss}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss_rnn:
            best_val_loss_rnn = val_loss
            best_epoch_rnn = epoch
            torch.save(model.state_dict(), f'charrnn_best_model_epoch_{epoch}.pth')

    plot_loss(train_losses, val_losses, "CharRNN")

    print("Training CharLSTM")
    
    model = CharLSTM(len(shakespeare_dataset.chars), hidden_size=256, n_layers=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.002)

    train_losses = []
    val_losses = []
    best_val_loss_lstm = float('inf')  # Initialize with infinity
    best_epoch_lstm = -1

    for epoch in range(1, n_epochs + 1):
        trn_loss = train(model, train_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)
        train_losses.append(trn_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}, Training Loss: {trn_loss}, Validation Loss: {val_loss}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss_lstm:
            best_val_loss_lstm = val_loss
            best_epoch_lstm = epoch
            torch.save(model.state_dict(), f'charlstm_best_model_epoch_{epoch}.pth')

    plot_loss(train_losses, val_losses, "CharLSTM")

    print(f"Best CharRNN Model: Epoch {best_epoch_rnn}, Validation Loss: {best_val_loss_rnn}")
    print(f"Best CharLSTM Model: Epoch {best_epoch_lstm}, Validation Loss: {best_val_loss_lstm}")
