import torch
import dataset as ds
from model import CharRNN, CharLSTM

def generate(model, seed, temperature, char2idx, idx2char, device, length=100):
    model.eval()
    input_seq = torch.tensor([char2idx[ch] for ch in seed]).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    hidden = tuple([h.to(device) for h in hidden]) if isinstance(hidden, tuple) else hidden.to(device)
    samples = seed

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            output = output[-1, :]
            output = output.div(temperature).exp()
            char_idx = torch.multinomial(output, 1).item()
            char = idx2char[char_idx]
            samples += char
            input_seq = torch.cat((input_seq, torch.tensor([[char_idx]]).to(device)), dim=1)
            input_seq = input_seq[:, 1:]

    return samples

def save_generated_texts(filename, model_name, temperatures, seed_characters, texts):
    with open(filename, 'w') as f:
        for seed, seed_texts in zip(seed_characters, texts):
            f.write(f"Seed: {seed}\n")
            for temp, text in zip(temperatures, seed_texts):
                f.write(f"Model: {model_name}\n")
                f.write(f"Temperature: {temp}\n")
                f.write(text + "\n\n")
    print(f"Generated texts saved to {filename}")

if __name__ == '__main__':
    input_file = './shakespeare_train.txt'
    shakespeare_dataset = ds.Shakespeare(input_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed_characters = ["Apple", "University", "Seoul", "Coffee", "Computer"]
    temperatures = [0.5, 1.0, 1.5]
    
    # Load CharRNN model
    char_rnn = CharRNN(len(shakespeare_dataset.chars), hidden_size=256, n_layers=2).to(device)
    char_rnn.load_state_dict(torch.load('./charrnn_best_model_epoch_3.pth'))

    # Load CharLSTM model
    char_lstm = CharLSTM(len(shakespeare_dataset.chars), hidden_size=256, n_layers=2).to(device)
    char_lstm.load_state_dict(torch.load('./charlstm_best_model_epoch_19.pth'))

    rnn_generated_texts = []
    lstm_generated_texts = []

    print("Generating Texts (CharRNN):")
    for seed in seed_characters:
        rnn_texts = []
        for temp in temperatures:
            generated_text = generate(char_rnn, seed, temp, shakespeare_dataset.char2idx, shakespeare_dataset.idx2char, device)
            rnn_texts.append(generated_text)
        rnn_generated_texts.append(rnn_texts)

    print("Generating Texts (CharLSTM):")
    for seed in seed_characters:
        lstm_texts = []
        for temp in temperatures:
            generated_text = generate(char_lstm, seed, temp, shakespeare_dataset.char2idx, shakespeare_dataset.idx2char, device)
            lstm_texts.append(generated_text)
        lstm_generated_texts.append(lstm_texts)

    save_generated_texts("charrnn_generated_texts.txt", "CharRNN", temperatures, seed_characters, rnn_generated_texts)
    save_generated_texts("charlstm_generated_texts.txt", "CharLSTM", temperatures, seed_characters, lstm_generated_texts)
