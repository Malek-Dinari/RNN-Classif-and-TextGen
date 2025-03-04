import matplotlib.pyplot as plt
from utils.visualize import explain_with_shap, explain_with_lime

def plot_training_curves(rnn_loss, lstm_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(rnn_loss, label='RNN')
    plt.plot(lstm_loss, label='LSTM')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('logs/comparison.png')

def compare_models(rnn_model, lstm_model, sample_text):
    # SHAP comparison
    explain_with_shap(rnn_model, sample_text)
    explain_with_shap(lstm_model, sample_text)
    
    # LIME comparison
    explain_with_lime(rnn_model, sample_text)
    explain_with_lime(lstm_model, sample_text)