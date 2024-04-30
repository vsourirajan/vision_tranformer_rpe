import matplotlib.pyplot as plt 

def plot_loss_accuracy(train_losses, train_accuracies, val_losses, val_accuracies, dataset, rpe_method, subvariant):
    epochs = range(1, len(train_losses) + 1)

    if rpe_method == "general":
        rpe_method = "General Learnable RPE"
    elif rpe_method == "monotonic":
        rpe_method = "Monotonically Decreasing RPE"
    else:
        rpe_method = "Polynomial Ratio RPE"


    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title(f'Training and Validation Loss of {rpe_method} Subvariant {subvariant} on {dataset}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title(f'Training and Validation Accuracy of {rpe_method} Subvariant {subvariant} on {dataset}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
