# Training hyperparameters
batch_size    = 64
epochs        = 10
learning_rate = 0.001

# CNN architecture
conv1_out    = 32    # filters in the 1st conv layer
conv2_out    = 64    # filters in the 2nd conv layer
kernel_size1 = 3     # kernel size for conv1
kernel_size2 = 3     # kernel size for conv2
fc1_units    = 128   # neurons in the first FC layer
dropout_p    = 0.25  # dropout probability


folder_name/
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
└── predict.py
