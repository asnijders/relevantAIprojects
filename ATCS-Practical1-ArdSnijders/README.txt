Each model implementation can be found in the file train.py.

Before the models can be trained, some data needs to be downloaded; the SNLI dataset and the GloVe vectors, namely.
These will be downloaded automatically when running the program the first time, though this might take some time.

The script can be used in two ways, primarily;
- Training with --mode=train
- Testing with  --mode=test

Training
When running the script in training mode, one can specify the model one wishes to train,
with config.model_type. This can be set to either 'AVERAGE' or 'LSTM' for training the average embedding encoder or
LSTM, respectively. The other config.arguments can be used to specify additional hyperparameters:

'--hidden_dim', type = int, default = 2048, help='Number of units in LSTM hidden layer')

'--emb_dim', type = int, default = 300,help='Embedding dimension for word embeddings')

'--bidirectional', type = bool, default=False,help='Flag for bi-LSTM')

'--maxpool', type=bool, default=False,help='Flag for maxpooling')

'--dropout', type=float, default=0.5,help='dropout probability for LSTM')

'--epochs', type=int, default=15,help='max amount of epochs to train for')

'--batch_size', type=int, default=64,help='Number of datapoints per minibatch')

'--learning_rate', type = float, default = 0.1,help='Learning rate for SGD')

'--print_every', type=int, default=1000,help= 'Number of batches between loss prints')

'--path', type=str, default=None, help='path for saving/loading model and classifier checkpoints')

Testing
When running the script in test mode, one can load the desired model with config.model_type. Furthermore,
applicable config args can be used to specify bidirectionality, maxpooling, etcetera. In test mode, --epochs is
used to specify which checkpoint should be loaded. The load-path is then dynamically created based on the args
provided for these parameters.

The checkpoints for both the encoder and classifier models can be found in the folder models.

The notebook provided in this zip should be pre-loaded; if this is not the case, the included vocabulary.pickle and glovevecs.pickle, along with the model checkpoints in ./models can be used to run everything. 



