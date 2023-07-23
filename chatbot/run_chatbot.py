
import math
import numpy
import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds

from matplotlib import pyplot as plt

from chatbot import transformer

# Maximum number of samples to preprocess
MAX_SAMPLES = 50000

# Maximum sentence length
MAX_LENGTH = 40

BATCH_SIZE = 64
BUFFER_SIZE = 20000

EPOCHS = 20


def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  # adding a start and an end token to the sentence
  return sentence


def load_conversations(path_to_lines, path_to_conversations):
  # dictionary of line id to text
  id2line = {}
  with open(path_to_lines, errors='ignore') as file:
    lines = file.readlines()
  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')
    id2line[parts[0]] = parts[4]

  inputs, outputs = [], []
  with open(path_to_conversations, 'r') as file:
    lines = file.readlines()
  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')
    # get conversation in a list of line ID
    conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
    for i in range(len(conversation) - 1):
      inputs.append(preprocess_sentence(id2line[conversation[i]]))
      outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
      if len(inputs) >= MAX_SAMPLES:
        return inputs, outputs
  return inputs, outputs


def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)



# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

#   def __init__(self, d_model, warmup_steps=4000):
#     super(CustomSchedule, self).__init__()

#     self.d_model = d_model
#     # self.d_model = tf.cast(self.d_model, tf.float32)

#     self.warmup_steps = warmup_steps

#   def __call__(self, step):

#     # arg1 = tf.math.rsqrt(step + 1)
#     # arg2 = step * (self.warmup_steps**-1.5)
#     # return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

#     breakpoint()

#     step = float(step)
#     arg1 = math.sqrt(step + 1)
#     arg2 = step * (self.warmup_steps**-1.5)
#     lr = math.sqrt(self.d_model) * min(arg1, arg2)

#     print(step, arg1, arg2, lr)

#     return lr


def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence):
  prediction = evaluate(sentence)
  predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
  return predicted_sentence


def get_questions_answers():
   
    path_to_zip = tf.keras.utils.get_file(
        'cornell_movie_dialogs.zip',
        origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
        extract=True,
        cache_dir="./datasets"
    )

    path_to_dataset = os.path.join(
        os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

    path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')

    path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt')

    print(path_to_dataset)
    print(path_to_movie_lines)
    print(path_to_movie_conversations)

    questions, answers = load_conversations(
      path_to_movie_lines, path_to_movie_conversations
    )

    print('Sample question: {}'.format(questions[20]))
    print('Sample answer: {}'.format(answers[20]))

    # Build tokenizer using tfds for both questions and answers
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2**13)

    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2

    print('Sample question: {}'.format(questions[20]))
    print('Tokenized sample question: {}'.format(tokenizer.encode(questions[20])))

    # Tokenize, filter and pad sentences
    def tokenize_and_filter(inputs, outputs):
        tokenized_inputs, tokenized_outputs = [], []
        
        for (sentence1, sentence2) in zip(inputs, outputs):
            # tokenize sentence
            sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
            # check tokenized sentence max length
            if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)
        
        # pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
        
        return tokenized_inputs, tokenized_outputs

    questions, answers = tokenize_and_filter(questions, answers)

    return questions, answers, tokenizer, VOCAB_SIZE, START_TOKEN, END_TOKEN



def custom_learning_rate_scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

custom_learning_rate = tf.keras.callbacks.LearningRateScheduler(
   custom_learning_rate_scheduler
)


###############################################################################



if __name__ == "__main__":


    # sample_learning_rate = CustomSchedule(d_model=128)
    # plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
    # plt.ylabel("Learning Rate")
    # plt.xlabel("Train Step")
    # plt.savefig("sample_learning_rate.png")


    questions, answers, tokenizer, VOCAB_SIZE, START_TOKEN, END_TOKEN = get_questions_answers()


    print('Vocab size: {}'.format(VOCAB_SIZE))
    print('Number of samples: {}'.format(len(questions)))


    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1]
        },
        {
            'outputs': answers[:, 1:]
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print(dataset)

    tf.keras.backend.clear_session()

    # Hyper-parameters
    NUM_LAYERS = 2
    D_MODEL = 256
    NUM_HEADS = 8
    UNITS = 512
    DROPOUT = 0.1

    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    # learning_rate = CustomSchedule(0.01)
    # learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(
       beta_1=0.9, beta_2=0.98, epsilon=1e-9,
    )

    def accuracy(y_true, y_pred):
        # ensure labels have shape (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    model.fit(dataset, epochs=EPOCHS, callbacks=[custom_learning_rate])


    def evaluate(sentence):
        sentence = preprocess_sentence(sentence)

        sentence = tf.expand_dims(
            START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

        output = tf.expand_dims(START_TOKEN, 0)

        for i in range(MAX_LENGTH):
            predictions = model(inputs=[sentence, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, END_TOKEN[0]):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)


    def predict(sentence):
        prediction = evaluate(sentence)

        predicted_sentence = tokenizer.decode(
            [i for i in prediction if i < tokenizer.vocab_size])

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

        return predicted_sentence
    

    output = predict('Where have you been?')

    # feed the model with its previous output
    sentence = 'I am not crazy, my mother had me tested.'
    for _ in range(5):
        sentence = predict(sentence)
        print('')

    breakpoint()