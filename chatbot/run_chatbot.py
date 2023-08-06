
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


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses an exponential decay schedule.

    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies an exponential decay function
    to an optimizer step, given a provided initial learning rate.

    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:

    ```python
    def decayed_learning_rate(step):
      return initial_learning_rate * decay_rate ^ (step / decay_steps)
    ```

    If the argument `staircase` is `True`, then `step / decay_steps` is
    an integer division and the decayed learning rate follows a
    staircase function.

    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate.
    Example: When fitting a Keras model, decay every 100000 steps with a base
    of 0.96:

    ```python
    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(data, labels, epochs=5)
    ```

    The learning rate schedule is also serializable and deserializable using
    `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.

    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate=0.001,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=False,
        name=None,
    ):
        """Applies exponential decay to the learning rate.

        Args:
          initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the decay computation above.
          decay_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The decay rate.
          staircase: Boolean.  If `True` decay the learning rate at discrete
            intervals
          name: String.  Optional name of the operation.  Defaults to
            'ExponentialDecay'.
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "LearningRateSchedule") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            decay_rate = tf.cast(self.decay_rate, dtype)

            global_step_recomp = tf.cast(step, dtype)
            p = global_step_recomp / decay_steps
            if self.staircase:
                p = tf.floor(p)
            return tf.multiply(
                initial_learning_rate, tf.pow(decay_rate, p), name=name
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name,
        }


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


def get_conversation_data():
   
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

    return questions, answers


def get_tokenizer(sentence_pairs):

    questions, answers = sentence_pairs

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


class Tokenizer(object):
   
    def __init__(self, questions, answers):

        # Build tokenizer using tfds for both questions and answers
        self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=2**13)

        # Define start and end token to indicate the start and end of a sentence
        self.START_TOKEN, self.END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

        # Vocabulary size plus start and end token
        self.VOCAB_SIZE = self.tokenizer.vocab_size + 2

        print('Sample question: {}'.format(questions[20]))
        print('Tokenized sample question: {}'.format(tokenizer.encode(questions[20])))

        self.questions, self.answers = self.tokenize_and_filter(questions, answers)

    # Tokenize, filter and pad sentences
    def tokenize_and_filter(self, inputs, outputs):
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



# def custom_learning_rate_scheduler(epoch, lr):
#   if epoch < 10:
#     return lr
#   else:
#     return lr * tf.math.exp(-0.1)

# custom_learning_rate = tf.keras.callbacks.LearningRateScheduler(
#    custom_learning_rate_scheduler
# )


###############################################################################



if __name__ == "__main__":


    # sample_learning_rate = LearningRateSchedule()
    # plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
    # plt.ylabel("Learning Rate")
    # plt.xlabel("Train Step")
    # plt.savefig("sample_learning_rate.png")

    sentence_pairs = get_conversation_data()

    questions, answers, tokenizer, VOCAB_SIZE, START_TOKEN, END_TOKEN = get_tokenizer(sentence_pairs)


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

    # learning_rate = CustomSchedule(D_MODEL)
    learning_rate = LearningRateSchedule()

    optimizer = tf.keras.optimizers.Adam(
       learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9,
    )

    def accuracy(y_true, y_pred):
        # ensure labels have shape (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    model.fit(dataset, epochs=EPOCHS)


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
    

    # feed the model with its previous output
    sentence = 'I am not crazy, my mother had me tested.'
    for _ in range(5):
        sentence = predict(sentence)
        print('')

    breakpoint()