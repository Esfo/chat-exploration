import numpy as np
import matplotlib.pyplot as plt

#one tiny language-model training loop: words become ids, ids become vectors,
#one attention layer mixes prior words, and the model predicts the next word.
#edit only the paragraph if you want to change the training text.

paragraph = """
The cat sat because it was tired
Penguins slide down icy ramps whenever they feel bored
Dogs run when they're happy
The bird slept because it was quiet
Fish never swim when waters are calm
"""

paragraph_lines = [line.strip().lower() for line in paragraph.strip().splitlines() if line.strip()]
line_words = [line.split() for line in paragraph_lines]

tokens = [word for words in line_words for word in words]
vocab = sorted(set(tokens))
word_to_id = {word: word_id for word_id, word in enumerate(vocab)}
id_to_word = {word_id: word for word, word_id in word_to_id.items()}

line_token_ids = [
    np.array([word_to_id[word] for word in words])
    for words in line_words
]

context_len = 10             #how many prior words the model can see
vocab_size = len(vocab)
model_vector_size = 24      #size of each word vector inside the model
attention_vector_size = 24  #size of query/key/value vectors
learning_rate = 0.05
max_rounds = 5000
batch_size = 16

rng = np.random.default_rng(0)

training_examples = []

for line_index, token_ids in enumerate(line_token_ids):
    for target_position in range(1, len(token_ids)):
        training_examples.append((line_index, target_position))

if len(training_examples) == 0:
    raise ValueError("paragraph needs at least one line with at least two words")

def softmax(values):
    values = values - values.max()
    exponentials = np.exp(values)
    return exponentials / exponentials.sum()

#trainable weights
#embedding_weights turns one-hot word ids into model vectors
#query_weights makes the current word's query vector
#key_weights makes each prior word's key vector
#value_weights makes each prior word's value vector
#attention_output_weights writes the attention result back into model-vector space
#output_token_weights turns the final model vector into raw next-word scores
embedding_weights = rng.normal(0, 1 / np.sqrt(vocab_size), (vocab_size, model_vector_size))
query_weights = rng.normal(0, 1 / np.sqrt(model_vector_size), (model_vector_size, attention_vector_size))
key_weights = rng.normal(0, 1 / np.sqrt(model_vector_size), (model_vector_size, attention_vector_size))
value_weights = rng.normal(0, 1 / np.sqrt(model_vector_size), (model_vector_size, attention_vector_size))
attention_output_weights = rng.normal(0, 1 / np.sqrt(attention_vector_size), (attention_vector_size, model_vector_size))
output_token_weights = rng.normal(0, 1 / np.sqrt(model_vector_size), (model_vector_size, vocab_size))

losses, accuracies = [], []
rounds = 0
streak = 0

while True:
    #gradient_for_ means "how this weight should change to reduce the loss"
    #these start at zero each round and accumulate gradients across the batch
    gradient_for_embedding_weights = np.zeros_like(embedding_weights)
    gradient_for_query_weights = np.zeros_like(query_weights)
    gradient_for_key_weights = np.zeros_like(key_weights)
    gradient_for_value_weights = np.zeros_like(value_weights)
    gradient_for_attention_output_weights = np.zeros_like(attention_output_weights)
    gradient_for_output_token_weights = np.zeros_like(output_token_weights)

    batch_loss = 0.0
    batch_correct = 0

    for _ in range(batch_size):
        line_index, target_position = training_examples[rng.integers(0, len(training_examples))]
        token_ids = line_token_ids[line_index]

        context_start = max(0, target_position - context_len)
        training_context_ids = token_ids[context_start:target_position]
        target_word_id = token_ids[target_position]
        context_length = len(training_context_ids)

        #make one-hot rows for the words in this training context
        context_word_one_hots = np.zeros((context_length, vocab_size))
        context_word_one_hots[np.arange(context_length), training_context_ids] = 1

        #forward: word ids -> word vectors -> attention -> next-word prediction
        context_word_vectors = context_word_one_hots @ embedding_weights

        query_vector = context_word_vectors[-1] @ query_weights
        key_vectors = context_word_vectors @ key_weights
        value_vectors = context_word_vectors @ value_weights

        relevance_values = key_vectors @ query_vector / np.sqrt(attention_vector_size)
        attention_probabilities = softmax(relevance_values)

        mixed_value_vector = attention_probabilities @ value_vectors
        attention_layer_vector = mixed_value_vector @ attention_output_weights

        #logits are raw next-word scores before softmax turns them into probabilities
        logits = attention_layer_vector @ output_token_weights
        next_word_probabilities = softmax(logits)

        batch_loss += -np.log(next_word_probabilities[target_word_id] + 1e-12)
        batch_correct += int(next_word_probabilities.argmax() == target_word_id)

        #backward: each gradient says how the loss changes when that value changes
        gradient_for_logits = next_word_probabilities.copy()
        gradient_for_logits[target_word_id] -= 1

        gradient_for_output_token_weights += np.outer(attention_layer_vector, gradient_for_logits)
        gradient_for_attention_layer_vector = output_token_weights @ gradient_for_logits

        gradient_for_attention_output_weights += np.outer(mixed_value_vector, gradient_for_attention_layer_vector)
        gradient_for_mixed_value_vector = attention_output_weights @ gradient_for_attention_layer_vector

        gradient_for_attention_probabilities = value_vectors @ gradient_for_mixed_value_vector
        gradient_for_value_vectors = np.outer(attention_probabilities, gradient_for_mixed_value_vector)

        gradient_for_relevance_values = attention_probabilities * (
            gradient_for_attention_probabilities
            - gradient_for_attention_probabilities @ attention_probabilities
        )
        gradient_for_relevance_values /= np.sqrt(attention_vector_size)

        gradient_for_key_vectors = np.outer(gradient_for_relevance_values, query_vector)
        gradient_for_query_vector = key_vectors.T @ gradient_for_relevance_values

        gradient_for_value_weights += context_word_vectors.T @ gradient_for_value_vectors
        gradient_for_key_weights += context_word_vectors.T @ gradient_for_key_vectors
        gradient_for_query_weights += np.outer(context_word_vectors[-1], gradient_for_query_vector)

        gradient_for_context_word_vectors = gradient_for_value_vectors @ value_weights.T
        gradient_for_context_word_vectors += gradient_for_key_vectors @ key_weights.T
        gradient_for_context_word_vectors[-1] += gradient_for_query_vector @ query_weights.T

        gradient_for_embedding_weights += context_word_one_hots.T @ gradient_for_context_word_vectors

    #gradient descent: update every trainable weight slightly
    embedding_weights -= learning_rate * gradient_for_embedding_weights / batch_size
    query_weights -= learning_rate * gradient_for_query_weights / batch_size
    key_weights -= learning_rate * gradient_for_key_weights / batch_size
    value_weights -= learning_rate * gradient_for_value_weights / batch_size
    attention_output_weights -= learning_rate * gradient_for_attention_output_weights / batch_size
    output_token_weights -= learning_rate * gradient_for_output_token_weights / batch_size

    losses.append(batch_loss / batch_size)
    accuracies.append(batch_correct / batch_size)

    rounds += 1
    streak = streak + 1 if accuracies[-1] == 1.0 else 0

    if streak >= 200:
        break
    if rounds >= max_rounds:
        break

print("solved" if streak >= 200 else "gave up", "after", rounds, "rounds")

#evaluate every next-word prediction from every line
eval_loss = 0.0
eval_correct = 0

print()
print("line predictions")

for line_index, token_ids in enumerate(line_token_ids):
    print()
    print("line:", paragraph_lines[line_index])

    predicted_line_words = [id_to_word[token_ids[0]]]

    for target_position in range(1, len(token_ids)):
        context_start = max(0, target_position - context_len)
        training_context_ids = token_ids[context_start:target_position]
        target_word_id = token_ids[target_position]
        context_length = len(training_context_ids)

        context_word_one_hots = np.zeros((context_length, vocab_size))
        context_word_one_hots[np.arange(context_length), training_context_ids] = 1

        context_word_vectors = context_word_one_hots @ embedding_weights

        query_vector = context_word_vectors[-1] @ query_weights
        key_vectors = context_word_vectors @ key_weights
        value_vectors = context_word_vectors @ value_weights

        relevance_values = key_vectors @ query_vector / np.sqrt(attention_vector_size)
        attention_probabilities = softmax(relevance_values)

        mixed_value_vector = attention_probabilities @ value_vectors
        attention_layer_vector = mixed_value_vector @ attention_output_weights

        logits = attention_layer_vector @ output_token_weights
        next_word_probabilities = softmax(logits)

        predicted_word_id = next_word_probabilities.argmax()
        predicted_line_words.append(id_to_word[predicted_word_id])

        eval_loss += -np.log(next_word_probabilities[target_word_id] + 1e-12)
        eval_correct += int(predicted_word_id == target_word_id)

        context_words = [id_to_word[word_id] for word_id in training_context_ids]

        print("context:  ", " ".join(context_words))
        print("target:   ", id_to_word[target_word_id])
        print("predicted:", id_to_word[predicted_word_id])
        print("attention:", np.round(attention_probabilities, 3))
        print()

    print("predicted line:", " ".join(predicted_line_words))

print()
print("paragraph evaluation")
print(f"accuracy: {eval_correct / len(training_examples):.3f}")
print(f"loss:     {eval_loss / len(training_examples):.3f}")

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(losses, color="crimson", label="loss")
ax.plot(accuracies, color="green", label="accuracy")
ax.set_xlabel("round")
ax.set_title(f"trained next-word prediction in {rounds} rounds")
ax.legend()

plt.tight_layout()
plt.show()
