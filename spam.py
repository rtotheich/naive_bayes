import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # Remove characters that do not necessarily help us determine if spam or not

    for unnormalized_char in [',', '.', '!', '?', '"', "'", '(', ')', '*', '&', '$', '@', '/']:
        message = message.replace(unnormalized_char, '')
    return message.lower().split()
    

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """
    
    # First, get the counts of each word and store in a dict
    
    counts = {}
    for message in messages:
        words = get_words(message)
        for words in set(words):
            counts[words] = counts.get(words, 0) + 1

    # Then, build a list of all the word-freq pairs with freq >= 5

    vocabulary_list = []
    for word_freq_pair in counts.items():
        word, freq = word_freq_pair
        if freq >= 5:
            vocabulary_list.append(word)

    # Finally, build another dict with the word mapping to an iterated index
    
    return {word:token_id for token_id, word in enumerate(vocabulary_list)}


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """

    # For a set of messages, build a bag-of-word matrix from zeros adding
    # up all word counts per index in the vocabulary

    bow_matrix = np.zeros((len(messages), len(word_dictionary.items())))
    for message_number, message in enumerate(messages):
        tokenized = get_words(message)
        for token in tokenized:
            token_id = word_dictionary.get(token, -1)
            # Only add counts if the word is in the vocab!
            if (token_id) != -1:
                bow_matrix[message_number][token_id] += 1
    return bow_matrix



def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # We will store all model weights in a dictionary

    weight_dict = {}

    # Initialize an array of ones for the BOW vector

    vocab_size = len(matrix[0])

    spam_counts = np.zeros(vocab_size)
    ham_counts = np.zeros(vocab_size)

    total_spam = 0
    total_ham = 0

    spam_words = 0
    ham_words = 0

    # Add up num of spam/ham words and total num of spam/ham emails

    for i, message in enumerate(matrix):
        if labels[i] == 1:
            spam_counts += message
            total_spam += 1
            spam_words += np.sum(message)
        else:
            ham_counts += message
            total_ham += 1
            ham_words += np.sum(message)

    # Take log probability and store in the weights dictionary

    num_messages = len(matrix)

    weight_dict['spam_w'] = np.log((1 + spam_counts) / (vocab_size + spam_words))
    weight_dict['ham_w'] = np.log((1 + ham_counts) / (vocab_size + ham_words))
    weight_dict['p_spam'] = np.log(total_spam / num_messages)
    weight_dict['p_ham'] = np.log(total_ham / num_messages)

    return weight_dict


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """

    # Do dot product to sum (log probas per word * message counts)

    p_spam = (matrix @ model['spam_w']) + model['p_spam']
    p_ham = (matrix @ model['ham_w']) + model['p_ham']

    # Get a True-False array with predictions

    predictions = p_spam > p_ham

    # Return a binary int array of the predictions

    return np.asarray(predictions).astype(int)


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """

    # Sort words by their token index (dictionary value) using lambda

    sorted_keys = sorted(list(dictionary.items()), key=lambda x: x[1])
    predictors = []

    # Compute the log odds (prob success over prob failure)

    odds = model['spam_w'] - model['ham_w']

    # Sort in non-increasing order to get spammiest words

    all_spam_predictors = odds.argsort()[::-1]
    i = 0

    # Iterate through, adding top 5 spammy words to a list

    while len(predictors) < 5:
        predictor_word = sorted_keys[all_spam_predictors[i]][0]
        predictors.append(predictor_word)
        i += 1

    return predictors


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)


if __name__ == "__main__":
    main()
