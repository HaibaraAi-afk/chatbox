import os
import argparse
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/chatbot_data.csv')
    parser.add_argument('--model_path', type=str, default='model/chatbot_model.h5')
    return parser.parse_args()

#Load and Preprocess Data
def load_data(data_dir):
    data_path = os.path.join(data_dir, './chatbot_data.jsonl')
    df = pd.read_json(data_path, lines=True)
    
    #split data
    X = df["input"].values
    y = df["output"].values
    
    return X, y

def preprocess_data(X, y, vocab_size=5000, max_length=50):
    #Tokenizer for input data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(X)
    X_sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')
    
    #Encode output data
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)
    
    return X_padded, y_categorical, tokenizer, label_encoder

#Build Model
def build_model(vocab_size, embedding_dim=128, max_length=50, num_classes=10):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )
    
    return model
# Save model to GCS
def saved_model_to_gcs(model, model_dir):
    local_model_path = os.path.join(model_dir, 'chatbot_model.h5')
    model.save(local_model_path)
    os.system(f'gsutil -m cp -r {local_model_path} gs://chatbot-models')


#Main
def main():
    args = parse_args()
    
    #Load Data
    X, y = load_data(args.data_path)
    
    #Preprocess Data
    X_padded, y_categorical, tokenizer, label_encoder = preprocess_data(X, y)
    
    #Split Data
    X_train, X_val, y_train, y_val = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)    
    #Build Model
    model = build_model(vocab_size=5000, num_classes=y_categorical.shape[1])
    
    #Train Model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)
    
    #Save Model to GCS
    saved_model_to_gcs(model, args.model_dir)
    
if __name__ == '__main__':
    main()
    
        
        