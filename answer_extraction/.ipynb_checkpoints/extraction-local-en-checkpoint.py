# ripped this off of https://medium.com/@rukaiya.rk24/bridging-linguistic-barriers-inside-googles-mt5-multilingual-technology-4a85e6ca056f

import tensorflow as tf
from datasets import load_dataset
from transformers import (
    TFMT5ForConditionalGeneration,
    MT5Tokenizer,
    DataCollatorForSeq2Seq,
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = TFMT5ForConditionalGeneration.from_pretrained("google/mt5-small")

dataset = load_dataset("csv", data_files="fine-tune.csv")
dataset = dataset["train"].shuffle(seed=42)

# create a 70 - 15 - 15 train, test, validation split

train_dataset, test_dataset = train_test_split(dataset, test_size=0.7)
validate_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5)


def preprocess_function(examples):
    padding = "max_length"
    max_length = 200

    inputs = [
        ex for ex in examples["Input"]
    ]  # Replace input by the text column in your data
    targets = [
        ex for ex in examples["Output"]
    ]  # Replace output by the labels of your data
    model_inputs = tokenizer(
        inputs, max_length=max_length, padding=padding, truncation=True
    )
    labels = tokenizer(targets, max_length=max_length, padding=padding, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id,
    pad_to_multiple_of=64,
    return_tensors="np",
)

tf_train_dataset = model.prepare_tf_dataset(
    train_dataset, collate_fn=data_collator, batch_size=8, shuffle=True
)

model.compile(optimizer=Adam(3e-5))
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
model.fit(tf_train_dataset, epochs=10, callbacks=[early_stopping])
