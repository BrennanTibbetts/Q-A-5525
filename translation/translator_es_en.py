import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Translator:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = 'Helsinki-NLP/opus-mt-es-en'

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        ).to(device)

    def translate(self, text: str) -> str:

        tokenized_text = self.tokenizer(text, return_tensors="pt")
        translated_tokens = self.model.generate(**tokenized_text, max_new_tokens=1600)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        return translated_text

# def translate(text):

#     model_name = 'Helsinki-NLP/opus-mt-en-es'
#     model = MarianMTModel.from_pretrained(model_name)
#     tokenizer = MarianTokenizer.from_pretrained(model_name)

#     text = tokenizer(text, return_tensors="pt")

#     translated = model.generate(**text, max_new_tokens=512)

#     translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

#     return translated_text


# def translateT5(text):
    
#     model_name = 't5-small'
#     model = T5ForConditionalGeneration.from_pretrained(model_name)
#     tokenizer = T5Tokenizer.from_pretrained(model_name)

#     text = tokenizer("translate English to French: " +text, return_tensors="pt")

#     translated = model.generate(**text, max_length=1024)

#     translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

#     return translated_text

# def translateMT5(text):

#     model_name = "google/mt5-base"
#     tokenizer = T5Tokenizer.from_pretrained(model_name)
#     model = MT5ForConditionalGeneration.from_pretrained(model_name)

#     inputs = tokenizer.encode("translate English to Spanish: " + text, return_tensors="pt", padding=True)
#     outputs = model.generate(inputs, max_length=400, num_beams=2, early_stopping=True)

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# def translateTunedMT5(text):

#     model_name = "HURIDOCS/mt5-small-spanish-es"
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
#     input_ids = tokenizer(
#         ["translate Spanish to English: " + text],
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=512
#     )["input_ids"]
    
#     output_ids = model.generate(
#         input_ids=input_ids,
#         max_length=400,
#         no_repeat_ngram_size=2,
#         num_beams=4
#     )[0]
    
#     result_text = tokenizer.decode(
#         output_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )

#     return result_text

Translator = Translator()
print(Translator.translate("Todo depende de lo que quieras. Puede confiar en que estaremos a su lado en las buenas y en las malas, hasta el amargo final. Y puede confiar en nosotros para guardar cualquier secreto suyo, más cerca que usted mismo. Pero no puedes confiar en que te dejaremos afrontar los problemas solo y marcharte sin decir una palabra. Somos tus amigos, Frodo. En fin: ahí está. Sabemos la mayor parte de lo que Gandalf les ha contado. Sabemos mucho sobre el Anillo. Tenemos un miedo terrible, pero iremos con vosotros; o seguirte como perros de caza."))


