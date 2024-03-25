from transformers import MarianMTModel, MarianTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

def translate(text):

    model_name = 'Helsinki-NLP/opus-mt-en-es'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    text = tokenizer(text, return_tensors="pt")

    translated = model.generate(**text, max_new_tokens=512)

    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated_text


def translateT5(text):
    
    model_name = 't5-small'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    text = tokenizer("translate English to French: " +text, return_tensors="pt")

    translated = model.generate(**text, max_length=1024)

    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated_text

print(translate("It all depends on what you want. You can trust us to stick to you through thick and thin – to the bitter end. And you can trust us to keep any secret of yours – closer than you keep it yourself. But you cannot trust us to let you face trouble alone, and go off without a word. We are your friends, Frodo. Anyway: there it is. We know most of what Gandalf has told you. We know a good deal about the Ring. We are horribly afraid – but we are coming with you; or following you like hounds."))
