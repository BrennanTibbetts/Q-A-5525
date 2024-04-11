from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# NOTE: If used in final, cite https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap


class QuestionGenerator:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
            )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
            ).to(device)


    def generate_question(self, answer, context, max_length=64):
        input_text = f"answer: {answer}  context: {context} </s>"
        features = self.tokenizer([input_text], return_tensors='pt')

        output = self.model.generate(input_ids=features['input_ids'], 
                    attention_mask=features['attention_mask'],
                    max_length=max_length)

        return self.tokenizer.decode(output[0])


if __name__ == '__main__':
    qg = QuestionGenerator()
    answer = "the Alps"
    context = """France, often hailed as the heart of Europe, stands as a beacon of culture, 
        history, and culinary mastery. This illustrious nation, woven into the very fabric of the
        continent's identity, offers a diverse tapestry of landscapes that range from the rugged
        cliffs of Brittany to the sun-drenched lavender fields of Provence, and from the 
        snow-capped peaks of the Alps to the serene vineyards of Bordeaux. The Alps are a 
        majestic mountain range that stretches across eight countries, including France,"""

    question = qg.generate_question(answer, 
                                    context)
    print(question)



