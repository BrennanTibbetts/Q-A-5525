from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class QuestionGenerator:
    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "voidful/context-only-question-generator"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "voidful/context-only-question-generator"
        ).to(device)

    def generate_questions(
        self, context: str, num_questions: int = 4, temperature: float = 1.0
    ):
        """
        Generate questions from a given context using the model
        """

        input_ids = self.tokenizer(context, return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=50,
            num_return_sequences=num_questions,
            temperature=temperature,
            do_sample=True,
        )
        questions = list(
            {
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            }
        )
        return questions
