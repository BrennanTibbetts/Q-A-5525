from transformers import pipeline
import langchain.schema.document as d
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import numpy

#intialize the embeddings model used to create embeddings
def initialize_embeddings_model():
    # Define the path to the pre-trained model you want to use
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device': 'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings_model = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    return embeddings_model


#Calculate the similarity score between two sentences/phrases/strings
def calculate_similarity(embeddings_model, s1, s2):

    #Set up text embedding model:
    s1_emb = embeddings_model.embed_query(s1)
    s2_emb = embeddings_model.embed_query(s2)
    sim_score = numpy.dot(s1_emb, s2_emb)/(numpy.linalg.norm(s1_emb)*numpy.linalg.norm(s2_emb))
    return sim_score

#Return a similarity score between our generated answer and an answer generated from a pretrained model
def evaluate_answer(context, question, answer):

    #Find expected answer using pretrained question answering model dynamic tinybert from Hugging Face
    pipe = pipeline("question-answering", model="Intel/dynamic_tinybert")
    expected_result = pipe(question, context)
    print("Expected Result:", expected_result)
    expected_answer = expected_result['answer']
    expected_answer_confidence = expected_result['score']

    #Find the similarity score between our generated answer and an expected answer
    embeddings_model = initialize_embeddings_model()
    similarity_score = calculate_similarity(embeddings_model, expected_answer, answer)
    
    #We could possibly multiple the similarity score by the expected answer confidence to reflect the lower confidence in the "correct" answer
    return similarity_score



#testing -- checking similarity scores

# pairs = [
#     ("42", "How many points did the Panthers defense surrender?", "34", "The Panthers defense gave up just 308 points, ranking sixth in the league, while also leading the NFL in interceptions with 24 and boasting four Pro Bowl selections. Pro Bowl defensive tackle Kawann Short led the team in sacks with 11, while also forcing three fumbles and recovering two. Fellow lineman Mario Addison added 6\u00bd sacks. The Panthers line also featured veteran defensive end Jared Allen, a 5-time pro bowler who was the NFL's active career sack leader with 136, along with defensive end Kony Ealy, who had 5 sacks in just 9 starts. Behind them, two of the Panthers three starting linebackers were also selected to play in the Pro Bowl: Thomas Davis and Luke Kuechly. Davis compiled 5\u00bd sacks, four forced fumbles, and four interceptions, while Kuechly led the team in tackles (118) forced two fumbles, and intercepted four passes of his own. Carolina's secondary featured Pro Bowl safety Kurt Coleman, who led the team with a career high seven interceptions, while also racking up 88 tackles and Pro Bowl cornerback Josh Norman, who developed into a shutdown corner during the season and had four interceptions, two of which were returned for touchdowns."),
#     ("Kawann", "Who registered the most sacks on the team this season?", "Kawann Short", "The Panthers defense gave up just 308 points, ranking sixth in the league, while also leading the NFL in interceptions with 24 and boasting four Pro Bowl selections. Pro Bowl defensive tackle Kawann Short led the team in sacks with 11, while also forcing three fumbles and recovering two. Fellow lineman Mario Addison added 6\u00bd sacks. The Panthers line also featured veteran defensive end Jared Allen, a 5-time pro bowler who was the NFL's active career sack leader with 136, along with defensive end Kony Ealy, who had 5 sacks in just 9 starts. Behind them, two of the Panthers three starting linebackers were also selected to play in the Pro Bowl: Thomas Davis and Luke Kuechly. Davis compiled 5\u00bd sacks, four forced fumbles, and four interceptions, while Kuechly led the team in tackles (118) forced two fumbles, and intercepted four passes of his own. Carolina's secondary featured Pro Bowl safety Kurt Coleman, who led the team with a career high seven interceptions, while also racking up 88 tackles and Pro Bowl cornerback Josh Norman, who developed into a shutdown corner during the season and had four interceptions, two of which were returned for touchdowns.")
# ]
# for our_answer, q, a, context in pairs:
#     sim_score = evaluate_answer(context, q, our_answer)
#     print(sim_score)







#Testing -- siilarity scores
# pairs_correct = [
#     ("What is the capital of Italy?", "Rome is the capital of Italy."),
#     ("Who wrote 'Romeo and Juliet'?", "William Shakespeare wrote 'Romeo and Juliet'."),
#     ("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius."),
#     ("What is the capital of France?", "The Eiffel Tower is in France."),
#     ("What is the population of Australia?", "Kangaroos are native to Australia."),
#     ("Who discovered gravity?", "The earth revolves around the sun.")
# ]

# #Incorrect pairs
# pairs_incorrect = [
#     ("What is the capital of Italy?", "Rome."),
#     ("Who wrote 'Romeo and Juliet'?", "William Shakespeare."),
#     ("What is the boiling point of water?", "100 degrees Celsius."),
#     ("What is the capital of France?", "The Eiffel Tower is in France."),
#     ("What is the population of Australia?", "Kangaroos are native to Australia."),
#     ("Who discovered gravity?", "The earth revolves around the sun.")
#]

# for q,a  in pairs_correct:
#     similarity = calculate_similarity(embeddings_model, q, a)
#     print(f"Question: {q} Answer: {a} \nSimilarity Score: {similarity}\n\n")
