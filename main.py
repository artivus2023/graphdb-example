from ctransformers import AutoModelForCausalLM

from agent.chat import respond_to, history
from agent.describe_world_state import describe_world_state
from agent.summarize import summarize

from utils.triplet_extractor import extract_triplets

# Initialize the model
llm = AutoModelForCausalLM.from_pretrained(
    'models/nous-hermes-llama-2-7b.ggmlv3.q2_K.bin',
    max_new_tokens=2048,
    model_type='llama',
    stream=True,
    temperature=0.1,
    stop=['\n', 'User: ', '### Example ']
)

# Main REPL
def store_word_state(extracted_triplets):

    pass


def main(history):
    summary = ""
    while True:
        user_input = input("User: ")

        # Extract graph from summary
        # Get normalised description of input text
        world_state_description = describe_world_state(user_input, llm)
        prefix = "Speaker: User, Addressed to: AI, Content: "
        print(f"World state description: {prefix + world_state_description}")
        # We need to use the tokenizer manually since we need special tokens.
        extracted_triplets = extract_triplets(world_state_description)
        store_word_state(extracted_triplets)
        # print(extracted_triplets)
        for triplet in extracted_triplets:
            print(triplet)
        history = respond_to(user_input, history, llm)
        summary = summarize(history, llm)
        print(f"Summary: {summary}")

test_text = """Intel Corporation is an American multinational corporation and technology company headquartered in Santa Clara, California. It is one of the world's largest semiconductor chip manufacturer by revenue, and is one of the developers of the x86 series of instruction sets found in most personal computers."""
if __name__ == '__main__':
    main(history)
    # extracted_text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(
    #     test_text, return_tensors=True, return_text=False)[0]["generated_token_ids"]]
    #                                                           )
    # extracted_triplets = extract_triplets(extracted_text[0])


