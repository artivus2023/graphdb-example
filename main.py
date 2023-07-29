from ctransformers import AutoModelForCausalLM
import tqdm

llm = AutoModelForCausalLM.from_pretrained('models/nous-hermes-llama-2-7b.ggmlv3.q2_K.bin', model_type='llama')

system_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: The following is a conversation between an AI and a human. The AI is going to say something to the human. Write what the AI is going to say
in response to the human. The AI should respond in a way that is appropriate to the human's statement, and encourages the conversation
to continue with follow up questions or statements.

### Response:
User: {input_prompt}
AI: """

def respond_to(input_text):
    # Update system prompt with the user's input
    prompt = system_prompt.format(input_prompt=input_text)
    response = llm(prompt)
    return response

def main():
    while True:
        user_input = input("User: ")
        response = respond_to(user_input)
        print(f"AI: {response}")

if __name__ == '__main__':
    main()
