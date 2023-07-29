import tqdm
from collections import deque

# Initialize the conversation history
history = deque(maxlen=10)

MAX_TOKENS = 2048

system_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: The following is a conversation between an AI and a human. The AI is going to say something to the human. Write what the AI is going to say
in response to the human. The AI should respond in a way that is appropriate to the human's statement, and encourages the conversation
to continue with follow up questions or statements

### Response:{}

AI: """



def update_history(history, role, text):
    # Add the new interaction to the history
    history.append((role, text))
    # Format the history into a string
    history_str = "\n".join(f"{role}: {text}" for role, text in history)
    return history, history_str

def respond_to(input_text, history, llm):
    # Update the history with the user's input
    history, history_str = update_history(history, "User", input_text)
    # Update system prompt with the history and user's input
    prompt = system_prompt.format(history_str)
    # Initialize the response
    response = ""
    # Generate the AI's response
    print("AI: ", end='', flush=True)  # Print the "AI: " prefix
    for token in llm(prompt):
        response += token
        print(token, end='', flush=True)
    print()  # print a newline at the end of the response
    # Update the history with the AI's response
    history, _ = update_history(history, "AI", response)
    return history

