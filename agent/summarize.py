summary_prompt = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: You are asked to summarize the following conversation between an AI and a human. 
The summary should only include facts from the conversation and should not add anything else. 
The summary should be concise and no more than a few sentences long. It should tidy up & formalise the language
used in the conversation, and should be a complete summary of the conversation.

### Response:
### Example 1:
Conversation:
- User: Hello, I am John. I am a data scientist and I work for a big tech company.
- AI: Hello John, nice to meet you. How can I assist you in your data science work?

Summary:
The user, John, introduced himself as a data scientist working for a big tech company and asked the AI for assistance with his work.

### Example 2:
Conversation:
- User: Can you tell me what the weather is like in New York today?
- AI: I'm sorry, I don't have real-time capabilities to provide current weather updates.

Summary:
The user asked the AI for the current weather in New York, but the AI responded that it does not have the ability to provide real-time weather updates.

### Example 3:
### Conversation: {}

Summary:

"""

def summarize(history, llm):
    # Format the history into a string
    history_str = "\n".join(f"{role}: {text}" for role, text in history)
    # Update the summary prompt with the history
    prompt = summary_prompt.format(history_str)
    # Initialize the summary
    summary = ""
    # Generate the summary
    for token in llm(prompt):
        summary += token
    return summary
