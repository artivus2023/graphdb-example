summary_prompt = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: Below is the input from a user. You should describe the input in a structured fashion in terms of who is speaking, who is being addressed, and what information or facts are being conveyed. This description should be objective and could be used to describe the world state implied by the user's statement.

For example:

### Example 1:
- User Input: "Hello, I am John. I am a data scientist and I work for a big tech company."
- Description: "Speaker: User (John), Role: Data scientist at a big tech company, Addressed to: AI, Content: Introduction and job role."

### Example 2:
- User Input: "Can you tell me what the weather is like in New York today?"
- Description: "Speaker: User, Addressed to: AI, Content: Request for current weather in New York."

### Example 3:
- User Input: "{}"
- Description: "Speaker: User, Addressed to: AI, Content: """

def describe_world_state(input, llm):
    # We're just storing world state here as an example,
    # but we can easily check if we have semantic knowledge of the
    # entities in the world state, update relations etc
    prompt = summary_prompt.format(input)
    description = ""
    # TODO update semantic memory, plus store conversation, messages, world state, etc in a graph
    for token in llm(prompt):
        description += token
    return description
