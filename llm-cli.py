import langchain as lc
import os

# Initialize the models
gpt3 = lc.GPT3(os.getenv('OPENAI_KEY'))
gpt4 = lc.GPT4(os.getenv('OPENAI_KEY'))
gemini = lc.Gemini(os.getenv('GOOGLE_AI_KEY'))

# Set the initial model to GPT-3
current_model = gpt3
prompt = 'gpt3> '

# Initialize the chat history
chat_history = []

while True:
    # Get the user's input
    user_input = input(prompt)

    # Check if the user wants to switch models
    if user_input.startswith('use '):
        model_name = user_input.split(' ')[1]
        if model_name == 'gpt3':
            current_model = gpt3
            prompt = 'gpt3> '
        elif model_name == 'gpt4':
            current_model = gpt4
            prompt = 'gpt4> '
        elif model_name == 'gemini':
            current_model = gemini
            prompt = 'gemini> '
        else:
            print('Unknown model:', model_name)
    else:
        # Add the user's input to the chat history
        chat_history.append(('user', user_input))

        # Generate a response using the current model
        response = current_model.generate(user_input)

        # Add the model's response to the chat history
        chat_history.append((prompt[:-2], response))

        # Print the model's response
        print(prompt, response)