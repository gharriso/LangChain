import streamlit as st

# Create two columns
col1, col2 = st.columns(2)

# Add a radio button to each column and store the selected option
option1 = col1.radio('Radio button 1', ['Option 1', 'Option 2'])
option2 = col2.radio('Radio button 2', ['Option 1', 'Option 2'])

# Add a text entry box below the buttons and store the entered text
text = st.text_input('Enter some text')

# Add a button
button = st.button('Display text and selected options')

# Display the selected options and the entered text when the button is clicked
if button:
    st.write(f'Selected option in radio button 1: {option1}')
    st.write(f'Selected option in radio button 2: {option2}')
    st.write(f'Entered text: {text}')