import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set up Streamlit UI
st.header("Friendly LLM Chat App")

# Load model and tokenizer only once
@st.cache_resource
def get_model_and_tokenizer():
    chosen_model = "microsoft/DialoGPT-small"
    tokenizer_instance = AutoTokenizer.from_pretrained(chosen_model)
    model_instance = AutoModelForCausalLM.from_pretrained(chosen_model)
    return tokenizer_instance, model_instance

tokenizer, model = get_model_and_tokenizer()

# Keep track of conversation
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for speaker, line in st.session_state.history:
    if speaker == "user":
        st.write(f"👤 **You:** {line}")
    else:
        st.write(f"🤖 **AI:** {line}")

# Get user input
prompt = st.text_input("Say something to the chatbot:")

if prompt:
    # Add the user's message
    st.session_state.history.append(("user", prompt))

    # Build context from all previous exchanges
    context = ""
    for who, msg in st.session_state.history:
        context += msg + tokenizer.eos_token

    # Generate AI response
    input_ids = tokenizer.encode(context, return_tensors="pt")
    response_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )
    # Extract new part of the response
    answer = tokenizer.decode(
        response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True
    )

    # Add AI response to the chat
    st.session_state.history.append(("ai", answer))
    st.write(f"🤖 **AI:** {answer}")

# Clear conversation
if st.button("Reset Conversation"):
    st.session_state.history = []
    st.experimental_rerun()
