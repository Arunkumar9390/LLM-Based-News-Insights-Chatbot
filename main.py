import os
from constants import openai_key
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_openai import OpenAI  # ✅ use updated import
import streamlit as st

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Celebrity Search Results')
input_text = st.text_input("Search the Celebrity You Want to Know About")

# Prompt templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

# OpenAI LLM
llm = OpenAI(temperature=0.8)

# Chains
chain = LLMChain(llm=llm, prompt=first_input_prompt, output_key='person', verbose=True)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, output_key='dob', verbose=True)

third_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="Mention 5 major events happened around {dob} in the world?"
)

chain3 = LLMChain(llm=llm, prompt=third_input_prompt, output_key='description', verbose=True)

# Sequential Chain
parent_chain = SequentialChain(
    chains=[chain, chain2, chains3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description'],
    verbose=True
)

# Run the chain if input is given
if input_text:
    result = parent_chain({'name': input_text})
    st.write(result)
