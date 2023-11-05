
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey


st.title('🦜🔗 AutoGPT Blog generator')
prompt = st.text_input('Plug in your prompt here')

title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a blog title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write me a blog based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)


llm = OpenAI(temperature=0.5)
title_chain = LLMChain(llm=llm, prompt=title_template,
                       verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template,
                        verbose=True, output_key='script')

wiki = WikipediaAPIWrapper()

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)
