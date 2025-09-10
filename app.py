import streamlit as st
from agent import CrewAgent
from p_caching import clear_cache, init_cache_dir

if 'cache_initialized' not in st.session_state:
    init_cache_dir()
    clear_cache()
    st.session_state['cache_initialized'] = True

st.title('Gia sư hỗ trợ học tập')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'agent' not in st.session_state:
    st.session_state['agent'] = CrewAgent()

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input('Tôi có thể giúp gì cho bạn?'):
    st.session_state['messages'].append(
        {
            'role': 'user',
            'content': prompt,
        }
    )

    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        response = st.session_state['agent'].work(prompt)
        st.markdown(response)

    st.session_state['messages'].append(
        {
            'role': 'assistant',
            'content': response
        }
    )


