import time
import streamlit as st
# import SessionState
import numpy as np
from predictor import Predictor
# from dataset.add_noise import SynthesizeData

# state = SessionState.get(text_correct="", input="", noise="")
# import nltk

def main():
    model = load_model()
    st.title("Sửa lỗi chính tả tiếng Việt")
    # Load model
    # state.input = ""
    # state.noise = ""
    text_input = st.text_area("Nhập đầu vào:")
    text_input = text_input.strip()
    if st.button("Correct"):
        # state.noise = text_input
        # state.text_correct = model.spelling_correct(state.noise)
        text_correct = model.spelling_correct(text_input)
        st.text("Câu nhiễu: ")
        st.success(text_input)
        st.text("Kết quả:")
        st.success(text_correct)


    # if st.button("Add noise and Correct"):
    #     noise = synther.add_noise(text_input, percent_err=0.3)
    #     # state.output = noise_text
    #     text_correct = model.spelling_correct(noise)
    #     st.text("Câu nhiễu: ")
    #     st.success(noise)
    #     st.text("Kết quả:")
    #     st.success(text_correct)

# @st.cache(allow_output_mutation=True)  # hash_func
def load_model():
    print("Loading model ...")
    # nltk.download('punkt')
    model = Predictor(weight_path='weights/seq2seq_luong_1000_baomoi.pth', have_att=True)
    # synther = SynthesizeData()
    return model


if __name__ == "__main__":
    main()