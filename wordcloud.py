import streamlit as st
import pandas as pd
import file_checkpoint as fc

placeholder = st.empty()
init_df = True

if not fc.checkpoint.CheckDataframe():
    init_df = False
    with placeholder.container():
        st.write("Dataframe not initialized")

if init_df:
    with placeholder.container():
        st.write("Dataframe initialized")
        st.write(fc.checkpoint.GetDataframe())