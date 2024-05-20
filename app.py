import streamlit as st
from streamlit_option_menu import option_menu

option_menu(
    menu_title=None,
    options=["Tab1", "Tab2", "Tab3"],
    icons=["bar-chart", "body-text", "calendar-check"],
    orientation="horizontal",
)
