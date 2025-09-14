import requests
import streamlit as st

st.title("Кредитная карта Premium")
st.write("Новая кредитная карта с мгновенным одобрением")

with st.form("Подать заявку"):
    age = st.number_input("Ваш возраст: ", min_value=18, max_value=65)
    income = st.number_input("Ваш доход в тысячах рублей: ", min_value=0)
    education = st.checkbox("У меня есть высшее образование")
    work = st.checkbox("У меня есть стабильная работа")
    car = st.checkbox("У меня есть автомобиль")
    submit = st.form_submit_button("Подать заявку")

if submit:
    data = {"age": age, "income": income, "education": education, "work": work, "car": car}
    # st.write(data) # можно вывести отправленный json
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    if response.json()["approved"]:
        st.success("Поздравляем, ваша заявка одобрена!")
    else:
        st.success("Подобрали для вас альтернативу - дебетовая карта с 3% cashback")
