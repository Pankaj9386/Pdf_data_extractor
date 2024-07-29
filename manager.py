import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdfplumber as pp
import re
import io
import google.generativeai as genai
from PIL import Image
import requests
import json
API_KEY="AIzaSyCgVzE1OsxtrMr61B8CBdAdcHmTBDShOHg"
import os

def get_answer_from_gemini(data):
    
    API_KEY = "AIzaSyCgVzE1OsxtrMr61B8CBdAdcHmTBDShOHg"
    genai.configure(api_key=API_KEY)
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(history=[])
    # newresponse=model.generate_content()
    response = chat_session.send_message(data)

    return response




def scatterplot(data,selected_option1,selected_option2):
    fig, ax = plt.subplots()
    ax.scatter(data[selected_option1], data[selected_option2], color='blue', alpha=0.5, edgecolors='w', s=100)
    ax.set_xlabel(selected_option1)
    ax.set_ylabel(selected_option2)
    ax.set_title('Scatter Plot')
    st.pyplot(fig)

def bargraph(data,selected_option1,selected_option2):
    fig, ax = plt.subplots()
    ax.bar(data[selected_option1], data[selected_option2], color='blue', alpha=0.7)
    ax.set_xlabel(selected_option1)
    ax.set_ylabel(selected_option2)
    ax.set_title('Bar Graph')
    st.pyplot(fig)

def histogram(data, selected_option, bins, color):
    fig, ax = plt.subplots()
    ax.hist(data[selected_option], bins=bins, color=color, alpha=0.7)
    ax.set_xlabel(selected_option)
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram')
    st.pyplot(fig)
    
def linegraph(data,selected_option1,selected_option2):
    fig, ax = plt.subplots()
    ax.plot(data[selected_option1], data[selected_option2], color='blue', alpha=0.7)
    ax.set_xlabel(selected_option1)
    ax.set_ylabel(selected_option2)
    ax.set_title('Line Graph')
    st.pyplot(fig)
    
def piechart(data,selected_option1,selected_option2):
    fig, ax = plt.subplots()
    ax.pie(data[selected_option2], labels=data[selected_option1], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Pie Chart')
    st.pyplot(fig)

def extract_images(file):
    images=[]
    with pp.open(file)as pdf:
        for page_num,page in enumerate(pdf.pages):
            try:
                img_obj=page.to_image()
                for img_index, img in enumerate(page.images):
                    x0,top,x1,bottom=img['x0'],img['top'],img['x1'],img['bottom']
                    cropped_image=img_obj.original.crop((x0,top,x1,bottom))
                    image_bytes=io.BytesIO()
                    cropped_image.save(image_bytes,format='JPEG')
                    image_bytes=image_bytes.getvalue()
                    image_filename = f"image_{page_num+1}_{img_index+1}.jpg"
                    images.append((image_filename, image_bytes))
            except Exception as e:
                st.error(f"Error processing page {page_num + 1}: {e}")
    return images

def analyze(data):
    
    columnsl=data.columns.tolist()
    st.write("select columns to compare")
    selected_option1 = st.selectbox('Select column1:', columnsl)
    columnsl.remove(selected_option1)
    selected_option2 = st.selectbox('Select column2:', columnsl)
                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    col1, col2, col3, col4,col5,col6= st.columns(6)
    with col1:
        view_data=st.button("View data")
    with col2:
        bar_graph = st.button('Bar Graph')
    with col3:
        scatter_plot = st.button('Scatter Plot')
    with col4:
        histogram_plot = st.button('Histogram')
    with col5:
        line_graph = st.button('Line Graph')
    with col6:
        pie_chart=st.button("Pie Chart")
    
    if view_data:
        st.write(data[[selected_option1,selected_option2]])
    if bar_graph:
        bargraph(data, selected_option1, selected_option2)
    if scatter_plot:
        scatterplot(data, selected_option1, selected_option2)
    if histogram_plot:
        bins = st.slider('Select number of bins:', min_value=5, max_value=30, value=5)
        st.subheader('Select color for the histogram:')
        color = st.color_picker('Pick a color', '#FFFF00')
        histogram(data, selected_option1, bins, color)
    if line_graph:
        linegraph(data, selected_option1, selected_option2)
    if pie_chart:
        piechart(data,selected_option1,selected_option2)

def extract_text_by_replacing_images(file):
    images=[]
    with pp.open(file)as pdf:
        extracted_text=""
        for page_num,page in enumerate(pdf.pages):
            text=page.extract_text()
            
                # extracted_text+=text
                
            images=page.images
                
                
            
            img_obj=page.to_image()
            for img_index, img in enumerate(page.images):
                # x0,top,x1,bottom=img['x0'],img['top'],img['x1'],img['bottom']
                    
                text += f"image_{page_num+1}_{img_index+1}.jpg"
            extracted_text+=text 
            extracted_text+=f"  \npage number-{page_num+1}  \n"   
            
    return extracted_text


    
def extract_tables(uploaded_file):
    with pp.open(uploaded_file)as f:
        # for page in f.pages:
        table_list=[]
        for page_index, page in enumerate(f.pages):
            tables=page.extract_table()
            
                
            if tables:
                df=pd.DataFrame(tables)
                # st.write(df)
                # st.write(f"Page number: {page_index+1}")
                table_list.append(df)
                previous_page=page_index-1
                table_text=""
                
                for i,page in enumerate(f.pages):
                    if i>=previous_page and i<=previous_page+1:
                        table_text+=page.extract_text()
                    if i>previous_page+1:
                        break
                    
                question=f"""i have given a table and the text. 
                        Provide the topic and the content related 
                        to the given table from the provided data.
                        
                        table:{tables}
                        
                        text:{table_text}
                        
                        """
            
                # question_list = {
                #     "table": df,
                #        "content": table_text,
                #        "prompt":question
                #         }
                
                json_table = df.to_json(orient="records")

                
                output = {
                    "text": table_text,
                    "table": json.loads(json_table), 
                   
                }

                # Convert the entire dictionary to JSON
                question_json = json.dumps(output, indent=4)
                
                response=get_answer_from_gemini(question)
                response=response.text
                # st.write(response)
            
            
                table_list.append(f"Page number: {page_index+1}\n\n{response}")
                # table_list.append(f"Page number: {page_index+1}")
                # table_list[f"Page number: {page_index+1}"]=df
                # all_tables.append(df)
                # new_tables=pd.DataFrame(all_tables)
        for i in range(len(table_list)):
            if i%2==0:
                df=pd.DataFrame(table_list[i])
                st.write(df)
            else:
                st.write(table_list[i])
        # st.write(table_list)       
        # if new_tables:
        #     # final_df=pd.concat(all_tables,ignore_index=True)
            
        #     st.write("all tables from pdf")
        #     st.write(new_tables)
        # else:
        #     st.error("no tables found")


def extract_textpdf(uploaded_file):
    text1=""
    with pp.open(uploaded_file)as f:
        for page in f.pages:
            text1+=page.extract_text()
        
    return text1

 
def get_answer(data):
    API_KEY = "AIzaSyCgVzE1OsxtrMr61B8CBdAdcHmTBDShOHg"
    genai.configure(api_key=API_KEY)
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(history=[])
    # newresponse=model.generate_content()
    response = chat_session.send_message(data)

    return response








st.title("file analyzer")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "pdf"], accept_multiple_files=False)

# data=pd.read_csv(uploaded_file)

# st.print(data)
#gemini_api_key ="AIzaSyBGqyXPQjQjPJsAKPLgJGqr-8l9LoiPAqE"
if uploaded_file is not None:
    all_tables = []
    if uploaded_file.name.endswith('.csv'):
        data=pd.read_csv(uploaded_file)
        
        # st.write(data)
        analyze(data)
    
    elif uploaded_file.name.endswith('.pdf'):
        
        column1,column2,column3,column4=st.columns(4)
        with column1:
            ex_image=st.button("Images")
        with column2:
            ex_tables=st.button("Tables")
        with column3:
            ex_question=st.button("Ask Question")
        with column4:
            image_labels=st.button("Text with image labels")
            
        if image_labels:
            text=extract_text_by_replacing_images(uploaded_file)
            st.write(text)
            
        if ex_question:
            chat=st.chat_input("Type your message here...")
            text=extract_textpdf(uploaded_file)
            
            
        
        if ex_image:
            text=extract_textpdf(uploaded_file)
            images=extract_images(uploaded_file)
            
            if images:
                st.write("Extracted Images:")
                for image_filename, image_bytes in images:
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption=image_filename)
                    st.download_button(
                        label="Download Image",
                        data=image_bytes,
                        file_name=image_filename,
                        mime="image/jpeg"
                    )
            else:
                st.write("No images found in the PDF file.")
        
        
        if ex_tables:
            #   text=extract_textpdf(uploaded_file)
              tables=extract_tables(uploaded_file) 
        
           
        
        
        
    