import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdfplumber as pp
import re
import mysql.connector
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
    extracted_text = ""
    
    with pp.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = ""
            image_positions = []
            word_position=page.extract_words()
            # st.write(f"  \n{word_position}  \n  page-number{page_num}  \n")
            # Extract images and their positions
            images = page.images
            img_obj = page.to_image()
            for img_index, img in enumerate(images):
                x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
                cropped_image = img_obj.original.crop((x0, top, x1, bottom))
                image_filename = f"image_{page_num+1}_{img_index+1}.jpg"
                image_positions.append((x0,top,x1,bottom))

                for i in range(len(word_position)):
                    for j in range(len(image_positions)):
                        if word_position[i]["top"]>image_positions[j][1] and word_position[i]["bottom"]>image_positions[j][3]:
                            image={
                                "text":f" image_{page_num+1}_{img_index+1}.jpg ",
                                "top":image_positions[j][1],
                                "bottom":image_positions[j][3]
                            }
                            word_position.insert(i,image)
                            # image_positions.remove()
                            del image_positions[j]
                            break
            # Extract text and keep track of its position
            # page_text=page.extract_text()
            for k in range(len(word_position)):
                extracted_text+=word_position[k]["text"] +" "
            extracted_text+=f"  \n===============  page number-{page_num+1} ================  \n"           
            # extracted_text += page_text + f"\nPage number-{page_num+1}\n"

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

def extract_only_images(uploaded_file):
    pdf_file_path = f"{uploaded_file.name}"
    mycursor = conn.cursor()
    query="select id from text_with_image_labels where file_name=%s"
    value=(pdf_file_path,)
    mycursor.execute(query,value)
    ids=mycursor.fetchall()
    if len(ids)>0:
        id=ids[0][0]
    else:
        query="insert into text_with_image_labels(file_name) values (%s)"
        value=(pdf_file_path,)
        mycursor.execute(query,value)
        query="select id from text_with_image_labels where file_name=%s;"
        value=(pdf_file_path,)
        mycursor.execute(query,value)
        ids=mycursor.fetchall()
        id=ids[0][0]
            
    query = "select pdf_images_table from text_with_image_labels where pdf_images_table=%s;"
    table_name=f"table{id}"
    value=(table_name,)
    mycursor.execute(query,value)
    answer=mycursor.fetchall()
    print(f"table {table_name} of file-{pdf_file_path} occurs this times",len(answer))
            
    if len(answer)>0:
        query=f"select image,image_label from `{table_name}`;"
        mycursor.execute(query)
        list=mycursor.fetchall()
        for image_data, image_label in list:
                
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption=image_label)
    else:
        pdf_file_path = f"{uploaded_file.name}"
        query=f"UPDATE text_With_image_labels SET pdf_images_table = %s WHERE file_name = %s"
        value=(table_name,pdf_file_path)
        mycursor.execute(query,value)
        query=f"create table `{table_name}` (id INT AUTO_INCREMENT,image blob,image_label varchar (30), primary key (id))"
        value=(table_name,)
        mycursor.execute(query)
        images=extract_images(uploaded_file)
        
        if images:
                    st.write("Extracted Images:")
                    for image_filename, image_bytes in images:
                        image = Image.open(io.BytesIO(image_bytes))
                        image_data = io.BytesIO(image_bytes).read()
                        query=f"insert into `{table_name}` (image,image_label) values(%s,%s)"
                        value=(image_data,image_filename)
                        try:
                            mycursor.execute(query,value)
                            conn.commit()
                        except mysql.connector.Error as err:
                            st.error(f"Error: {err}")
                            conn.rollback()
                            
                        
                        st.image(image, caption=image_filename)
                        st.download_button(
                        label="Download Image",
                        data=image_bytes,
                        file_name=image_filename,
                        mime="image/jpeg"
                        )
        else:
                        st.write("No images found in the PDF file.")

st.title("file analyzer")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "pdf"], accept_multiple_files=False)

if uploaded_file is not None:
    all_tables = []
    if uploaded_file.name.endswith('.csv'):
        data=pd.read_csv(uploaded_file)
        
        # st.write(data)
        analyze(data)
    
    elif uploaded_file.name.endswith('.pdf'):
        conn=mysql.connector.connect(host="localhost",password="12345678",user="root",database="files")
        if conn.is_connected():
            st.info("database connected")
        column1,column2,column3,column4,column5=st.columns(5)
        with column1:
            ex_image=st.button("Images")
        with column2:
            ex_tables=st.button("Tables")
        with column3:
            ex_question=st.button("Ask Question")
        with column4:
            image_labels=st.button("Text with image labels")
        with column5:
            image_pdf=st.button("Get the separate pdf of images")
        
        if image_pdf:
            pdf_file_path = f"{uploaded_file.name}"
            mycursor = conn.cursor()
            query="select id from text_with_image_labels where file_name=%s"
            value=(pdf_file_path,)
            mycursor.execute(query,value)
            ids=mycursor.fetchall()
            if len(ids)>0:
                id=ids[0][0]
                query="insert into text_with_image_labels(file_name) values (%s)"
                value=(pdf_file_path,)
                mycursor.execute(query,value)
                query="select id from text_with_image_labels where file_name=%s;"
                value=(pdf_file_path,)
                mycursor.execute(query,value)
                ids=mycursor.fetchall()
                id=ids[0][0]
            
            query = "select pdf_images_table from text_with_image_labels where pdf_images_table=%s;"
            table_name=f"table{id}"
            value=(table_name,)
            mycursor.execute(query,value)
            answer=mycursor.fetchall()
            print(f"table {table_name} of file-{pdf_file_path} occurs this times",len(answer))
            
            if len(answer)>0:
                query=f"select image,image_label from `{table_name}`;"
                mycursor.execute(query)
                list=mycursor.fetchall()
                images=[]
                for image_data, image_label in list:
                
                    image = Image.open(io.BytesIO(image_data))
                    pdf_image=image.convert('RGB')
                    images.append(pdf_image)
                    pdf_bytes = io.BytesIO()
                    
                    images[0].save(pdf_bytes, format='PDF', save_all=True, append_images=images[1:])
                    # pdf_bytes.seek(0)
                    
            
                    # st.image(image, caption=image_label)
                
                st.download_button(label="Download PDF", data=pdf_bytes, file_name=pdf_file_path, mime="application/pdf")
            else:
                pdf_file_path = f"{uploaded_file.name}"
                query=f"UPDATE text_With_image_labels SET pdf_images_table = %s WHERE file_name = %s"
                value=(table_name,pdf_file_path)
                mycursor.execute(query,value)
                query=f"create table `{table_name}` (id INT AUTO_INCREMENT,image blob,image_label varchar (30), primary key (id))"
                value=(table_name,)
                mycursor.execute(query)
                images=extract_images(uploaded_file)
        
                if images:
                            st.write("Extracted Images:")
                            images=[]
                            for image_filename, image_bytes in images:
                                image = Image.open(io.BytesIO(image_bytes))
                                pdf_image=image.convert("RGB")
                                images.append(pdf_image)
                                image_data = io.BytesIO(image_bytes).read()
                                query=f"insert into `{table_name}` (image,image_label) values(%s,%s)"
                                value=(image_data,image_filename)
                                try:
                                    mycursor.execute(query,value)
                                    conn.commit()
                                except mysql.connector.Error as err:
                                    st.error(f"Error: {err}")
                                    conn.rollback()
                            
                                pdf_bytes = io.BytesIO()
                    
                                images[0].save(pdf_bytes, format='PDF', save_all=True, append_images=images[1:])
                                # st.image(image, caption=image_filename)
                                # st.download_button(
                                # label="Download Image",
                                # data=image_bytes,
                                # file_name=image_filename,
                                # mime="image/jpeg"
                                # )
                            st.download_button(label="Download PDF", data=pdf_bytes, file_name=pdf_file_path, mime="application/pdf")      
                else:
                                st.write("No images found in the PDF file.")
        
                if image_labels:
                    mycursor = conn.cursor()
            
                    query = """
                            select file_name from text_with_image_labels where file_name=(%s);
                        """
                    pdf_file_path = f"{uploaded_file.name}"
                    st.info(pdf_file_path)
                    mycursor.execute(query, (pdf_file_path,))
                    answer=mycursor.fetchall()
                    print(len(answer))
                    if len(answer)>0:
                        query="select file_text from text_with_image_labels where file_name=(%s)"
                        value=(pdf_file_path,)
                        mycursor.execute(query,value)
                        file_text=mycursor.fetchone()
                        st.write(file_text[0])
                   
                    else:
                
                        text=extract_text_by_replacing_images(uploaded_file)
                        mycursor.execute("select count(distinct(id)) from text_with_image_labels")
                        count=mycursor.fetchone()
                        id=count[0]+1
                        query="insert into text_with_image_labels(id,file_name,file_text) values (%s,%s,%s)"
                        values=(id,pdf_file_path,text)
                        mycursor.execute(query,values)
                        conn.commit()
                
                        st.write(text)
            
            
        if ex_question:
            chat=st.chat_input("Type your message here...")
            text=extract_textpdf(uploaded_file)
        
        if ex_image:
            extract_only_images(uploaded_file)
        
        if ex_tables:
            #   text=extract_textpdf(uploaded_file)
              tables=extract_tables(uploaded_file) 
        
           
        
        
        
    
