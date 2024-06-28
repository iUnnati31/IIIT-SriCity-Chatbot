import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv

load_dotenv()

groq_api_key=os.getenv('GROQ_API_KEY')

st.title("KNOW  ABOUT IIIT SRICITY 🏢")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=OllamaEmbeddings(model="llama3")
        st.session_state.loader=WebBaseLoader(web_paths=("https://www.iiits.ac.in/",                                                       
"https://www.iiits.ac.in/research/workshop/", "https://www.iiits.ac.in/research/research-centers/center-for-smart-cities/", "https://www.iiits.ac.in/research/research-centers/design-and-innovation-center-mhrd/", "https://www.iiits.ac.in/research/research-groups/computer-vision-group/", "https://www.iiits.ac.in/research/research-groups/nextgen-communications/", "https://www.iiits.ac.in/research/research-groups/smart-transportation-group/",
"https://www.iiits.ac.in/research/publications/journals/", "https://www.iiits.ac.in/research/publications/conferences/", "https://www.iiits.ac.in/research/intellectual-property/patent-filings/", "https://www.iiits.ac.in/research/research-resources/e-shodh-sindhu/", "https://www.iiits.ac.in/research/research-resources/e-vidwan/",
"https://www.iiits.ac.in/research/research-scholars/", "https://www.iiits.ac.in/research/conference/", "https://www.iiits.ac.in/innovation-tbi/nisp-2/",  "https://www.iiits.ac.in/innovation-tbi/institute-innovation-cell/", "https://www.iiits.ac.in/innovation-tbi/gyan-circle-venture-tide2/about/", "https://www.iiits.ac.in/innovation-tbi/gyan-circle-venture-tide2/what-we-do/",
"https://www.iiits.ac.in/innovation-tbi/gyan-circle-venture-tide2/advisory-board/", "https://www.iiits.ac.in/innovation-tbi/gyan-circle-venture-tide2/current-startups/", "https://www.iiits.ac.in/innovation-tbi/gyan-circle-venture-tide2/selection-mentoring-committee/", "https://www.iiits.ac.in/innovation-tbi/gyan-circle-venture-tide2/announcements/", "https://www.iiits.ac.in/innovation-tbi/gyan-circle-venture-tide2/contact-us/",
"https://www.iiits.ac.in/contact-us/academics/", "https://www.iiits.ac.in/contact-us/admissions/", "https://www.iiits.ac.in/contact-us/industry-collaboration/", "https://www.iiits.ac.in/contact-us/research-collaboration/", "https://www.iiits.ac.in/contact-us/international-collaboration/", "https://www.iiits.ac.in/contact-us/placement-team/", "https://www.iiits.ac.in/contact-us/press-and-media/",
"https://www.iiits.ac.in/contact-us/gallary/convocation/", "https://www.iiits.ac.in/contact-us/gallary/acacemic-activities/", "https://www.iiits.ac.in/student-development-council/", "https://www.iiits.ac.in/student-life-council/", "https://www.iiits.ac.in/student-life/marketing-team/", "https://www.iiits.ac.in/student-life/design-team/", "https://www.iiits.ac.in/student-life/unnat-bharat-abhiyan/", 
"https://www.iiits.ac.in/student-life/ebsb/", "https://www.iiits.ac.in/student-life/sports/", "https://www.iiits.ac.in/student-life/gallery/events-iiit-sri-city/", "https://www.iiits.ac.in/student-life/gallery/videos/", "https://www.iiits.ac.in/ai-and-ml-club/", "https://www.iiits.ac.in/cyber-security-club-enigma/", "https://www.iiits.ac.in/data-science-club/", "https://www.iiits.ac.in/developer-club-gdsc/",
"https://www.iiits.ac.in/e-cell-club/", "https://www.iiits.ac.in/internet-of-things-club/", "https://www.iiits.ac.in/programming-club/", "https://www.iiits.ac.in/project-club-iota/", "https://www.iiits.ac.in/dance-club/", "https://www.iiits.ac.in/film-club/", "https://www.iiits.ac.in/music-club/", "https://www.iiits.ac.in/photography-club/", "https://www.iiits.ac.in/nirvana-club/", "https://www.iiits.ac.in/fitness-club/",
"https://www.iiits.ac.in/keynote/",  "https://www.iiits.ac.in/home/about-iiit-sri-city/", "https://www.iiits.ac.in/home/strategic-location/", "https://www.iiits.ac.in/home/mission-and-vision/",
"https://www.iiits.ac.in/home/partners/government-of-andhra-pradesh/", "https://www.iiits.ac.in/home/partners/sri-city-pvt-limited/", "https://www.iiits.ac.in/home/governance/chairman/",
"https://www.iiits.ac.in/home/governance/board-of-governors/", "https://www.iiits.ac.in/home/governance/finance-committee/", "https://www.iiits.ac.in/home/governance/building-and-works-committee/",
"https://www.iiits.ac.in/home/governance/senate/", "https://www.iiits.ac.in/home/administration/director/", "https://www.iiits.ac.in/home/governance/former-directors/", "https://www.iiits.ac.in/home/administration/registrar/",
"https://www.iiits.ac.in/home/administration/academic-administration/", "https://www.iiits.ac.in/home/administration/research-development/", "https://www.iiits.ac.in/home/administration/industry-engagement-international-relations/",
"https://www.iiits.ac.in/home/infrastructure/academics/", "https://www.iiits.ac.in/home/infrastructure/hostels-dining/", "https://www.iiits.ac.in/home/infrastructure/technology-business-incubator/",
"https://www.iiits.ac.in/home/infrastructure/access-to-sri-city-social-infrastructure/", "https://www.iiits.ac.in/careersiiits/faculty/", "https://www.iiits.ac.in/careersiiits/staff/",
"https://www.iiits.ac.in/careersiiits/jrf-srf-project-positions/", "https://www.iiits.ac.in/people/lecturers/", "https://www.iiits.ac.in/people/visiting-faculty/", "https://www.iiits.ac.in/people/prospective-faculty/",
"https://www.iiits.ac.in/people/students/",  "https://www.iiits.ac.in/semester-long-project/", "https://www.iiits.ac.in/career-center/summer-internship/",
"https://www.iiits.ac.in/career-center/higher-study-support/", "https://www.iiits.ac.in/career-center/mentoring-programme/impact/", "https://www.iiits.ac.in/career-center/national-career-service-mhrd/", "https://www.iiits.ac.in/admissions/b-tech-programme/josaa-csab/",
"https://www.iiits.ac.in/admissions/b-tech-programme/dasa/", "https://www.iiits.ac.in/admissions/b-tech-programme/study-in-india/", "https://www.iiits.ac.in/admissions/m-tech-programme/aiml/", "https://www.iiits.ac.in/admissions/m-tech-programme/cyber-security/",
"https://www.iiits.ac.in/admissions/ph-d-admissions-full-time-monsoon-2024/", "https://www.iiits.ac.in/admissions/ph-d-admissions-part-time-monsoon-2024/", "https://www.iiits.ac.in/admissions/faqs/", "https://www.iiits.ac.in/academics/b-tech-programme/artificial-intelligence-and-data-science/b-tech-ai-ds-curriculum/", 
"https://www.iiits.ac.in/academics/b-tech-programme/electronics-communication-engineering/curriculum/", "https://www.iiits.ac.in/academics/b-tech-programme/regulations/ece-specialization/cyber-physical-systems/", 
"https://www.iiits.ac.in/academics/b-tech-programme/regulations/ece-specialization/next-generation-wireless-communication/", "https://www.iiits.ac.in/academics/b-tech-programme/regulations/", "https://www.iiits.ac.in/academics/b-tech-programme/fee-structure/",
"https://www.iiits.ac.in/academics/m-tech-programme/ai-machine-learning/progarmme-advisory-group/", "https://www.iiits.ac.in/academics/m-tech-programme/ai-machine-learning/about/", "https://www.iiits.ac.in/academics/m-tech-programme/cyber-security/programme-advisory-group/",
"https://www.iiits.ac.in/academics/m-tech-programme/ai-machine-learning/eligibility/", "https://www.iiits.ac.in/academics/m-tech-programme/ai-machine-learning/curriculum/", "https://www.iiits.ac.in/academics/m-tech-programme/cyber-security/about/",
"https://www.iiits.ac.in/academics/m-tech-programme/cyber-security/eligibility/", "https://www.iiits.ac.in/academics/m-tech-programme/cyber-security/curriculum/", "https://www.iiits.ac.in/academics/m-tech-programme/regulations/",
"https://www.iiits.ac.in/academics/m-tech-programme/fee-structure/", "https://www.iiits.ac.in/academics/ph-d-programme/full-time/", "https://www.iiits.ac.in/academics/ph-d-programme/part-time/", "https://www.iiits.ac.in/academics/ph-d-programme/fees-structure/",
"https://www.iiits.ac.in/iiits-content/uploads/2021/07/PhD-Regulations-MinorRevisions-IIIT-Sri-City.pdf", "https://www.iiits.ac.in/academics/almanac-time-table/", "https://www.iiits.ac.in/academics/library/", "https://www.iiits.ac.in/academics/academic-collaboration/workshops/",
"https://www.iiits.ac.in/academics/student-academic-council/", "https://www.iiits.ac.in/academics/academic-collaboration/nasscom-dsci/", "https://www.iiits.ac.in/academics/academic-collaboration/mou-moa/the-art-of-living/", 
"https://www.iiits.ac.in/academics/digital-learning/swayam/", "https://www.iiits.ac.in/academics/digital-learning/nptel/", "https://www.iiits.ac.in/academics/digital-learning/mooc/", "https://www.iiits.ac.in/academics/digital-learning/national-digital-library/", "https://www.iiits.ac.in/academics/b-tech-programme/computer-science-engineering/curriculum/cse-specializations/specialization-in-data-science/",
"https://www.iiits.ac.in/academics/b-tech-programme/computer-science-engineering/curriculum/cse-specializations/specializations-in-ai-ml/", "https://www.iiits.ac.in/academics/b-tech-programme/computer-science-engineering/curriculum/cse-specializations/specialization-in-cybersecurity/",
"https://www.iiits.ac.in/academics/b-tech-programme/computer-science-engineering/full-stack-development/", "https://www.iiits.ac.in/people/alumni/", "https://www.iiits.ac.in/campus-placement/", "https://www.iiits.ac.in/admissions/financial-aid/", "https://www.iiits.ac.in/innovation-tbi/e-cell-iiit-sri-city/",
"https://www.iiits.ac.in/academics/digital-learning/talk-to-teacher/", "https://www.iiits.ac.in/academics/digital-learning/virtual-labs/", "https://www.iiits.ac.in/academics/b-tech-programme/computer-science-engineering/curriculum/",  "https://www.iiits.ac.in/research/sponsored-projects/", "https://www.iiits.ac.in/people/regular-faculty/"
                                                         ),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer())) ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector Ollama embeddings
        st.session_state.initialized = True

prompt1=st.text_input("Enter Your Question:")

if st.button("Get Info"):
    vector_embedding()
    st.write("DB Is Ready")

import time

if prompt1:
    # if "vectors" not in st.session_state:
    #     st.error("Please click 'Get Info' to initialize the database before asking a question.")
    # else:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        start=time.process_time()
        response=retrieval_chain.invoke({'input':prompt1})
        print("Response time :",time.process_time()-start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
             # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("-------------------------------")