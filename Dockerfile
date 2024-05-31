ARG VLLM_VERSION=v0.4.2
ARG BASE_IMAGE=vllm/vllm-openai:${VLLM_VERSION}
FROM ${BASE_IMAGE}
LABEL maintainer "Andrew Naylor <anaylor@lbl.gov>"

# Update and Install required libs
RUN apt-get update -y 
RUN pip install --no-cache-dir \
        faiss-cpu~=1.8.0 \
        sentence-transformers~=2.5.0 \
        whoosh \
        gensim \
        sfapi_client~=0.0.6 \
        "scipy<1.13"

#Add a user
RUN groupadd -g 10001 chatbot \
        && useradd -u 10000 -g chatbot chatbot \
        && mkdir /home/chatbot \
        && chown -R chatbot:chatbot /home/chatbot

USER chatbot:chatbot

#Start the chatbot
# ENTRYPOINT ["python3", "-m", "chatbot"]