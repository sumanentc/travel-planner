FROM rasa/rasa:3.5.2

USER root

WORKDIR /app
# Copy any additional custom requirements, if necessary (uncomment next line)
COPY requirements.txt ./

RUN apt-get update
#RUN apt update

# Install extra requirements for actions code, if necessary (uncomment next line)
RUN pip install -r requirements.txt

EXPOSE 5005

COPY . /app
COPY ./data /app/data
#RUN pip install --upgrade pip --user
RUN  python -m rasa train

RUN chgrp -R 0 /app && chmod -R g=u /app

# Switch back to a non-root user
USER 1001

#VOLUME /app
#VOLUME /app/data
#VOLUME /app/models


CMD [ "run","-m","/app/models","--enable-api","--cors","*","--debug"]