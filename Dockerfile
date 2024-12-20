#Using python
FROM python:3.10-slim
# Using Layered approach for the installation of requirements
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
#Copy files to your container
COPY . ./
#Running your APP and doing some PORT Forwarding
CMD gunicorn -b 0.0.0.0:8072 app:server