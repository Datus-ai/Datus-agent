# Use Python 3.12 as base image from Aliyun mirror
FROM datusai/datus-agent:0.1.12

RUN pip uninstall -y datus-agent

RUN pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ datus-agent==0.1.13


WORKDIR /app

# Keep the container running
CMD ["tail", "-f", "/dev/null"]


