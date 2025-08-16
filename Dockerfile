# Use Python 3.12 as base image from Aliyun mirror
FROM luochen2025/datus-agent:0.1.5a4

RUN pip uninstall -y datus-agent

RUN pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ datus-agent==0.1.8


WORKDIR /app

# Keep the container running
CMD ["tail", "-f", "/dev/null"]


