from setuptools import setup, find_packages
setup(name = "e-commerce-bot", version="0.0.1", author = "manish" ,
      author_email = "manishsurabhi23@gmail.com",
      packages=find_packages(),
      install_requires = ['langchain-astradb','langchain'])