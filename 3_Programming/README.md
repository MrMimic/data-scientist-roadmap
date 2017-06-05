# 3_ Programming

## 1_ Python Basics

### About

Python is a high-level programming langage. I can be used in a wide range of works.

Commonly used in data-science, [Python](https://www.python.org/)  has a huge set of libraries, helpful to quickly do something.

Most of informatics systems already support Python, without installing anything.

### Execute a script

* Download the .py file on your computer
* Make it executable (_chmod +x file.py_ on Linux)
* Open a terminal and go to the directory containing the python file
* _python file.py_ to run with Python2 or _python3 file.py_ with Python3

### Install a library

Python actually has two mainly used distributions. Python2 and python3.

You can install a library with [pip](https://pypi.python.org/pypi/pip?).

	# __python2__ 
	sudo pip install [PCKG_NAME]
	# __python3__ 
	sudo pip3 install [PCKG_NAME]



## 2_ Working in excel

## 3_ R setup / R studio

### About

R is a programming language specialized in statistics and mathematical visualizations.

I can be used withs manually created scripts launched in the terminal, or directly in the R console.

### Installation

#### Linux

	sudo apt-get install r-base
	
	sudo apt-get install r-base-dev

#### Windows

Download the .exe setup available on [CRAN](https://cran.rstudio.com/bin/windows/base/) website.

### R-studio

Rstudio is a graphical interface for R. It is available for free on [their website](https://www.rstudio.com/products/rstudio/download/).

This interface is divided in 4 main areas :

![rstudio](https://owi.usgs.gov/R/training-curriculum/intro-curriculum/static/img/rstudio.png)

* The top left is the script you are working on (highlight code you want to execute and press Ctrl + Enter)
* The bottom left is the console to instant-execute some lines of codes
* The top right is showing your environment (variables, history, ...)
* The bottom right show figures you plotted, packages, help ... The result of code execution
