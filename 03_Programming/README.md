# 3_ Programming

## 1_ Python Basics

### About Python

Python is a high-level programming langage. I can be used in a wide range of works.

Commonly used in data-science, [Python](https://www.python.org/)  has a huge set of libraries, helpful to quickly do something.

Most of informatics systems already support Python, without installing anything.

### Execute a script

* Download the .py file on your computer
* Make it executable (_chmod +x file.py_ on Linux)
* Open a terminal and go to the directory containing the python file
* _python file.py_ to run with Python2 or _python3 file.py_ with Python3

## 2_ Working in excel

## 3_ R setup / R studio

### About R

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

## 4_ R basics

R is an open source programming language and software environment for statistical computing and graphics that is supported by the R Foundation for Statistical Computing.

The R language is widely used among statisticians and data miners for developing statistical software and data analysis.

Polls, surveys of data miners, and studies of scholarly literature databases show that R's popularity has increased substantially in recent years.

## 5_ Expressions

## 6_ Variables

## 7_ IBM SPSS

## 8_ Rapid Miner

## 9_ Vectors

## 10_ Matrices

## 11_ Arrays

## 12_ Factors

## 13_ Lists

## 14_ Data frames

## 15_ Reading CSV data

CSV is a format of __tabular data__ comonly used in data science. Most of structured data will come in such a format.

To __open a CSV file__ in Python, just open the file as usual :
 
 raw_file = open('file.csv', 'r')
 
* 'r': Reading, no modification on the file is possible
* 'w': Writing, every modification will erease the file
* 'a': Adding, every modification will be made at the end of the file

### How to read it ?

Most of the time, you will parse this file line by line and do whatever you want on this line. If you want to store data to use them later, build lists or dictionnaries.

To read such a file row by row, you can use :

* Python [library csv](https://docs.python.org/3/library/csv.html)
* Python [function open](https://docs.python.org/2/library/functions.html#open)

## 16_ Reading raw data

## 17_ Subsetting data

## 18_ Manipulate data frames

## 19_ Functions

A function is helpful to execute redondant actions.

First, define the function:

 def MyFunction(number):
  """This function will multiply a number by 9"""
  number = number * 9
  return number

## 20_ Factor analysis

## 21_ Install PKGS

Python actually has two mainly used distributions. Python2 and python3.

### Install pip

Pip is a library manager for Python. Thus, you can easily install most of the packages with a one-line command. To install pip, just go to a terminal and do:
 
 # __python2__
 sudo apt-get install python-pip
 # __python3__
 sudo apt-get install python3-pip
 
You can then install a library with [pip](https://pypi.python.org/pypi/pip?) via a terminal doing:

 # __python2__ 
 sudo pip install [PCKG_NAME]
 # __python3__ 
 sudo pip3 install [PCKG_NAME]

You also can install it directly from the core (see 21_install_pkgs.py)
