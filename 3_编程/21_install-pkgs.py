#!/usr/bin/python3

# Import pip
import pip

# Build an installation function
def install(package):
    pip.main(['install', package])

# Execution
install('fooBAR')
