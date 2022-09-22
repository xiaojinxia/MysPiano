import os
mypaths = os.getenv('PATH').split(';')  # replace 'PATH' by 'your search path' if needed.

for i in mypaths:
    if i.find('python'):
        print(i)
