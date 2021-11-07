import re

text = "I disapprove of what you say, but I will defend to the death your right to say it."
itext = re.finditer(r'\bt\w+\b', text)
for i in itext: print(i.group())

line='1853   1    8.4     2.7       4    62.8'
itext = re.split(r' +', line)
print(itext)