import re

text = "I disapprove of what you say, but I will defend to the death your right to say it."
itext = re.finditer(r'\bt\w+\b', text)
for i in itext: print(i.group())
