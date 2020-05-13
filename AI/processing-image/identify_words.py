#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytesseract as pytesseract
from PIL import Image

print pytesseract.image_to_string(Image.open('./data/Xnip2020-03-14_14-13-49.jpg'))
