# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:23:52 2018

@author: c14238a
"""

# This exception should be raised if Article.dowonloadHTML hasn't been invoked
class DownloadError(Exception):
    pass

# This exception should be raised if the HTML tag is not found
class NotFoundError(Exception):
    pass