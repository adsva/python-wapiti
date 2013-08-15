import os
from distutils.core import setup, Extension


wapiti_src_c = ['Wapiti/src/' + i for i in filter(
    lambda x:x.endswith("c"), os.listdir("Wapiti/src"))]
wapiti_src_c.append('libwapiti/src/api.c')

setup(name='python wapiti bindings',
      version='0.1',
      py_modules=['wapiti'],
      description="Python bindings for libwapiti",
      long_description="",
      author="Adam Svanberg",
      author_email="asvanberg@gmail.com",
      ext_modules=[
          Extension(
              'libwapiti',
              sources=wapiti_src_c,
              extra_compile_args=['-std=c99'],
              include_dirs=["Wapiti/src", "libwapiti"],
              extra_link_args=['-lm', '-lpthread'],
          )
      ],)
