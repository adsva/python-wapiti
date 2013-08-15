import os
from distutils.core import setup, Extension


wapiti_src_c = ['Wapiti/src/' + i for i in filter(
    lambda x:x.endswith("c"), os.listdir("Wapiti/src"))]
wapiti_src_h = ['Wapiti/src/' + i for i in filter(
    lambda x:x.endswith("h"), os.listdir("Wapiti/src"))]

setup(name='python wapiti bindings',
      version='0.1',
      py_modules=['wapiti'],
      description="Python bindings for libwapiti",
      long_description="",
      author="Adam Svanberg",
      author_email="asvanberg@gmail.com",
      packages=['wapiti'],
      ext_modules=[
          Extension(
              'libwapiti',
              wapiti_src_c,
              extra_compile_args=['-std=c99'],
              depends=wapiti_src_h,
              include_dirs=["wapiti/src", "libwapiti"],
              extra_link_args=['-lm', '-lpthread'],
          )
      ],)
