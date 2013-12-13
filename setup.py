import os
from subprocess import check_call
from setuptools import setup, find_packages
from setuptools.extension import Extension

here = os.path.abspath(os.path.dirname(__file__))


def update_submodule():
    if os.path.exists(os.path.join(here, '.git')):
        os.chdir(here)
        check_call(['git', 'submodule', 'init'])
        check_call(['git', 'submodule', 'update'])
update_submodule()

wapiti_src_c = ['cwapiti/src/' + i for i in filter(
    lambda x:x.endswith("c"), os.listdir("cwapiti/src"))]
wapiti_src_c.append('libwapiti/src/api.c')


setup(name='libwapiti',
      version='0.2',
      packages=find_packages(),
      description="Python bindings for libwapiti",
      long_description="",
      author="Adam Svanberg",
      author_email="asvanberg@gmail.com",
      include_package_data=True,
      ext_modules=[
          Extension(
              'libwapiti',
              sources=wapiti_src_c,
              extra_compile_args=['-std=c99'],
              include_dirs=["cwapiti/src", "libwapiti"],
              extra_link_args=['-lm', '-lpthread'],
          )
      ],
      scripts=['scripts/wapiti.py',],
      install_requires=['six',],
)
