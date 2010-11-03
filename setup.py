from distutils.core import setup
import wapiti
setup(name='foo',
      version=wapiti.__version__,
      py_modules=['wapiti'],
      description="Python bindings for libwapiti",
      long_description=wapiti.__doc__,
      author="Adam Svanberg",
      author_email="asvanberg@gmail.com",
      )
