from distutils.core import setup, Extension

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
              '_wapiti',
              sources=['wapiti/src/bcd.c', 'wapiti/src/lbfgs.c', 'wapiti/src/pattern.c', 'wapiti/src/reader.c', 'wapiti/src/thread.c', 'wapiti/src/wapiti.c', 
              'wapiti/src/decoder.c', 'wapiti/src/model.c', 'wapiti/src/progress.c', 'wapiti/src/rprop.c', 'wapiti/src/tools.c', 'wapiti/src/gradient.c', 
              'wapiti/src/options.c', 'wapiti/src/quark.c', 'wapiti/src/sgdl1.c', 'wapiti/src/vmath.c', 'libwapiti/src/api.c'],
              include_dirs=['wapiti/src', 'libwapiti/src'],
              extra_compile_args=['-std=c99'],
              extra_link_args=['-lm', '-lpthread'],
          )
      ],
    )
