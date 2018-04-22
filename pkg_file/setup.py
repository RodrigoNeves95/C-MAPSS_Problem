from setuptools import setup, find_packages

setup(name='turbofan_pkg',
      version='0.0',
      author='Rodrigo Neves',
      author_email='rodrigo.neves@jungle.ai',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)