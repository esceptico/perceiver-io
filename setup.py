from setuptools import setup


setup(
    name='perceiver-io-pytorch',
    version='0.1.5',
    packages=['perceiver_io'],
    package_dir={'': 'src'},
    url='https://github.com/esceptico/perceiver-io',
    license='MIT',
    author='Timur Ganiev',
    author_email='ganiev.tmr@gmail.com',
    description='Unofficial Perceiver IO implementation',
    install_requires=['torch']
)
