from setuptools import setup


def parse_requirements(path: str = 'requirements.txt'):
    with open(path) as fp:
        return fp.read().strip().split()


setup(
    name='perceiver-io-pytorch',
    version='0.1.0',
    packages=['perceiver_io'],
    package_dir={'': 'src'},
    url='https://github.com/esceptico/perceiver-io',
    license='MIT',
    author='Timur Ganiev',
    author_email='ganiev.tmr@gmail.com',
    description='Unofficial Perceiver IO implementation',
    install_requires=parse_requirements()
)
