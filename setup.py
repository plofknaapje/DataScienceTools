import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='linear-algebra-robin',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='',
    license='MIT',
    author='Robin Verhoef',
    author_email='',
    description='Small linear algebra module',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
