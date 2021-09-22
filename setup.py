from distutils.core import setup


setup(  name='cameray',
        install_requires=[
            'networkx',
            'taichi',
            'numpy'
        ],
        setup_requires=[],
        tests_require=[],
        extra_require={},
        python_requires='>=3.6, <=3.9',
        version='0.0.1',

        author='Shuoliu Yang',

        author_email='visysl@outlook.com',

        url='https://github.com/yslib/Cameray',

        packages=['src','src.base','src.cameray','src.gui'],

        classifiers = [
        # https://pypi.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.8',
        ]
        )
