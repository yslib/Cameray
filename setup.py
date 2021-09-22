from setuptools import setup, find_packages

setup(  name='cameray',
        install_requires=[
            'networkx',
            'taichi',
            'numpy',
            'dearpygui'
        ],
        setup_requires=[],
        python_requires='>=3.6, <=3.9',
        version='0.0.2',

        author='Shuoliu Yang',

        author_email='visysl@outlook.com',

        url='https://github.com/yslib/Cameray',

        package_dir={'':'src'},
        packages=find_packages(where='src'),

        entry_points={
            'console_scripts':[
                'cameray = gui.app:main'
            ]

        },

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
