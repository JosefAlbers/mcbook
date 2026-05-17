from setuptools import find_packages, setup

requirements = [
    "rcrlm",
]

# extras_require = {
#     'cuda': ['mlx[cuda]'],
#     'cpu':  ['mlx[cpu]'],
#     'no_mlx': [],
# }

setup(
    name='mcbook',
    url='https://github.com/JosefAlbers/mcbook',
    packages=find_packages(),
    version='0.0.1a0',
    readme="README.md",
    author_email="albersj66@gmail.com",
    description="A graph-based knowledge management system for LLM-driven research, technical editing, and automated peer-review workflows.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="J Joe",
    license="Apache-2.0",
    # python_requires=">=3.12.8",
    install_requires=requirements,
    # extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "mcb=mcbook.mcbook:test",
        ],
    },
)
