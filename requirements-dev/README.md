This directory contains `*.in` files that will be used (along with this
project's main `requirements.in` file) by [pip-tools] to compile the
`requirements-dev.txt` file. Please do not pin any requirements here and instead
provide constrained versions. The pinned requirements will automatically be
generated and updated via the `make update-requirements` command.

[pip-tools]: https://github.com/jazzband/pip-tools
