conda install --yes -q anaconda-client jinja2
cd %PYTHON_MINICONDA%\conda-bld
dir /s /b %PACKAGENAME%-dev-*.tar.bz2 > files.txt
for /F %%filename in (files.txt) do (
    echo "uploading file %%~filename"
    anaconda -t %BINSTAR_TOKEN% upload --force -u %ORGNAME% -p %PACKAGENAME%-dev %%~filename
)
