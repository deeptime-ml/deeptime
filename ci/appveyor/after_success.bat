% Deploy to binstar
conda install --yes anaconda-client jinja2
cd %PYTHON%\conda-bld
for %%filename in (*\%PACKAGENAME%-dev-*.tar.bz2) do (
    echo "removing file %%~filename"
    anaconda -t %BINSTAR_TOKEN% remove --force %ORGNAME%\%PACKAGENAME%-dev\%%~filename
    anaconda -t %BINSTAR_TOKEN% upload --force -u %ORGNAME% -p %PACKAGENAME%-dev %%~filename
)
