function deploy() {
    # install tools
    pip install wheel twine
    
    # create wheel and win installer
    python setup.py bdist_wheel bdist_wininst
    
    # upload to pypi with twine
    twine upload -i $env:myuser -p $env:mypass dist/*
}

new_tag = ($env:APPVEYOR_REPO_TAG -eq true)
new_tag = true # temporarily enable for all commits

if (new_tag) {
	deploy
}