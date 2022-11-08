echo "Remove .DS_Store files"
rm -rf .DS_Store
echo "Updating requirements.txt"
source wildfire_env/bin/activate
pip3 freeze > requirements.txt
echo "Push routine: \n\
git add .\n\
git commit -m <commit name>\n\
git push"