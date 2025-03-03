# download the data
curl -L -o data.zip https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1

# unzip the data
unzip data -d data

# pretty save the data
cat "data/data/dev.json" | python -m json.tool > "dev.json"

# remove the data
rm data.zip
rm -r data/