TVAR="$(grep -r -H 'sys.path.append')"
VAREXIT=0
while IFS= read -r line; do
    if [[ $line == .github/* ]]; 
    then
        echo "sys.path.append found in .github directory, ignoring"
    else
        echo "sys.path.append found in the following files:"
        echo "$line"
        VAREXIT=1
    fi
done <<< "$TVAR"

if [ "$VAREXIT" -eq 1 ]; then
    exit 1
else
    exit 0
fi